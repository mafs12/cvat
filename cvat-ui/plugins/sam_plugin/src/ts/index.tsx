// Copyright (C) 2023 CVAT.ai Corporation
//
// SPDX-License-Identifier: MIT

import { InferenceSession, Tensor } from 'onnxruntime-web';
import { LRUCache } from 'lru-cache';
import { PluginEntryPoint, APIWrapperEnterOptions, ComponentBuilder } from 'components/plugins-entrypoint';
import lodash from 'lodash';

interface SAMPlugin {
    name: string;
    description: string;
    cvat: {
        lambda: {
            call: {
                enter: (
                    plugin: SAMPlugin,
                    taskID: number,
                    model: any,
                    args: any,
                ) => Promise<null | APIWrapperEnterOptions>;
                leave: (
                    plugin: SAMPlugin,
                    result: any,
                    taskID: number,
                    model: any,
                    args: any,
                ) => Promise<any>;
            };
        };
        jobs: {
            get: {
                leave: (
                    plugin: SAMPlugin,
                    results: any[],
                    query: { jobID?: number }
                ) => Promise<any>;
            };
        };
    };
    data: {
        core: any;
        jobs: Record<number, any>;
        modelID: string;
        modelURL: string;
        embeddings: LRUCache<string, Tensor>;
        lowResMasks: LRUCache<string, Tensor>;
        session: InferenceSession | null;
    };
    callbacks: {
        onStatusChange: ((status: string) => void) | null;
    };
}

interface ONNXInput {
    image_embeddings: Tensor;
    point_coords: Tensor;
    point_labels: Tensor;
    orig_im_size: Tensor;
    mask_input: Tensor;
    has_mask_input: Tensor;
    readonly [name: string]: Tensor;
}

interface ClickType {
    clickType: -1 | 0 | 1,
    height: number | null,
    width: number | null,
    x: number,
    y: number,
}

function getModelScale(w: number, h: number): number {
    // Input images to SAM must be resized so the longest side is 1024
    const LONG_SIDE_LENGTH = 1024;
    const samScale = LONG_SIDE_LENGTH / Math.max(h, w);
    return samScale;
}

function modelData(
    {
        clicks, tensor, modelScale, maskInput,
    }: {
        clicks: ClickType[];
        tensor: Tensor;
        modelScale: { height: number; width: number; samScale: number };
        maskInput: Tensor | null;
    },
): ONNXInput {
    const imageEmbedding = tensor;

    const n = clicks.length;
    // If there is no box input, a single padding point with
    // label -1 and coordinates (0.0, 0.0) should be concatenated
    // so initialize the array to support (n + 1) points.
    const pointCoords = new Float32Array(2 * (n + 1));
    const pointLabels = new Float32Array(n + 1);

    // Add clicks and scale to what SAM expects
    for (let i = 0; i < n; i++) {
        pointCoords[2 * i] = clicks[i].x * modelScale.samScale;
        pointCoords[2 * i + 1] = clicks[i].y * modelScale.samScale;
        pointLabels[i] = clicks[i].clickType;
    }

    // Add in the extra point/label when only clicks and no box
    // The extra point is at (0, 0) with label -1
    pointCoords[2 * n] = 0.0;
    pointCoords[2 * n + 1] = 0.0;
    pointLabels[n] = -1.0;

    // Create the tensor
    const pointCoordsTensor = new Tensor('float32', pointCoords, [1, n + 1, 2]);
    const pointLabelsTensor = new Tensor('float32', pointLabels, [1, n + 1]);
    const imageSizeTensor = new Tensor('float32', [
        modelScale.height,
        modelScale.width,
    ]);

    const prevMask = maskInput ||
        new Tensor('float32', new Float32Array(256 * 256), [1, 1, 256, 256]);
    const hasMaskInput = new Tensor('float32', [maskInput ? 1 : 0]);

    return {
        image_embeddings: imageEmbedding,
        point_coords: pointCoordsTensor,
        point_labels: pointLabelsTensor,
        orig_im_size: imageSizeTensor,
        mask_input: prevMask,
        has_mask_input: hasMaskInput,
    };
}

const samPlugin: SAMPlugin = {
    name: 'Segment Anything',
    description: 'Plugin handles non-default SAM serverless function output',
    cvat: {
        jobs: {
            get: {
                async leave(
                    plugin: SAMPlugin,
                    results: any[],
                    query: { jobID?: number },
                ): Promise<any> {
                    if (typeof query.jobID === 'number') {
                        [plugin.data.jobs[query.jobID]] = results;
                    }
                    return results;
                },
            },
        },
        lambda: {
            call: {
                async enter(
                    plugin: SAMPlugin,
                    taskID: number,
                    model: any, { frame }: { frame: number },
                ): Promise<null | APIWrapperEnterOptions> {
                    if (model.id === plugin.data.modelID) {
                        if (!plugin.data.session) {
                            throw new Error('SAM plugin is not ready, session was not initialized');
                        }

                        const key = `${taskID}_${frame}`;
                        if (plugin.data.embeddings.has(key)) {
                            return { preventMethodCall: true };
                        }
                    }

                    return null;
                },

                async leave(
                    plugin: SAMPlugin,
                    result: any,
                    taskID: number,
                    model: any,
                    { frame, pos_points, neg_points }: {
                        frame: number, pos_points: number[][], neg_points: number[][],
                    },
                ): Promise<
                    {
                        mask: number[][];
                        bounds: [number, number, number, number];
                    }> {
                    if (model.id !== plugin.data.modelID) {
                        return result;
                    }

                    const job = Object.values(plugin.data.jobs)
                        .find((_job) => _job.taskId === taskID);
                    if (!job) {
                        throw new Error('Could not find a job corresponding to the request');
                    }

                    const { height: imHeight, width: imWidth } = await job.frames.get(frame);
                    const key = `${taskID}_${frame}`;

                    if (result) {
                        const bin = window.atob(result.blob);
                        const uint8Array = new Uint8Array(bin.length);
                        for (let i = 0; i < bin.length; i++) {
                            uint8Array[i] = bin.charCodeAt(i);
                        }
                        const float32Arr = new Float32Array(uint8Array.buffer);
                        plugin.data.embeddings.set(key, new Tensor('float32', float32Arr, [1, 256, 64, 64]));
                    }

                    const modelScale = {
                        width: imWidth,
                        height: imHeight,
                        samScale: getModelScale(imWidth, imHeight),
                    };

                    const composedClicks = [...pos_points, ...neg_points].map(([x, y], index) => ({
                        clickType: index < pos_points.length ? 1 : 0 as 0 | 1 | -1,
                        height: null,
                        width: null,
                        x,
                        y,
                    }));

                    const feeds = modelData({
                        clicks: composedClicks,
                        tensor: plugin.data.embeddings.get(key) as Tensor,
                        modelScale,
                        maskInput: plugin.data.lowResMasks.has(key) ? plugin.data.lowResMasks.get(key) as Tensor : null,
                    });

                    function toMatImage(input: number[], width: number, height: number): number[][] {
                        const image = Array(height).fill(0);
                        for (let i = 0; i < image.length; i++) {
                            image[i] = Array(width).fill(0);
                        }

                        for (let i = 0; i < input.length; i++) {
                            const row = Math.floor(i / width);
                            const col = i % width;
                            image[row][col] = input[i] * 255;
                        }

                        return image;
                    }

                    function onnxToImage(input: any, width: number, height: number): number[][] {
                        return toMatImage(input, width, height);
                    }

                    const data = await (plugin.data.session as InferenceSession).run(feeds);
                    const { masks, low_res_masks: lowResMasks } = data;
                    const imageData = onnxToImage(masks.data, masks.dims[3], masks.dims[2]);
                    plugin.data.lowResMasks.set(key, lowResMasks);

                    const xtl = Number(data.xtl.data[0]);
                    const xbr = Number(data.xbr.data[0]);
                    const ytl = Number(data.ytl.data[0]);
                    const ybr = Number(data.ybr.data[0]);

                    let regions = [];
                    if (result) {
                        const wrapper = window.document.getElementById('cvat_canvas_wrapper');
                        const canvas = window.document.getElementById('cvat_canvas_background');
                        const test = window.document.getElementById('cvat_canvas_bitmap');

                        window.magicButtonAction = () => {
                            const objects = [];
                            for (const [masks, xtl, xbr, ytl, ybr, imageData, result] of regions) {
                                const object = new plugin.data.core.classes.ObjectState({
                                    frame,
                                    objectType: plugin.data.core.enums.ObjectType.SHAPE,
                                    source: plugin.data.core.enums.Source.AUTO,
                                    label: job.labels[0],
                                    shapeType: plugin.data.core.enums.ShapeType.MASK,
                                    points: result,
                                    occluded: false,
                                    zOrder: 0,
                                });
                                objects.push(object);
                            }

                            window.onCreateAnnotations(job, 0, objects);
                        };



                        test.style.display = 'block';
                        test.style.background = 'none';
                        test.getContext('2d').globalAlpha = 0.5;
                        test.getContext('2d').clearRect(0,0,100000, 10000);
                        test.width = canvas.width;
                        test.height = canvas.height;

                        let pointsIntereset = [];
                        const gridX = 15;
                        const gridY = 15;
                        const segmentX = canvas.width / gridX;
                        const segmentY = canvas.height / gridY;
                        for (let i = 0; i < gridX; i++) {
                            for (let j = 0; j < gridY; j++) {
                                pointsIntereset.push({
                                    x: Math.round(segmentX * i + segmentX / 2),
                                    y: Math.round(segmentY * j + segmentY / 2),
                                    excluded: false,
                                });
                            }
                        }

                        for await (const point of pointsIntereset) {
                            if (point.excluded) {
                                continue;
                            }

                            const feeds1 = modelData({
                                clicks: [{
                                    clickType: 1,
                                    height: null,
                                    width: null,
                                    x: point.x,
                                    y: point.y,
                                }],
                                tensor: plugin.data.embeddings.get(key) as Tensor,
                                modelScale,
                                maskInput: null,
                            });

                            const data = await (plugin.data.session as InferenceSession).run(feeds1)
                            const { masks } = data;
                            const maskWidth = masks.dims[3];
                            const maskHeight = masks.dims[2];
                            const imageData = new ImageData(maskWidth, maskHeight);
                            for (let i = 0; i < masks.data.length; i++) {
                                if (masks.data[i]) {
                                    imageData.data[i * 4] = 137;
                                    imageData.data[i * 4 + 1] = 205;
                                    imageData.data[i * 4 + 2] = 211;
                                    imageData.data[i * 4 + 3] = 128;
                                }
                            };
                            // const imageData = onnxToImage(masks.data, masks.dims[3], masks.dims[2]);

                            const xtl = Number(data.xtl.data[0]);
                            const xbr = Number(data.xbr.data[0]);
                            const ytl = Number(data.ytl.data[0]);
                            const ybr = Number(data.ybr.data[0]);

                            const resultingImage = onnxToImage(masks.data, masks.dims[3], masks.dims[2]).flat();
                            resultingImage.push(xtl, ytl, xbr, ybr);
                            regions.push([masks, xtl, xbr, ytl, ybr, imageData, resultingImage]);

                            for (const checkPoint of pointsIntereset) {
                                const { x, y } = checkPoint;
                                if (x >= xtl && y >= ytl && x <= xbr && y <= ybr) {
                                    const localX = x - xtl;
                                    const localY = y - ytl;
                                    if (masks.data[localY * maskWidth + localX] && point !== checkPoint) {
                                        checkPoint.excluded = true;
                                    }
                                }
                            }
                        }

                        // pointsIntereset.forEach(({ x, y, excluded }) => {
                        //     if (!excluded) {
                        //         canvas.getContext('2d').fillRect(x - 15, y - 15, 30, 30)
                        //     }
                        // })



                        // cv.then((res) => {
                        //     const orig = res.imread('cvat_canvas_background')
                        //     const cv = res;
                        //     var orb = new cv.FastFeatureDetector();
                        //     let des = new cv.Mat();
                        //     let img3 = new cv.Mat();
                        //     var kp1 = new cv.KeyPointVector();
                        //     // find the keypoints with ORB
                        //     orb.detect(orig, kp1);
                        //     const points = [];
                        //     for (let i = 0; i < kp1.size(); i++) {
                        //         points.push(kp1.get(i));
                        //     }

                        //     let sample = new cv.Mat(points.length, 2, cv.CV_32F);
                        //     let j = 0;
                        //     for (const point of points) {
                        //         const { x,y } = point.pt;

                        //         sample.data32F[j] = x;
                        //         sample.data32F[j + 1] = y;
                        //         j += 2;
                        //     }

                        //     var clusterCount = 20;
                        //     var labels= new cv.Mat();
                        //     var attempts = 5;
                        //     var centers= new cv.Mat();

                        //     var crite= new cv.TermCriteria(cv.TermCriteria_EPS + cv.TermCriteria_MAX_ITER, 10000, 0.0001);
                        //     var criteria = [1,10,0.0001];

                        //     cv.kmeans(sample, clusterCount, labels, crite, attempts, cv.KMEANS_PP_CENTERS, centers);

                        //     const finishedCenters = [];
                        //     for (let k = 0; k < centers.data32F.length; k += 2) {
                        //         finishedCenters.push({
                        //             x: centers.data32F[k],
                        //             y: centers.data32F[k + 1],
                        //         });
                        //     }

                        //     finishedCenters.forEach(({ x, y }) => canvas.getContext('2d').fillRect(x - 15, y - 15, 30, 30))

                        //     // cv.drawKeypoints(orig, kp1, orig);
                        //     // cv.imshow('cvat_canvas_background', orig);

                        //     // compute the descriptors with ORB
                        //     // var das=new cv.Mat();
                        //     // orb.compute(orig, kp1, das);

                        //     // const detector = new res.AKAZE();
                        //     // const vector = new res.KeyPointVector();
                        //     // detector.detect(res.imread('cvat_canvas_background'), vector);
                        //     // const points = [];

                        //     // for (let i = 0; i < vector.size(); i++) {
                        //     //     const point = vector.get(i);
                        //     //     points.push(point);
                        //     //     console.log(points);
                        //     // }
                        // });

                        const listener = lodash.debounce((e) => {
                            const bbox = canvas?.getBoundingClientRect();
                            const { clientX, clientY } = e;
                            const { height: renderHeight, width: renderWidth, top, left } = bbox;
                            const { height, width } = canvas;
                            const canvasX = Math.round(((clientX - left) / renderWidth) * width);
                            const canvasY = Math.round(((clientY - top) / renderHeight) * height);
                            if (canvasX > 0 && canvasX < width && canvasY > 0 && canvasY < height) {
                                // const feeds1 = modelData({
                                //     clicks: [{
                                //         clickType: 1,
                                //         height: null,
                                //         width: null,
                                //         x: canvasX,
                                //         y: canvasY,
                                //     }],
                                //     tensor: plugin.data.embeddings.get(key) as Tensor,
                                //     modelScale,
                                //     maskInput: plugin.data.lowResMasks.has(key) ? plugin.data.lowResMasks.get(key) as Tensor : null,
                                // });

                                for (const [masks, xtl, xbr, ytl, ybr, imageData] of regions) {
                                    const maskWidth = masks.dims[3];
                                    const maskHeight = masks.dims[2];
                                    const localX = canvasX - xtl;
                                    const localY = canvasY - ytl;
                                    if (canvasX >= xtl && canvasY >= ytl && canvasX <= xbr && canvasY <= ybr) {
                                        if (masks.data[localY * maskWidth + localX]) {
                                            // const imageData = new ImageData(maskWidth, maskHeight);
                                            // for (let i = 0; i < masks.data.length; i++) {
                                            //     if (masks.data[i]) {
                                            //         imageData.data[i * 4] = 137;
                                            //         imageData.data[i * 4 + 1] = 205;
                                            //         imageData.data[i * 4 + 2] = 211;
                                            //         imageData.data[i * 4 + 3] = 128;
                                            //     }
                                            // };
                                            test.getContext('2d').clearRect(0,0,100000, 10000);
                                            test.getContext('2d').putImageData(imageData, xtl, ytl)
                                            return;

                                        }
                                    }

                                }

                                // (plugin.data.session as InferenceSession).run(feeds1).then((data) => {
                                //     const { masks } = data;
                                //     const maskWidth = masks.dims[3];
                                //     const maskHeight = masks.dims[2];
                                //     const imageData = new ImageData(maskWidth, maskHeight);
                                //     for (let i = 0; i < masks.data.length; i++) {
                                //         if (masks.data[i]) {
                                //             imageData.data[i * 4] = 137;
                                //             imageData.data[i * 4 + 1] = 205;
                                //             imageData.data[i * 4 + 2] = 211;
                                //             imageData.data[i * 4 + 3] = 128;
                                //         }
                                //     };
                                //     // const imageData = onnxToImage(masks.data, masks.dims[3], masks.dims[2]);

                                //     const xtl = Number(data.xtl.data[0]);
                                //     const xbr = Number(data.xbr.data[0]);
                                //     const ytl = Number(data.ytl.data[0]);
                                //     const ybr = Number(data.ybr.data[0]);

                                //     // const { width: testWidth, height: testHeight } = test;
                                //     // const left = (xtl / width) * testWidth;
                                //     // const top = (ytl / height) * testHeight;

                                //     test.getContext('2d').clearRect(0,0,100000, 10000);
                                //     test.getContext('2d').putImageData(imageData, xtl, ytl)
                                // });
                            }
                            console.log(canvasX, canvasY);
                        })

                        // wrapper.addEventListener('mousemove', listener)

                        // canvas?.addEventListener('mousemove', (e) => console.log(e));
                    }

                    return {
                        mask: imageData,
                        bounds: [xtl, ytl, xbr, ybr],
                    };
                },
            },
        },
    },
    data: {
        core: null,
        jobs: {},
        modelID: 'pth-facebookresearch-sam-vit-h',
        modelURL: '/api/lambda/sam_detector.onnx',
        embeddings: new LRUCache({
            // float32 tensor [256, 64, 64] is 4 MB, max 512 MB
            max: 128,
            updateAgeOnGet: true,
            updateAgeOnHas: true,
        }),
        lowResMasks: new LRUCache({
            // float32 tensor [1, 256, 256] is 0.25 MB, max 32 MB
            max: 128,
            updateAgeOnGet: true,
            updateAgeOnHas: true,
        }),
        session: null,
    },
    callbacks: {
        onStatusChange: null,
    },
};

const SAMModelPlugin: ComponentBuilder = ({ core }) => {
    samPlugin.data.core = core;
    InferenceSession.create(samPlugin.data.modelURL).then((session) => {
        samPlugin.data.session = session;
        core.plugins.register(samPlugin);
    });

    return {
        name: 'Segment Anything model',
        destructor: () => {},
    };
};

function register(): void {
    if (Object.prototype.hasOwnProperty.call(window, 'cvatUI')) {
        (window as any as { cvatUI: { registerComponent: PluginEntryPoint } })
            .cvatUI.registerComponent(SAMModelPlugin);
    }
}

window.addEventListener('plugins.ready', register, { once: true });
