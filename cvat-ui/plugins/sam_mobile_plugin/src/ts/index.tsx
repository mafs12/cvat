// Copyright (C) 2023 CVAT.ai Corporation
//
// SPDX-License-Identifier: MIT

import React, { useState } from 'react';
import { InferenceSession, Tensor } from 'onnxruntime-web';
import { LRUCache } from 'lru-cache';
import lodash from 'lodash';

import openCVWrapper from 'utils/opencv-wrapper/opencv-wrapper';
import { PluginEntryPoint, APIWrapperEnterOptions, ComponentBuilder } from 'components/plugins-entrypoint';
import { useSelector } from 'react-redux';
import { CombinedState } from 'reducers';
import { useDispatch } from 'react-redux';
import { createAnnotationsAsync } from 'actions/annotation-actions';
import Icon from '@ant-design/icons/lib/components/Icon';
import { AimOutlined } from '@ant-design/icons';
import { Button } from 'antd';

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
            list: {
                leave: (plugin: SAMPlugin, results: any) => Promise<any>;
            },
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
        modelID: string;
        jobs: Record<number, any>;
        encoderInputSize: number;
        encoderModelURL: string;
        decoderModelURL: string;
        encoder: InferenceSession | null;
        decoder: InferenceSession | null;

        embeddings: LRUCache<string, Tensor>;
        lowResMasks: LRUCache<string, Tensor>;

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
            list: {
                async leave(
                    plugin: SAMPlugin,
                    results: any,
                ): Promise<any> {
                    return {
                        models: [new plugin.data.core.classes.MLModel({
                            animated_gif: 'https://raw.githubusercontent.com/opencv/cvat/develop/site/content/en/images/hrnet_example.gif',
                            description: 'SAM in the browser',
                            framework: 'pytorch',
                            help_message: 'The interactor allows to get a mask of an object using at least one positive, and any negative points inside it',
                            id: plugin.data.modelID,
                            kind: 'interactor',
                            labels: [],
                            min_neg_points: 0,
                            min_pos_points: 1,
                            name: 'Segment Anything Mobile',
                            startswith_box: false,
                            version: 2,
                        }), ...results.models],
                        count: results.count + 1,
                    };
                    // return results;
                },
            },
            call: {
                async enter(
                    plugin: SAMPlugin,
                    taskID: number,
                    model: any, { frame }: { frame: number },
                ): Promise<null | APIWrapperEnterOptions> {
                    if (model.id === plugin.data.modelID) {
                        const job = Object.values(plugin.data.jobs)
                            .find((_job) => _job.taskId === taskID);
                        if (!job) {
                            throw new Error('Could not find a job corresponding to the request');
                        }

                        const embeddings = await getEmbeddings(plugin, job, frame);
                        return { preventMethodCall: true };
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

                    const data = await (plugin.data.decoder as InferenceSession).run(feeds);
                    const { masks, low_res_masks: lowResMasks } = data;
                    const imageData = onnxToImage(masks.data, masks.dims[3], masks.dims[2]);
                    plugin.data.lowResMasks.set(key, lowResMasks);

                    const xtl = Number(data.xtl.data[0]);
                    const xbr = Number(data.xbr.data[0]);
                    const ytl = Number(data.ytl.data[0]);
                    const ybr = Number(data.ybr.data[0]);

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
        modelID: 'pth-facebookresearch-sam-mobile-vit-t',
        encoderInputSize: 1024,
        encoderModelURL: '/api/lambda/sam_mobile_encoder.onnx',
        decoderModelURL: '/api/lambda/sam_mobile_decoder.onnx',
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
        encoder: null,
        decoder: null,
    },
    callbacks: {
        onStatusChange: null,
    },
};

async function getEmbeddings(plugin: SAMPlugin, job: any, frame: number): Promise<Tensor> {
    const taskID = job.taskId;
    if (!plugin.data.encoder || !plugin.data.decoder) {
        throw new Error('SAM plugin is not ready, sessions were not initialized');
    }

    const key = `${taskID}_${frame}`;
    if (plugin.data.embeddings.has(key)) {
        return plugin.data.embeddings.get(key) as Tensor;
    }

    const frameData = await job.frames.get(frame);
    const { imageData: imageBitmap } = await frameData.data();
    const scale = 1024 / Math.max(imageBitmap.width, imageBitmap.height);

    const offscreenCanvas = new OffscreenCanvas(1024, 1024);
    const context = offscreenCanvas.getContext('2d');
    if (context) {
        context.drawImage(
            imageBitmap, 0, 0,
            Math.ceil(imageBitmap.width * scale),
            Math.ceil(imageBitmap.height * scale),
        );
        const scaledImageData = context.getImageData(0, 0, 1024, 1024);
        const imageDataNoAlpha = Float32Array
            .from(scaledImageData.data.filter((_, idx) => (idx + 1) % 4), (el) => el);
        const inputBlob = new Tensor('float32', imageDataNoAlpha, [1024, 1024, 3]);
        const feeds = { input_image: inputBlob };
        const results = await plugin.data.encoder.run(feeds);
        plugin.data.embeddings.set(key, results.image_embeddings);
        return results.image_embeddings;
    }

    throw new Error('Offscreen canvas 2D context not found');
}

const SAMModelPlugin: ComponentBuilder = ({ dispatch, core, REGISTER_ACTION, REMOVE_ACTION }) => {
    samPlugin.data.core = core;

    core.plugins.register(samPlugin);
    InferenceSession.create(samPlugin.data.encoderModelURL).then((session) => {
        samPlugin.data.encoder = session;
        InferenceSession.create(samPlugin.data.decoderModelURL).then((decoderSession) => {
            samPlugin.data.decoder = decoderSession;
        });
    });

    const Component = () => {
        const [fetching, setFetcing] = useState(false);
        const frame = useSelector((state: CombinedState) => state.annotation.player.frame.number);
        const job = useSelector((state: CombinedState) => state.annotation.job.instance);
        const dispatch = useDispatch();

        return (
            <Button
                disabled={fetching}
                style={{
                    width: 40,
                    height: 40,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    fontSize: 24,
                }}
                onClick={
                    async () => {
                        try {
                            setFetcing(true);
                            const frameData = await job.frames.get(frame);
                            const embeddings = await getEmbeddings(samPlugin, job, frame);

                            const wrapper = window.document.getElementById('cvat_canvas_wrapper');
                            const canvas = window.document.getElementById('cvat_canvas_background');
                            const test = window.document.getElementById('cvat_canvas_bitmap');

                            let regions = [];





                            // test.style.display = 'block';
                            // test.style.background = 'none';
                            // test.getContext('2d').globalAlpha = 0.5;
                            // test.getContext('2d').clearRect(0,0,100000, 10000);
                            // test.width = canvas.width;
                            // test.height = canvas.height;

                            let pointsIntereset = [];
                            const gridX = 15;
                            const gridY = 15;
                            const segmentX = frameData.width / gridX;
                            const segmentY = frameData.height / gridY;
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
                                    tensor: embeddings,
                                    modelScale: {
                                        width: frameData.width,
                                        height: frameData.height,
                                        samScale: getModelScale(frameData.width, frameData.height),
                                    },
                                    maskInput: null,
                                });

                                const data = await (samPlugin.data.decoder as InferenceSession).run(feeds1)
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
                            }, 100);

                            // canvas?.addEventListener('mousemove')

                            const objects = [];
                            for (const [masks, xtl, xbr, ytl, ybr, imageData, result] of regions) {
                                const object = new samPlugin.data.core.classes.ObjectState({
                                    frame,
                                    objectType: samPlugin.data.core.enums.ObjectType.SHAPE,
                                    source: samPlugin.data.core.enums.Source.AUTO,
                                    label: job.labels[0],
                                    shapeType: samPlugin.data.core.enums.ShapeType.MASK,
                                    points: result,
                                    occluded: false,
                                    zOrder: 0,
                                });
                                objects.push(object);
                            }

                            dispatch(createAnnotationsAsync(job, frame, objects));
                            // window.onCreateAnnotations(job, 0, objects);
                        } catch (error) {
                            console.log(error);
                        } finally {
                            setFetcing(false);
                        }
                    }
                }
            >
                <AimOutlined />
            </Button>
        );
    };

    dispatch({
        type: REGISTER_ACTION,
        payload: {
            path: 'annotationPage.controlsSidebar',
            component: Component,
        },
    });

    return {
        name: samPlugin.name,
        destructor: () => {
            dispatch({
                type: REMOVE_ACTION,
                payload: {
                    path: 'annotationPage.controlsSidebar',
                    component: Component,
                },
            });
        },
    };
};

function register(): void {
    if (Object.prototype.hasOwnProperty.call(window, 'cvatUI')) {
        (window as any as { cvatUI: { registerComponent: PluginEntryPoint } })
            .cvatUI.registerComponent(SAMModelPlugin);
    }
}

window.addEventListener('plugins.ready', register, { once: true });
