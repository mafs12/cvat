// Copyright (C) 2023 CVAT.ai Corporation
//
// SPDX-License-Identifier: MIT

import { InferenceSession, Tensor } from 'onnxruntime-web';
import { LRUCache } from 'lru-cache';
import { PluginEntryPoint, APIWrapperEnterOptions, ComponentBuilder } from 'components/plugins-entrypoint';

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
                        if (!plugin.data.encoder || !plugin.data.decoder) {
                            throw new Error('SAM plugin is not ready, sessions were not initialized');
                        }

                        // const scaleX = dimension / img.width;
                        // const scaleY = dimension / img.height;
                        // dataType?: 'float32' | 'uint8';
                        /**
                         * Tensor channel layout - default is 'NHWC'
                         */
                        // tensorLayout?: 'NHWC' | 'NCHW';



                        const key = `${taskID}_${frame}`;
                        if (plugin.data.embeddings.has(key)) {
                            return { preventMethodCall: true };
                        }


                        const maxSize = Math.max(window.document.getElementById('cvat_canvas_background').width, window.document.getElementById('cvat_canvas_background').height);
                        const bitmap = await createImageBitmap(window.document.getElementById('cvat_canvas_background'), 0, 0, maxSize, maxSize, { resizeHeight: 1024, resizeWidth: 1024 });
                        let canvas = new OffscreenCanvas(1024, 1024);
                        // canvas.width = 1024;
                        // canvas.height = 1024;
                        canvas.getContext('2d')?.drawImage(bitmap, 0, 0);
                        canvas.getContext('2d')?.getImageData(0, 0, 1024, 1024);

                        const src = canvas.getContext('2d')?.getImageData(0, 0, 1024, 1024);
                        const float = Float32Array.from(src.data.filter((_, idx) => (idx + 1) % 4), (el) => el);
                        const input_image = new Tensor('float32', float, [1024, 1024, 3]);



                        // const input_image = await Tensor.fromImage(bitmap, { width: 1284, height: 1284, dataFormat: 'NHWC' });
                        // const input_image = await Tensor.fromImage(window.document.getElementById('cvat_canvas_background').getContext('2d').getImageData(0, 0, 1284, 1284), { resizedWidth: 1024, resizedHeight: 1024, dataType: 'uint8', tensorLayout: 'NHWC' });


                        // imageDataTensor = new ort.Tensor(tf_tensor.dataSync(), tf_tensor.shape);
                        const feeds = { "input_image": input_image };
                        let results = await plugin.data.encoder.run(feeds);
                        plugin.data.embeddings.set(key, results.image_embeddings);

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

const SAMModelPlugin: ComponentBuilder = ({ core }) => {
    samPlugin.data.core = core;

    core.plugins.register(samPlugin);
    InferenceSession.create(samPlugin.data.encoderModelURL).then((session) => {
        samPlugin.data.encoder = session;
        InferenceSession.create(samPlugin.data.decoderModelURL).then((decoderSession) => {
            samPlugin.data.decoder = decoderSession;
        });
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
