// Copyright (C) 2023 CVAT.ai Corporation
//
// SPDX-License-Identifier: MIT

import {
    Project, Task, Job, CloudStorage, MLModel,
} from 'cvat-core-wrapper';
import { PluginEntryPoint, ComponentBuilder } from 'components/plugins-entrypoint';
import indexedStorage, { DatabaseStore } from './indexed-storage';

type PreviewFor = Project | Task | Job | CloudStorage | MLModel;

const previewEnter = (store: DatabaseStore) => async function (this: PreviewFor) {
    const item = await indexedStorage.getItem<Blob>(store, `${this.id}`);
    if (item) {
        return { preventMethodCall: true };
    }
    return null;
};

const previewLeave = (store: DatabaseStore) => async function (
    this: PreviewFor, _: CachePlugin, result: string,
): Promise<string> {
    if (result) {
        await indexedStorage.setItem<string>(store, `${this.id}`, result);
        return result;
    }
    const item = await indexedStorage.getItem<string>(store, `${this.id}`);
    return item || '';
};

type PreviewEnterType = (this: PreviewFor, _: CachePlugin) => Promise<{ preventMethodCall: boolean } | null>;
type PreviewLeaveType = (this: PreviewFor, _: CachePlugin, result: string) => Promise<string>;

interface CachePlugin {
    name: string;
    description: string;
    cvat: {
        classes: {
            Project: {
                prototype: {
                    preview: {
                        enter: PreviewEnterType;
                        leave: PreviewLeaveType;
                    };
                };
            };
            Task: {
                prototype: {
                    frames: {
                        preview: {
                            enter: PreviewEnterType;
                            leave: PreviewLeaveType;
                        };
                    };
                };
            };
            Job: {
                prototype: {
                    frames: {
                        preview: {
                            enter: PreviewEnterType;
                            leave: PreviewLeaveType;
                        };
                    };
                };
            };
            CloudStorage: {
                prototype: {
                    preview: {
                        enter: PreviewEnterType;
                        leave: PreviewLeaveType;
                    };
                };
            };
            MLModel: {
                prototype: {
                    preview: {
                        enter: PreviewEnterType;
                        leave: PreviewLeaveType;
                    };
                };
            };
            FrameData: {
                prototype: {
                    data: {
                        enter: () => {};
                        leave: () => {};
                    };
                };
            };
        };
    };
}

const cachePlugin: CachePlugin = {
    name: 'Cache plugin',
    description: 'Plugin enables cache for some static data',
    cvat: {
        classes: {
            Project: {
                prototype: {
                    preview: {
                        enter: previewEnter(DatabaseStore.PROJECTS_PREVIEW),
                        leave: previewLeave(DatabaseStore.PROJECTS_PREVIEW),
                    },
                },
            },
            Task: {
                prototype: {
                    frames: {
                        preview: {
                            enter: previewEnter(DatabaseStore.TASKS_PREVIEW),
                            leave: previewLeave(DatabaseStore.TASKS_PREVIEW),
                        },
                    },
                },
            },
            Job: {
                prototype: {
                    frames: {
                        preview: {
                            enter: previewEnter(DatabaseStore.JOBS_PREVIEW),
                            leave: previewLeave(DatabaseStore.JOBS_PREVIEW),
                        },
                    },
                },
            },
            CloudStorage: {
                prototype: {
                    preview: {
                        enter: previewEnter(DatabaseStore.CLOUDSTORAGES_PREVIEW),
                        leave: previewLeave(DatabaseStore.CLOUDSTORAGES_PREVIEW),
                    },
                },
            },
            MLModel: {
                prototype: {
                    preview: {
                        enter: previewEnter(DatabaseStore.FUNCTIONS_PREVIEW),
                        leave: previewLeave(DatabaseStore.FUNCTIONS_PREVIEW),
                    },
                },
            },
            FrameData: {
                prototype: {
                    data: {
                        enter: async function enter(this: any) { // FrameData was not typified
                            const { jid, number } = this;
                            const key = `compressed_${jid}_${number}`;
                            // todo: handle compressed/original chunks
                            const item = await indexedStorage.getItem<any>(DatabaseStore.COMPRESSED_JOB_IMAGES, key);
                            if (item) {
                                return { preventMethodCall: true };
                            }
                            return null;
                        },
                        leave: async function leave(
                            this: any, _: CachePlugin,
                            result: {
                                renderWidth: number;
                                renderHeight: number;
                                imageData: ImageBitmap | ImageData;
                            } | Blob,
                        ) {
                            const { jid, number } = this;
                            const key = `compressed_${jid}_${number}`;
                            if (result && !(result instanceof Blob) && result.imageData instanceof ImageBitmap) {
                                await indexedStorage
                                    .setItem<any>(DatabaseStore.COMPRESSED_JOB_IMAGES, key, result);
                                return result;
                            }
                            const item = await indexedStorage
                                .getItem<ImageBitmap>(DatabaseStore.COMPRESSED_JOB_IMAGES, key);
                            return item;
                        },
                    },
                },
            },
        },
    },
};

const SAMModelPlugin: ComponentBuilder = ({ core }) => {
    core.plugins.register(cachePlugin);

    return {
        name: 'Cache plugin',
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
