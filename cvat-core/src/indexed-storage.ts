// Copyright (C) 2023 CVAT.ai Corporation
//
// SPDX-License-Identifier = MIT

import config from './config';

export enum DatabaseStore {
    PROJECTS_PREVIEW = 'projects_preview',
    TASKS_PREVIEW = 'tasks_preview',
    JOBS_PREVIEW = 'jobs_preview',
    CLOUDSTORAGES_PREVIEW = 'cloudstorages_preview',
    FUNCTIONS_PREVIEW = 'functions_preview',
    COMPRESSED_JOB_CHUNKS = 'compressed_job_chunks',
    COMPRESSED_JOB_IMAGES = 'compressed_job_images',
    CONTEXT_IMAGES = 'context_images',
}

enum IDBTransactionMode {
    READONLY = 'readonly',
    READWRITE = 'readwrite',
}

interface IObject {
    id: string;
    payload: any;
}

const isInstance = (val: any, constructor: Function | string, store: DatabaseStore): void => {
    if (typeof constructor === 'string') {
        if (typeof val !== constructor) {
            throw new Error(`Store ${store}: validation error, object is not of type ${constructor}`);
        }
    } else if (!(val instanceof constructor)) {
        throw new Error(`Store ${store}: validation error, object is not of type ${constructor.name}`);
    }
};

async function blob2ArrayBuffer(blob: Blob): Promise<{ arrayBuffer: ArrayBuffer, type: string }> {
    const { type } = blob;
    const arrayBuffer = await blob.arrayBuffer();
    return {
        arrayBuffer,
        type,
    };
}

async function arrayBuffer2Blob({ arrayBuffer, type }: { arrayBuffer: ArrayBuffer, type: string }): Promise<Blob> {
    return new Blob([arrayBuffer], { type });
}

class CVATIndexedStorage {
    #dbName = 'cvat_db';
    #version = 1;
    #db: IDBDatabase | null = null;
    #dbUpgradedAnotherTab = false;
    #isQuotaExceed: boolean = false;
    #initializationPromise: Promise<void> | null = null;
    #storeConfiguration = {
        [DatabaseStore.PROJECTS_PREVIEW]: {
            options: { keyPath: 'id' },
            validator: (el: Blob) => isInstance(el, Blob, DatabaseStore.PROJECTS_PREVIEW),
            serialize: async (el: Blob): ReturnType<typeof blob2ArrayBuffer> => blob2ArrayBuffer(el),
            deserialize: async (
                el: Awaited<ReturnType<typeof blob2ArrayBuffer>>,
            ): Promise<Blob> => arrayBuffer2Blob(el),
        },
        [DatabaseStore.TASKS_PREVIEW]: {
            options: { keyPath: 'id' },
            validator: (el: Blob) => isInstance(el, Blob, DatabaseStore.TASKS_PREVIEW),
            serialize: async (el: Blob): ReturnType<typeof blob2ArrayBuffer> => blob2ArrayBuffer(el),
            deserialize: async (
                el: Awaited<ReturnType<typeof blob2ArrayBuffer>>,
            ): Promise<Blob> => arrayBuffer2Blob(el),
        },
        [DatabaseStore.JOBS_PREVIEW]: {
            options: { keyPath: 'id' },
            validator: (el: Blob) => isInstance(el, Blob, DatabaseStore.JOBS_PREVIEW),
            serialize: async (el: Blob): ReturnType<typeof blob2ArrayBuffer> => blob2ArrayBuffer(el),
            deserialize: async (
                el: Awaited<ReturnType<typeof blob2ArrayBuffer>>,
            ): Promise<Blob> => arrayBuffer2Blob(el),
        },
        [DatabaseStore.CLOUDSTORAGES_PREVIEW]: {
            options: { keyPath: 'id' },
            validator: (el: Blob) => isInstance(el, Blob, DatabaseStore.CLOUDSTORAGES_PREVIEW),
            serialize: async (el: Blob): ReturnType<typeof blob2ArrayBuffer> => blob2ArrayBuffer(el),
            deserialize: async (
                el: Awaited<ReturnType<typeof blob2ArrayBuffer>>,
            ): Promise<Blob> => arrayBuffer2Blob(el),
        },
        [DatabaseStore.FUNCTIONS_PREVIEW]: {
            options: { keyPath: 'id' },
            validator: (el: Blob) => isInstance(el, Blob, DatabaseStore.FUNCTIONS_PREVIEW),
            serialize: async (el: Blob): ReturnType<typeof blob2ArrayBuffer> => blob2ArrayBuffer(el),
            deserialize: async (
                el: Awaited<ReturnType<typeof blob2ArrayBuffer>>,
            ): Promise<Blob> => arrayBuffer2Blob(el),
        },
        [DatabaseStore.COMPRESSED_JOB_CHUNKS]: {
            options: { keyPath: 'id' },
            validator: (el: ArrayBuffer) => isInstance(el, ArrayBuffer, DatabaseStore.COMPRESSED_JOB_CHUNKS),
            serialize: (el: ArrayBuffer) => Promise.resolve(el),
            deserialize: (el: ArrayBuffer) => Promise.resolve(el),
        },
        [DatabaseStore.CONTEXT_IMAGES]: {
            options: { keyPath: 'id' },
            validator: (el: ArrayBuffer) => isInstance(el, ArrayBuffer, DatabaseStore.CONTEXT_IMAGES),
            serialize: (el: ArrayBuffer) => Promise.resolve(el),
            deserialize: (el: ArrayBuffer) => Promise.resolve(el),
        },
    };

    load(): Promise<void> {
        if (!indexedDB || this.#dbUpgradedAnotherTab) {
            return Promise.reject();
        }

        if (this.#db) {
            return Promise.resolve();
        }

        if (!this.#initializationPromise) {
            this.#initializationPromise = new Promise<void>((resolve, reject) => {
                const request = indexedDB.open(this.#dbName, this.#version);
                request.onupgradeneeded = () => {
                    const db = request.result;

                    Object.entries(this.#storeConfiguration).forEach(([storeName, conf]) => {
                        if (!db.objectStoreNames.contains(storeName)) {
                            db.createObjectStore(storeName, conf.options);
                        }
                    });
                };

                request.onsuccess = () => {
                    this.#db = request.result;
                    this.#db.onversionchange = () => {
                        if (this.#db) {
                            this.#db.close();
                        }

                        this.#db = null;
                        this.#dbUpgradedAnotherTab = true;
                        // a user upgraded (with new version in js code) or deleted the db in another tab
                        // we close connection and remove the database in current one

                        if (alert) {
                            // eslint-disable-next-line
                            alert('App was updated. Please, reload the browser page, otherwise further correct work is not guaranteed');
                        }
                    };

                    resolve();
                };

                request.onerror = (event: Event) => {
                    const error = (event.target as any).error as Error;
                    console.warn(error);
                    reject(request.error);
                };
            });
        }

        return this.#initializationPromise;
    }

    setItem<T>(storeName: DatabaseStore, id: string, object: T): Promise<boolean> {
        return new Promise((resolve) => {
            if (!config.enableIndexedDBCache || this.#isQuotaExceed) {
                resolve(false);
            }

            this.load().then(() => {
                try {
                    const { validator, serialize } = this.#storeConfiguration[storeName];
                    validator(object);

                    serialize(object).then((serialized) => {
                        const transaction = (this.#db as IDBDatabase)
                            .transaction(storeName, IDBTransactionMode.READWRITE);
                        transaction.onabort = (event: Event) => {
                            const error = (event.target as any).error as Error;
                            console.warn(error);
                            if (error.name === 'QuotaExceededError') {
                                this.#isQuotaExceed = true;
                            }
                            resolve(false);
                        };

                        transaction.onerror = (event: Event) => {
                            // The error event is fired on IDBTransaction when a request
                            // returns an error and the event bubbles up to the transaction object.
                            const error = (event.target as any).error as Error;
                            console.warn(error);
                            resolve(false);
                        };
                        const indexedStore = transaction.objectStore(storeName);
                        const request = indexedStore.put({
                            id,
                            payload: serialized,
                        });
                        request.onsuccess = () => resolve(true);
                    }).catch(() => resolve(false));
                } catch (error) {
                    console.warn(error);
                    resolve(false);
                }
            }).catch((error) => {
                if (error) {
                    console.warn(error);
                }
                resolve(false);
            });
        });
    }

    getItem<T>(storeName: DatabaseStore, id: string): Promise<T | null> {
        return new Promise((resolve) => {
            if (!config.enableIndexedDBCache) {
                resolve(null);
            }

            if (this.#isQuotaExceed) {
                // reading is also disabled
                // because the database might being clearing in another thread
                // and we do not want to freeze the user thread
                // TODO: Check it works
                resolve(null);
            }

            const { deserialize } = this.#storeConfiguration[storeName];

            this.load().then(() => {
                try {
                    const transaction = (this.#db as IDBDatabase)
                        .transaction(storeName, IDBTransactionMode.READONLY);
                    transaction.onabort = (event: Event) => {
                        const error = (event.target as any).error as Error;
                        console.warn(error);
                        resolve(null);
                    };
                    transaction.onerror = (event: Event) => {
                        const error = (event.target as any).error as Error;
                        console.warn(error);
                        resolve(null);
                    };

                    const indexedStore = transaction.objectStore(storeName);
                    const request = indexedStore.get(id);

                    request.onsuccess = (event: Event) => {
                        const result = (event.target as any).result as IObject;
                        if (typeof result === 'undefined') {
                            resolve(null);
                        } else {
                            deserialize(result.payload).then((deserialized: T) => {
                                resolve(deserialized);
                            }).catch(() => {
                                resolve(null);
                            });
                        }
                    };
                } catch (error) {
                    console.warn(error);
                    resolve(null);
                }
            }).catch((error) => {
                if (error) {
                    console.warn(error);
                }
                resolve(null);
            });
        });
    }

    estimate(): Promise<{ quota: number; usage: number } | null> {
        return new Promise((resolve) => {
            if (navigator?.storage) {
                navigator.storage.estimate().then((estimation: StorageEstimate) => {
                    if (typeof estimation.quota === 'number' && typeof estimation.usage === 'number') {
                        resolve({
                            quota: estimation.quota,
                            usage: estimation.usage,
                        });
                    } else {
                        resolve(null);
                    }
                }).catch(() => {
                    resolve(null);
                });
            } else {
                resolve(null);
            }
        });
    }

    clear(): Promise<void> {
        return new Promise((resolve, reject) => {
            if (this.#db) {
                this.#db.close();
            }
            const request = indexedDB.deleteDatabase(this.#dbName);

            request.onerror = (event) => {
                reject((event.target as any).error || new Error('Could not delete the database'));
            };

            request.onsuccess = () => {
                this.load().finally(() => resolve());
            };
        });
    }
}

export default new CVATIndexedStorage();
