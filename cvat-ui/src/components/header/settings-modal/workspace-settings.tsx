// Copyright (C) 2020-2022 Intel Corporation
//
// SPDX-License-Identifier: MIT

import React, { useEffect, useState } from 'react';

import { Row, Col } from 'antd/lib/grid';
import Checkbox, { CheckboxChangeEvent } from 'antd/lib/checkbox';
import InputNumber from 'antd/lib/input-number';
import Text from 'antd/lib/typography/Text';
import Slider from 'antd/lib/slider';
import Select from 'antd/lib/select';

import {
    MAX_ACCURACY,
    marks,
} from 'components/annotation-page/standard-workspace/controls-side-bar/approximation-accuracy';
import { getCore } from 'cvat-core-wrapper';
import { clamp } from 'utils/math';
import { Button, Modal } from 'antd';

const core = getCore();

interface Props {
    autoSave: boolean;
    autoSaveInterval: number;
    aamZoomMargin: number;
    showAllInterpolationTracks: boolean;
    showObjectsTextAlways: boolean;
    automaticBordering: boolean;
    intelligentPolygonCrop: boolean;
    defaultApproxPolyAccuracy: number;
    textFontSize: number;
    controlPointsSize: number;
    textPosition: 'center' | 'auto';
    textContent: string;
    showTagsOnFrame: boolean;
    enableImagesCache: boolean;
    onSwitchAutoSave(enabled: boolean): void;
    onSwitchImagesCache(enabled: boolean): void;
    onChangeAutoSaveInterval(interval: number): void;
    onChangeAAMZoomMargin(margin: number): void;
    onChangeDefaultApproxPolyAccuracy(approxPolyAccuracy: number): void;
    onSwitchShowingInterpolatedTracks(enabled: boolean): void;
    onSwitchShowingObjectsTextAlways(enabled: boolean): void;
    onSwitchAutomaticBordering(enabled: boolean): void;
    onSwitchIntelligentPolygonCrop(enabled: boolean): void;
    onChangeTextFontSize(fontSize: number): void;
    onChangeControlPointsSize(pointsSize: number): void;
    onChangeTextPosition(position: 'auto' | 'center'): void;
    onChangeTextContent(textContent: string[]): void;
    onSwitchShowingTagsOnFrame(enabled: boolean): void;
}

function WorkspaceSettingsComponent(props: Props): JSX.Element {
    const [storageEstimation, setStorageEstimation] = useState<StorageEstimate | null>(null);
    const {
        autoSave,
        autoSaveInterval,
        aamZoomMargin,
        showAllInterpolationTracks,
        showObjectsTextAlways,
        automaticBordering,
        intelligentPolygonCrop,
        defaultApproxPolyAccuracy,
        textFontSize,
        controlPointsSize,
        textPosition,
        textContent,
        showTagsOnFrame,
        enableImagesCache,
        onSwitchAutoSave,
        onSwitchImagesCache,
        onChangeAutoSaveInterval,
        onChangeAAMZoomMargin,
        onSwitchShowingInterpolatedTracks,
        onSwitchShowingObjectsTextAlways,
        onSwitchAutomaticBordering,
        onSwitchIntelligentPolygonCrop,
        onChangeDefaultApproxPolyAccuracy,
        onChangeTextFontSize,
        onChangeControlPointsSize,
        onChangeTextPosition,
        onChangeTextContent,
        onSwitchShowingTagsOnFrame,
    } = props;

    const minAutoSaveInterval = 1;
    const maxAutoSaveInterval = 60;
    const minAAMMargin = 0;
    const maxAAMMargin = 1000;
    const minControlPointsSize = 4;
    const maxControlPointsSize = 8;

    useEffect(() => {
        core.storage.estimate().then((estimation: StorageEstimate | null) => {
            setStorageEstimation(estimation);
        });
    }, []);

    const storageAvailableMb = ((storageEstimation?.quota || 0) / (1024 * 1024 * 1024)).toFixed(2);
    const storageUsedMb = ((storageEstimation?.usage || 0) / (1024 * 1024 * 1024)).toFixed(2);
    const storageMessage = `Gigabytes available ${storageAvailableMb} and ${storageUsedMb} in use`;

    return (
        <div className='cvat-workspace-settings'>
            <Row>
                <Col>
                    <Checkbox
                        className='cvat-text-color cvat-workspace-settings-auto-save'
                        checked={autoSave}
                        onChange={(event: CheckboxChangeEvent): void => {
                            onSwitchAutoSave(event.target.checked);
                        }}
                    >
                        Enable auto save
                    </Checkbox>
                </Col>
            </Row>
            <Row>
                <Col className='cvat-workspace-settings-auto-save-interval'>
                    <Text type='secondary'> Auto save every </Text>
                    <InputNumber
                        min={minAutoSaveInterval}
                        max={maxAutoSaveInterval}
                        step={1}
                        value={Math.round(autoSaveInterval / (60 * 1000))}
                        onChange={(value: number | undefined | string): void => {
                            if (typeof value !== 'undefined') {
                                onChangeAutoSaveInterval(
                                    Math.floor(clamp(+value, minAutoSaveInterval, maxAutoSaveInterval)) * 60 * 1000,
                                );
                            }
                        }}
                    />
                    <Text type='secondary'> minutes </Text>
                </Col>
            </Row>
            <Row>
                <Col span={12} className='cvat-workspace-settings-enable-images-cache'>
                    <Row>
                        <Checkbox
                            className='cvat-text-color'
                            checked={enableImagesCache}
                            onChange={(event: CheckboxChangeEvent): void => {
                                if (event.target.checked) {
                                    window.navigator.storage.persisted().then((persisted) => {
                                        if (!persisted) {
                                            return window.navigator.storage.persist();
                                        }
                                        return Promise.resolve(true);
                                    }).then((res) => {
                                        if (!res) {
                                            Modal.info({
                                                title: 'Storage notification',
                                                content: 'Browser did not allow to use persistent storage, it means that cache can be removed by a browser any time',
                                            });
                                        }
                                    });
                                }
                                onSwitchImagesCache(event.target.checked);
                            }}
                        >
                            Enable images cache
                        </Checkbox>
                    </Row>
                    <Row className='cvat-workspace-settings-cache-size-info'>
                        <Text type='secondary'>Cache accelerates image navigation</Text>
                        { storageEstimation && <Text type='secondary'>{storageMessage}</Text>}
                        <Button
                            type='link'
                            onClick={() => {
                                core.storage.clear().then(() => {
                                    core.storage.estimate().then((estimation: StorageEstimate | null) => {
                                        setStorageEstimation(estimation);
                                    });
                                });
                            }}
                        >
                            clear
                        </Button>
                    </Row>
                </Col>
            </Row>
            <Row>
                <Col span={12} className='cvat-workspace-settings-show-interpolated'>
                    <Row>
                        <Checkbox
                            className='cvat-text-color'
                            checked={showAllInterpolationTracks}
                            onChange={(event: CheckboxChangeEvent): void => {
                                onSwitchShowingInterpolatedTracks(event.target.checked);
                            }}
                        >
                            Show all interpolation tracks
                        </Checkbox>
                    </Row>
                    <Row>
                        <Text type='secondary'> Show hidden interpolated objects in the side panel</Text>
                    </Row>
                </Col>
            </Row>
            <Row className='cvat-workspace-settings-show-text-always'>
                <Col span={24}>
                    <Checkbox
                        className='cvat-text-color'
                        checked={showObjectsTextAlways}
                        onChange={(event: CheckboxChangeEvent): void => {
                            onSwitchShowingObjectsTextAlways(event.target.checked);
                        }}
                    >
                        Always show object details
                    </Checkbox>
                </Col>
                <Col span={24}>
                    <Text type='secondary'>
                        Show text for an object on the canvas not only when the object is activated
                    </Text>
                </Col>
            </Row>
            <Row className='cvat-workspace-settings-text-settings'>
                <Col span={24}>
                    <Text>Content of a text</Text>
                </Col>
                <Col span={16}>
                    <Select
                        className='cvat-workspace-settings-text-content'
                        mode='multiple'
                        value={textContent.split(',').filter((entry: string) => !!entry)}
                        onChange={onChangeTextContent}
                    >
                        <Select.Option value='id'>ID</Select.Option>
                        <Select.Option value='label'>Label</Select.Option>
                        <Select.Option value='attributes'>Attributes</Select.Option>
                        <Select.Option value='source'>Source</Select.Option>
                        <Select.Option value='descriptions'>Descriptions</Select.Option>
                    </Select>
                </Col>
            </Row>
            <Row className='cvat-workspace-settings-text-settings'>
                <Col span={12}>
                    <Text>Position of a text</Text>
                </Col>
                <Col span={12}>
                    <Text>Font size of a text</Text>
                </Col>
                <Col span={12}>
                    <Select
                        className='cvat-workspace-settings-text-position'
                        value={textPosition}
                        onChange={onChangeTextPosition}
                    >
                        <Select.Option value='auto'>Auto</Select.Option>
                        <Select.Option value='center'>Center</Select.Option>
                    </Select>
                </Col>
                <Col span={12}>
                    <InputNumber
                        className='cvat-workspace-settings-text-size'
                        onChange={onChangeTextFontSize}
                        min={8}
                        max={20}
                        value={textFontSize}
                    />
                </Col>
            </Row>
            <Row className='cvat-workspace-settings-autoborders'>
                <Col span={24}>
                    <Checkbox
                        className='cvat-text-color'
                        checked={automaticBordering}
                        onChange={(event: CheckboxChangeEvent): void => {
                            onSwitchAutomaticBordering(event.target.checked);
                        }}
                    >
                        Automatic bordering
                    </Checkbox>
                </Col>
                <Col span={24}>
                    <Text type='secondary'>
                        Enable automatic bordering for polygons and polylines during drawing/editing
                    </Text>
                </Col>
            </Row>
            <Row className='cvat-workspace-settings-intelligent-polygon-cropping'>
                <Col span={24}>
                    <Checkbox
                        className='cvat-text-color'
                        checked={intelligentPolygonCrop}
                        onChange={(event: CheckboxChangeEvent): void => {
                            onSwitchIntelligentPolygonCrop(event.target.checked);
                        }}
                    >
                        Intelligent polygon cropping
                    </Checkbox>
                </Col>
                <Col span={24}>
                    <Text type='secondary'>Try to crop polygons automatically when editing</Text>
                </Col>
            </Row>
            <Row className='cvat-workspace-settings-show-frame-tags'>
                <Col span={24}>
                    <Checkbox
                        className='cvat-text-color'
                        checked={showTagsOnFrame}
                        onChange={(event: CheckboxChangeEvent): void => {
                            onSwitchShowingTagsOnFrame(event.target.checked);
                        }}
                    >
                        Show tags on frame
                    </Checkbox>
                </Col>
                <Col span={24}>
                    <Text type='secondary'>Show frame tags in the corner of the workspace</Text>
                </Col>
            </Row>
            <Row className='cvat-workspace-settings-aam-zoom-margin'>
                <Col>
                    <Text className='cvat-text-color'> Attribute annotation mode (AAM) zoom margin </Text>
                    <InputNumber
                        min={minAAMMargin}
                        max={maxAAMMargin}
                        value={aamZoomMargin}
                        onChange={(value: number | undefined | string): void => {
                            if (typeof value !== 'undefined') {
                                onChangeAAMZoomMargin(Math.floor(clamp(+value, minAAMMargin, maxAAMMargin)));
                            }
                        }}
                    />
                </Col>
            </Row>
            <Row className='cvat-workspace-settings-control-points-size'>
                <Col>
                    <Text className='cvat-text-color'> Control points size </Text>
                    <InputNumber
                        min={minControlPointsSize}
                        max={maxControlPointsSize}
                        value={controlPointsSize}
                        onChange={(value: number | undefined | string): void => {
                            if (typeof value !== 'undefined') {
                                onChangeControlPointsSize(
                                    Math.floor(clamp(+value, minControlPointsSize, maxControlPointsSize)),
                                );
                            }
                        }}
                    />
                </Col>
            </Row>
            <Row className='cvat-workspace-settings-approx-poly-threshold'>
                <Col>
                    <Text className='cvat-text-color'>Default number of points in polygon approximation</Text>
                </Col>
                <Col span={7} offset={1}>
                    <Slider
                        min={0}
                        max={MAX_ACCURACY}
                        step={1}
                        value={defaultApproxPolyAccuracy}
                        dots
                        onChange={onChangeDefaultApproxPolyAccuracy}
                        marks={marks}
                    />
                </Col>
                <Col span={24}>
                    <Text type='secondary'>Works for serverless interactors and OpenCV scissors</Text>
                </Col>
            </Row>
        </div>
    );
}

export default React.memo(WorkspaceSettingsComponent);
