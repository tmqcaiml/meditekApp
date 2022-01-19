/*
 * Copyright (C) 2017 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.mediatek.nn.benchmark;

public class NNTestList {
    /**
     * Define enum type for test names
     */
    public enum TestName {

        MobileNet_FLOAT ("MobileNet float32", 160.f),
        MobileNet_QUANT8 ("MobileNet quantized", 50.f),
        MobileNet_SSD_FLOAT ("Mobilenet SSD float", 160.f),
        MobileNet_SSD_QUANT8 ("Mobilenet SSD quantized", 50.f),
        Inception_FLOAT ("Inception V3 float", 160.f),
        Inception_QUANT8 ("Inception V3 quantized", 50.f),
        Resnet_FLOAT ("Resnet 50 float", 160.f),
        Resnet_QUANT8  ("Resnet 50 quantized", 50.f),
        Deeplab_FLOAT ("Deeplab V3+ float", 160.f),
        Deeplab_QUANT8 ("Deeplab V3+ quantized", 50.f),
        Srgan_FLOAT ("SRGAN float", 160.f);
        //Deeplab_INT8("Deep Lab MlPerf", 50.f),
        //MonileBert_FLOAT("Mobile Bert Float MLPerf", 160.f);

        private final String name;
        public final float baseline;

        private TestName(String s, float base) {
            name = s;
            baseline = base;
        }
        private TestName(String s) {
            name = s;
            baseline = 1.f;
        }

        // return quoted string as displayed test name
        public String toString() {
            return name;
        }
    }

    static NNTestBase newTest(TestName testName) {
        switch(testName) {
            case MobileNet_FLOAT:
                return new NNTestBase("mobilenet_float", new int[]{1, 224, 224, 3});
            case MobileNet_QUANT8:
                return new NNTestBase("mobilenet_quantized", new int[]{1, 224, 224, 3});
            case MobileNet_SSD_FLOAT:
                return new NNTestBase("mobilenet_ssd_pascal", new int[]{1, 300, 300, 3});
            case MobileNet_SSD_QUANT8:
                return new NNTestBase("mobilenet_ssd_pascal_quant", new int[]{1, 300, 300, 3});
            case Inception_FLOAT:
                return new NNTestBase("inception_v3_mtk", new int[]{1, 299, 299, 3});
            case Inception_QUANT8:
                return new NNTestBase("inception_v3_quant", new int[]{1, 299, 299, 3});
            case Resnet_FLOAT:
                return new NNTestBase("resnet_v1_mtk", new int[]{1, 224, 224, 3});
            case Resnet_QUANT8:
                return new NNTestBase("resnet_v1_quant", new int[]{1, 224, 224, 3});
            case Deeplab_FLOAT:
                return new NNTestBase("deeplab_v3_mtk", new int[]{1, 513, 513, 3});
            case Deeplab_QUANT8:
                return new NNTestBase("deeplab_v3_quant", new int[]{1, 513, 513, 3});
            case Srgan_FLOAT:
                return new NNTestBase("ESRGAN_float", new int[]{1, 50, 50, 3});
            default:
                return null;
        }
    }
}

