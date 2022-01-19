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

import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.util.Log;
import android.widget.TextView;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.List;
import java.nio.channels.FileChannel;

import android.os.Build;
import java.util.TreeMap;
import java.util.Map;
import java.util.HashMap;
import java.util.concurrent.ThreadLocalRandom;

public class NNTestBase  {
    private static final String TAG = "NNTestBase";
    protected final boolean ALLOW_EXEC_PARALLEL = true;
    protected final boolean USE_NNAPI = true;
    protected NNBenchmark mActivity;
    protected TextView mText;
    private String mModelName;
    private com.mediatek.neuropilot.Interpreter mInterpreter;
    private com.mediatek.neuropilot.Interpreter.Options mOptions;
    private com.mediatek.neuropilot_R.Interpreter mInterpreter_R;
    private com.mediatek.neuropilot_R.Interpreter.Options mOptions_R;
	private String mCacheDir;
    private int[] mInputShape;
    private List<String> mLabelList;
    private ByteBuffer mImgData;
    Map<Integer, Object> mOuts;

    public NNTestBase(String modelName, int[] inputShape) {
        mModelName = modelName;
        mInputShape = inputShape;
        mInterpreter = null;
        mInterpreter_R = null;
    }

    public final void createBaseTest(NNBenchmark ipact) {
        mActivity = ipact;
        mLabelList = loadLabelList(mModelName);
        mImgData = generateTestInout(mModelName);
        try {
            if (Build.VERSION.SDK_INT < 30) {
                mOptions = new com.mediatek.neuropilot.Interpreter.Options();
                mOptions.setUseNNAPI(USE_NNAPI);
                mOptions.setAllowFp16PrecisionForFp32(true);
                mCacheDir = "/data/data/com.mediatek.nnbenchmark/cache/";
                mOptions.setCacheDir(mCacheDir);
                mOptions.setPreference(com.mediatek.neuropilot.Interpreter.Options.
				    ExecutionPreference.kSustainedSpeed.getValue());
                mInterpreter = new com.mediatek.neuropilot.Interpreter(loadModelFile(), mOptions);
            } else {
                mOptions_R = new com.mediatek.neuropilot_R.Interpreter.Options();
                mOptions_R.setUseNNAPI(USE_NNAPI);
                mOptions_R.setAllowFp16PrecisionForFp32(true);
                mCacheDir = "/data/data/com.mediatek.nnbenchmark/cache/";
                mOptions_R.setCacheDir(mCacheDir);
                mOptions_R.setPreference(com.mediatek.neuropilot_R.Interpreter.Options.
				    ExecutionPreference.kSustainedSpeed.getValue());
                mInterpreter_R = new com.mediatek.neuropilot_R.Interpreter(loadModelFile(), mOptions_R);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public String getTestInfo() {
        return mModelName;
    }


    public void runTest() {
        mOuts = new HashMap<>();
        ByteBuffer mOutData;
        if (mInterpreter != null) {
            com.mediatek.neuropilot.Tensor outputTensor = mInterpreter.getOutputTensor(0);
            mOutData = ByteBuffer.allocateDirect(outputTensor.numBytes());
            mOutData.order(ByteOrder.nativeOrder());
            mOutData.rewind();
            mOuts.put(0, mOutData);
            mInterpreter.runForMultipleInputsOutputs(new Object[]{mImgData}, mOuts);
        } else if (mInterpreter_R != null) {
            com.mediatek.neuropilot_R.Tensor outputTensor = mInterpreter_R.getOutputTensor(0);
            mOutData = ByteBuffer.allocateDirect(outputTensor.numBytes());
            mOutData.order(ByteOrder.nativeOrder());
            mOutData.rewind();
            mOuts.put(0, mOutData);
            mInterpreter_R.runForMultipleInputsOutputs(new Object[]{mImgData}, mOuts);
        }

            /*File file = new File("/sdcard/out/" + System.currentTimeMillis());
            try {
                FileChannel wChannel = new FileOutputStream(file, false).getChannel();
                ((ByteBuffer)outs.get(0)).flip();
                wChannel.write((ByteBuffer)outs.get(0));
                wChannel.close();
            } catch (IOException e) {
                Log.e(TAG, "Write File Fail %s", e);
            }*/
    }

    public void destroy() {
        mInterpreter = null;
        mInterpreter_R = null;
        mImgData = null;
        mLabelList = null;
        mOptions = null;
        mOptions_R = null;
        mOuts = null;
    }

    /** Memory-map the model file in Assets. */
    private MappedByteBuffer loadModelFile() throws IOException {
        AssetFileDescriptor fileDescriptor = mActivity.getAssets().openFd(mModelName + ".tflite");
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }
    private ByteBuffer generateTestInout(String modelName) {
        int batchsize = 1;
        int imageSizeX = 224;
        int imageSizeY = 224;
        int dimPixelSize = 3;
        int numBytesPerChannel = 1;

        if(modelName.contentEquals(new String("inception_v3_mtk"))
                || modelName.contentEquals(new String("inception_v3_quant"))){
            imageSizeX = 299;
            imageSizeY = 299;
        }

        if(modelName.contentEquals(new String("mobilenet_ssd_pascal"))
                || modelName.contentEquals(new String("mobilenet_ssd_pascal_quant"))){
            imageSizeX = 300;
            imageSizeY = 300;
        }
        if(modelName.contentEquals(new String("deeplab_v3_mtk"))
                || modelName.contentEquals(new String("deeplab_v3_quant"))
                || modelName.contentEquals(new String("deeplabv3_mnv2_ade20k_int8"))){
            imageSizeX = 513;
            imageSizeY = 513;
        }
        if (modelName.contentEquals(new String("ESRGAN_float")) ||
                modelName.contentEquals(new String("ESRGAN_quant"))) {
            imageSizeX = 50;
            imageSizeY = 50;
        }
        if (modelName.contentEquals(new String("mobilebert_float_384_gpu"))) {
            imageSizeX = 384;
            imageSizeY = 384;
        }
        if (modelName.contentEquals(new String("mobilenet_float"))
                || modelName.contentEquals(new String("mobilenet_ssd_pascal"))
                || modelName.contentEquals(new String("inception_v3_mtk"))
                || modelName.contentEquals(new String("resnet_v1_mtk"))
                || modelName.contentEquals(new String("deeplab_v3_mtk"))
                || modelName.contentEquals(new String("ESRGAN_float"))) {
            numBytesPerChannel = 4;
        }

        if (modelName.contentEquals(new String("mobilenet_float"))) {
            numBytesPerChannel = 4;
        }

        mImgData = ByteBuffer.allocateDirect(batchsize
                                * imageSizeX
                                * imageSizeY
                                * dimPixelSize
                                * numBytesPerChannel);
        mImgData.order(ByteOrder.nativeOrder());
        mImgData.rewind();

        Bitmap bmp = Util.getBitmapFromAsset(mActivity.getApplicationContext(), "grace_hopper.bmp");
        Bitmap scaledBmp = Bitmap.createScaledBitmap(bmp, imageSizeX, imageSizeY, true);
//        if (modelName.contentEquals(new String("mobilenet_float"))) {
//            convertBitmapToFloatBuffer(scaledBmp, imageSizeX, imageSizeY, mImgData);
//        } else if (modelName.contentEquals(new String("mobilenet_quantized"))) {
//            convertBitmapToByteBuffer(scaledBmp, imageSizeX, imageSizeY, mImgData);
//        }

        if (modelName.contentEquals(new String("inception_v3_mtk"))
                || modelName.contentEquals(new String("mobilenet_float"))
                || modelName.contentEquals(new String("mobilenet_ssd_pascal"))
                || modelName.contentEquals(new String("resnet_v1_mtk"))
                || modelName.contentEquals(new String("deeplab_v3_mtk"))
                || modelName.contentEquals(new String("ESRGAN_float"))
                || modelName.contentEquals(new String("mobilebert_float_384_gpu"))){
            convertBitmapToFloatBuffer(scaledBmp, imageSizeX, imageSizeY, mImgData);
        } else if (modelName.contentEquals(new String("inception_v3_quant"))
                || modelName.contentEquals(new String("mobilenet_quantized"))
                || modelName.contentEquals(new String("mobilenet_ssd_pascal_quant"))
                || modelName.contentEquals(new String("resnet_v1_quant"))
                || modelName.contentEquals(new String("deeplab_v3_quant"))
                || modelName.contentEquals(new String("ESRGAN_quant"))
                || modelName.contentEquals(new String("deeplabv3_mnv2_ade20k_int8"))) {
            convertBitmapToByteBuffer(scaledBmp, imageSizeX, imageSizeY, mImgData);
        }

        return  mImgData;
    }

    private void convertBitmapToByteBuffer(Bitmap bitmap, int imageSizeX, int imageSizeY,
                                           ByteBuffer out) {
        if (out == null) {
            return;
        }
        out.rewind();
        int[] intValues = new int[imageSizeX * imageSizeY];
        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
        int pixel = 0;
        for (int i = 0; i < imageSizeX; ++i) {
            for (int j = 0; j < imageSizeY; ++j) {
                final int val = intValues[pixel++];
                mImgData.put((byte) ((val >> 16) & 0xFF));
                mImgData.put((byte) ((val >> 8) & 0xFF));
                mImgData.put((byte) (val & 0xFF));
            }
        }
    }

    private void convertBitmapToFloatBuffer(Bitmap bitmap, int imageSizeX, int imageSizeY,
                                            ByteBuffer out) {
        if (out == null) {
            return;
        }
        out.rewind();
        int[] intValues = new int[imageSizeX * imageSizeY];
        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
        int pixel = 0;
        for (int i = 0; i < imageSizeX; ++i) {
            for (int j = 0; j < imageSizeY; ++j) {
                final int val = intValues[pixel++];
                mImgData.putFloat((float) ((((val >> 16) & 0xFF) - 127.5) / 127.5));
                mImgData.putFloat((float) ((((val >> 8) & 0xFF) - 127.5) / 127.5));
                mImgData.putFloat((float) (((val & 0xFF) - 127.5) / 127.5));
            }
        }
    }

    /** Reads label list from Assets. */
    private List<String> loadLabelList(String modelName) {
        String labelPath = null;
        if (modelName.contentEquals(new String("mobilenet_float")) ||
                modelName.contentEquals(new String("mobilenet_quantized"))) {
            labelPath = "labels_imagenet_slim.txt";
        }
        Log.d(TAG, "Label path:" + labelPath);
        if (labelPath == null) {
            return null;
        }

        List<String> labelList = new ArrayList<String>();
        try {
            BufferedReader reader =
                    new BufferedReader(new InputStreamReader(mActivity.getAssets().open(labelPath)));

            String line;
            while ((line = reader.readLine()) != null) {
                labelList.add(line);
            }
            reader.close();
        } catch (IOException e) {
            Log.e(TAG, "Can not load label.");
            e.printStackTrace();
            return null;
        }
        return labelList;
    }
}
