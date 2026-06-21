# MobileCLIP2 ReID — 场外部署指南

## 目录

- [1. 核心问题解答](#1-核心问题解答)
- [2. 导出 ONNX 模型](#2-导出-onnx-模型)
- [3. 目标设备环境准备](#3-目标设备环境准备)
- [4. 推理脚本使用](#4-推理脚本使用)
- [5. 各语言 SDK 集成](#5-各语言-sdk-集成)
- [6. 生产部署建议](#6-生产部署建议)

---

## 1. 核心问题解答

### Q: ONNX 导出后，还需要模型结构文件吗？

**不需要。** ONNX 是自包含的"计算图+权重"二进制文件，包含了：
- 网络结构（每层的类型、输入输出形状）
- 权重参数（已序列化）
- 元数据（opset 版本、输入输出 name）

**唯一需要额外提供的是预处理配置** (`deploy_config.json`)，因为：
- ONNX 只定义"已归一化的 CHW float32 数组 → 特征向量"这一段
- 预处理（resize / 归一化 / ToTensor）不属于网络，需要你自己实现

```
deploy/
├── mobileclip2-reid.onnx        # 自包含模型 (≈ 几十 MB)
└── deploy_config.json           # 预处理参数 (≈ 1 KB)
```

### Q: 目标设备需要安装 PyTorch / timm / mobileclip 吗？

**完全不需要。** ONNX 是开放标准，只需 onnxruntime 即可运行：

| 设备 | 推荐 Runtime | GPU 支持 |
|------|-------------|---------|
| CPU (x86/arm) | `onnxruntime` | ❌ |
| NVIDIA GPU | `onnxruntime-gpu` | ✅ CUDA / TensorRT |
| Apple Silicon | `onnxruntime` (arm64) | ⚠️ CoreML via ORT |
| 移动端 / 嵌入式 | `onnxruntime-mobile` | ⚠️ 需量化 |

### Q: 预处理不一致会导致精度下降吗？

**会。** 这是 ReID 部署最常见的精度损失原因。本方案通过 `deploy_config.json` 保证训练/推理预处理严格一致：

| 参数 | 训练配置 | 部署必须值 |
|------|---------|-----------|
| `image_size` | `[256, 128]` | 必须是这个，不能随意改 |
| `pixel_mean` | `[0.0, 0.0, 0.0]` | **不要用 ImageNet 的 [0.485, 0.456, 0.406]** |
| `pixel_std` | `[1.0, 1.0, 1.0]` | **不要用 ImageNet 的 [0.229, 0.224, 0.225]** |

> ⚠️ **MobileCLIP2 预训练时用的是 mean=0, std=1（无归一化），这一点与原 CLIP-ReID (mean=0.5, std=0.5) 和 ImageNet 模型都不同。**

---

## 2. 导出 ONNX 模型

### 2.1 在训练机上执行导出

```bash
# 确保已安装依赖
pip install torch onnxruntime timm mobileclip open_clip_root

# 导出 (需要 CUDA)
CUDA_VISIBLE_DEVICES=0 python export_onnx.py \
    --checkpoint    logs/mobileclip2/MobileCLIP2-S0_60.pth \
    --config        configs/person/vit_mobileclip2.yml \
    --output_dir    deploy/

# 预期输出:
#   ✅ ONNX 导出完成
#   ├─ 模型文件  : deploy/mobileclip2-reid.onnx  (~50 MB)
#   ├─ 配置文件  : deploy/deploy_config.json
#   ├─ 特征维度  : 1024
#   └─ 推理尺寸  : 1 x 3 x 256 x 128
```

### 2.2 deploy_config.json 示例

```json
{
  "model_name": "MobileCLIP2-S0",
  "image_size": [256, 128],
  "pixel_mean": [0.0, 0.0, 0.0],
  "pixel_std": [1.0, 1.0, 1.0],
  "neck_feat": "before",
  "feat_norm": "yes",
  "onnx_input_name": "images",
  "onnx_output_name": "features",
  "onnx_output_dim": 1024,
  "onnx_file": "mobileclip2-reid.onnx"
}
```

### 2.3 复制到目标设备

```bash
# 只需这两个文件
scp -r deploy/ user@target-device:/opt/reid_model/
```

---

## 3. 目标设备环境准备

### 纯 CPU 推理 (最低依赖)

```bash
pip install numpy Pillow onnxruntime
# or for faster CPU execution (Intel/AMD):
pip install onnxruntime-openvino
```

### NVIDIA GPU 推理

```bash
pip install numpy Pillow onnxruntime-gpu
# 确保 CUDA / cuDNN 版本匹配
# onnxruntime-gpu 兼容 CUDA 11.x / 12.x
```

### ARM 嵌入式 (e.g. 瑞芯微 / 算能)

```bash
# RKNN 路线: ONNX → RKNN (需要 RKNN-Toolkit2)
pip install numpy Pillow onnx
# 然后用 RKNN-Toolkit2 转成 .rknn

# TNN / NCNN 路线: ONNX → TNN/NCNN (跨平台)
# 参考: https://github.com/Tencent/TNN
```

### Python 版本要求

- Python ≥ 3.8 (推荐 3.9+)
- numpy ≥ 1.20
- Pillow ≥ 9.0

---

## 4. 推理脚本使用

### 4.1 基础: 单图特征提取

```python
from inference_onnx import ReIDEngine, Preprocessor
from PIL import Image

# 加载 (只依赖 onnxruntime)
engine = ReIDEngine("mobileclip2-reid.onnx", "deploy_config.json")
preproc = Preprocessor(image_size=(256, 128),
                        pixel_mean=[0.0, 0.0, 0.0],
                        pixel_std=[1.0, 1.0, 1.0])

# 提取特征
img = Image.open("person.jpg").convert("RGB")
feat = engine.extract_from_pil(preproc, img)
print(f"特征形状: {feat.shape}")  # (1024,)
```

### 4.2 底库搜索

```bash
python inference_onnx.py \
    --onnx       deploy/mobileclip2-reid.onnx \
    --config     deploy/deploy_config.json \
    --image      assets/query/0001_c1s1_000301_00.jpg \
    --image_dir  assets/gallery/ \
    --top_k      10
```

输出示例:

```
图像: assets/query/0001_c1s1_000301_00.jpg
特征维度: (1024,)  范围=[-0.0442, 0.0391]

排名  距离       PID      相机   路径
--------------------------------------------------------------------------------
1     0.0000    1        2      0001_c2s3_051301_00.jpg
2     0.1287    1        3      0001_c3s1_075301_00.jpg
3     0.2154    1        4      0001_c4s2_090401_00.jpg
```

### 4.3 批量评测 (计算 mAP + CMC)

```bash
python inference_onnx.py \
    --onnx       deploy/mobileclip2-reid.onnx \
    --config     deploy/deploy_config.json \
    --query_dir  data/market1501/query/ \
    --image_dir  data/market1501/bounding_box_test/ \
    --top_k      50
```

### 4.4 C++ / Java / Go / C# 集成

只需读取 `deploy_config.json` 中的 `onnx_input_name`、`onnx_output_name`、
`onnx_output_dim`，然后调用对应语言的 ONNX Runtime API：

| 语言 | ONNX Runtime 包 |
|------|----------------|
| Python | `onnxruntime` / `onnxruntime-gpu` |
| C++ | `onnxruntime_cxx.h` (CMake) |
| Java | `onnxruntime` (Maven) |
| Go | `gopkg.in/gy-kong/onnxruntime.v1` |
| C#/.NET | `Microsoft.ML.OnnxRuntime` (NuGet) |

详见 [ONNX Runtime 官方文档](https://onnxruntime.ai/docs/)

---

## 5. 各语言 SDK 集成

### 5.1 Python (最简单)

```python
import onnxruntime as ort
import numpy as np
from PIL import Image

sess = ort.InferenceSession("mobileclip2-reid.onnx")
input_name = sess.get_inputs()[0].name   # "images"
output_name = sess.get_outputs()[0].name  # "features"

# 预处理
img = Image.open("x.jpg").resize((128, 256)).convert("RGB")  # W×H!
arr = np.array(img, dtype=np.float32) / 255.0
arr = (arr - [0, 0, 0]) / [1, 1, 1]  # MobileCLIP2 归一化
arr = arr.transpose(2, 0, 1)[np.newaxis, ...]  # [1,3,256,128]

# 推理
feat = sess.run([output_name], {input_name: arr})[0]
feat = feat / np.linalg.norm(feat, axis=1, keepdims=True)  # L2 归一化
```

### 5.2 C++ (Linux 推理服务器)

```cpp
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>

int main() {
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING);
    Ort::Session session(env, "mobileclip2-reid.onnx",
                         Ort::SessionOptions{nullptr});

    std::vector<const char*> input_names = {"images"};
    std::vector<const char*> output_names = {"features"};

    // 预处理 (OpenCV): resize + ToTensor + normalize
    cv::Mat img = cv::imread("person.jpg");
    cv::resize(img, img, {128, 256});  // W, H
    img.convertTo(img, CV_32FC3, 1.0/255.0);
    // MobileCLIP2: mean=0, std=1 (跳过减均值除标准差)
    img = img.reshape({1, 3, 256, 128});

    std::vector<float> input(1 * 3 * 256 * 128);
    memcpy(input.data(), img.ptr<float>(), input.size() * sizeof(float));

    auto output = session.Run(Ort::RunOptions{nullptr},
                              input_names.data(), input.data(), input.size(),
                              output_names.data(), 1);
    // output[0] 即特征向量 (1024维)
}
```

### 5.3 Android (Java + onnxruntime)

```java
// app/build.gradle
// implementation 'com.microsoft.onnxruntime:onnxruntime-android:1.17.0'

val session = OrtEnvironment.getEnvironment().createSession("mobileclip2-reid.onnx")
val inputBuffer = preprocessBitmap(bitmap)  // 纯 Java 图像处理
val output = session.run(arrayOf(inputBuffer))
val feature = output[0].floatBuffer.array()  // 1024 维
```

---

## 6. 生产部署建议

### 6.1 性能基准 (参考值, MobileCLIP2-S0)

| 配置 | 输入尺寸 | 延迟 (CPU) | 延迟 (RTX 3090) | 吞吐量 |
|------|---------|-----------|----------------|--------|
| S0 ONNX (float32) | 256×128 | ~30ms/图 | ~2ms/图 | ~500图/秒 (GPU batch) |
| S0 ONNX (int8量化) | 256×128 | ~8ms/图 | ~1ms/图 | ~2000图/秒 (GPU) |
| S2 ONNX (float32) | 256×128 | ~45ms/图 | ~3ms/图 | ~330图/秒 (GPU) |

### 6.2 量化加速 (可选)

```bash
# 训练机安装
pip install onnx onnxruntime

# INT8 动态量化 (无需校准数据集，速度提升 3-4x)
python -m onnxruntime.transformers.quantize \
    --input deploy/mobileclip2-reid.onnx \
    --output deploy/mobileclip2-reid-int8.onnx \
    --quant_format QOperator
```

### 6.3 HTTP API 服务模板 (Flask / FastAPI)

```python
# deploy/server.py — 推理 HTTP 服务
from fastapi import FastAPI, UploadFile, File
from inference_onnx import ReIDEngine, Preprocessor
import io
from PIL import Image

app = FastAPI()
engine = ReIDEngine("mobileclip2-reid.onnx", "deploy_config.json")
preproc = Preprocessor(image_size=(256, 128),
                        pixel_mean=[0.0,0.0,0.0],
                        pixel_std=[1.0,1.0,1.0])

@app.post("/extract")
async def extract(file: UploadFile = File(...)):
    img = Image.open(io.BytesIO(await file.read())).convert("RGB")
    feat = engine.extract_from_pil(preproc, img)
    return {"feature": feat.tolist(), "dim": len(feat)}
```

启动: `uvicorn deploy.server:app --host 0.0.0.0 --port 8000`

---

## 7. 常见问题

**Q: 导出时报错 `ModuleNotFoundError: mobileclip.models`**
> 训练机需要安装 ml-mobileclip 包:
> ```bash
> cd /path/to/ml-mobileclip && pip install -e .
> ```

**Q: onnxruntime 推理结果与 PyTorch 不完全一致**
> 正常。浮点运算的顺序差异会导致 ~1e-6 的误差，cosine similarity > 0.999 即可接受。

**Q: 推理报 `Input shape mismatch`**
> 检查 `image_size` 是否与导出时一致（必须是 `[H, W]`）。resize 时方向最容易搞反。

**Q: 部署到 RKNN 报 shape 不匹配**
> ONNX → RKNN 时需要用 `input_size_list` 指定每维 size。
> MobileCLIP2-S0 输入: `[1, 3, 256, 128]`
