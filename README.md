# 复眼光斑检测器 使用说明

## 环境准备
```bash
conda activate myenv
```

## 使用方法

### 1. 处理单张图像
```bash
python spot_detector.py --image "你的图像.bmp"
```

### 2. 运行速度测试
```bash
python spot_detector.py --benchmark
```

### 3. 运行极限重叠测试
```bash
python spot_detector.py --test
```

## 性能指标

| 指标 | 数值 |
|------|------|
| 5120×5120图像处理时间 | ~890ms |
| 帧率 | ~1 FPS |
| 检测精度 | 亚像素级 (误差<1px) |
| 最小可分离距离 | ~10像素 |

## 输出结果

检测结果保存在 `detection_output/` 目录。

## 核心算法

1. **降采样检测** - 4倍降采样加速粗检
2. **高斯平滑** - σ=5像素去噪
3. **局部最大值** - min_distance=200像素
4. **质心精修** - 30×30窗口加权质心
5. **重叠分离** - BIC模型选择(1/2/3高斯)
