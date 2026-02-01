"""
复眼光斑检测与分离 - 生产版
===========================
优化版本，支持：
1. 极近距离重叠分离测试
2. GPU加速处理
3. 毫秒级实时检测

使用方法:
    conda activate myenv
    python spot_detector.py --image your_image.bmp
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # 解决 OpenMP 库冲突

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
matplotlib.rcParams['axes.unicode_minus'] = False
from scipy import ndimage
from scipy.optimize import least_squares
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
import time
import os
import argparse

# GPU支持
try:
    import torch
    import torch.nn.functional as F
    HAS_GPU = torch.cuda.is_available()
    if HAS_GPU:
        DEVICE = torch.device('cuda')
        print(f"✓ GPU可用: {torch.cuda.get_device_name(0)}")
    else:
        DEVICE = torch.device('cpu')
except ImportError:
    HAS_GPU = False
    print("! PyTorch未安装")

OUTPUT_DIR = "detection_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ================== 核心算法 ==================

def detect_spots_fast(image, min_distance=180, threshold=50, use_gpu=True):
    """
    快速光斑检测 - 优化版
    
    Args:
        image: 输入图像 (numpy array)
        min_distance: 最小光斑间距 (像素) - 增加到250以避免重复检测
        threshold: 强度阈值
        use_gpu: 是否使用GPU (暂时用CPU更稳定)
    
    Returns:
        spots: 光斑列表 [{'y':, 'x':, 'intensity':}, ...]
    """
    start = time.time()
    
    # 动态阈值: 使用99.9分位数避免极值影响，至少为设定的threshold
    img_valid = image[image > 0]
    if img_valid.size > 0:
        p99 = np.percentile(img_valid, 99.9)
        dynamic_threshold = max(threshold, p99 * 0.15) # 稍微提高比例
    else:
        dynamic_threshold = threshold
    
    # 高斯平滑
    smoothed = ndimage.gaussian_filter(image.astype(np.float32), sigma=5)
    
    # 局部最大值
    # 先降采样找大致位置，再在原图精修
    downsample = 4
    small = smoothed[::downsample, ::downsample]
    small_max = ndimage.maximum_filter(small, size=min_distance//downsample)
    peak_mask_small = (small == small_max) & (small > dynamic_threshold)
    
    # 获取降采样后的峰值位置
    peaks_small = np.array(np.where(peak_mask_small)).T
    
    # 在原图精修位置
    spots = []
    h, w = image.shape
    border_margin = 200  # 边缘忽略区域 - 增加到200
    
    for py, px in peaks_small:
        # 在原图的局部区域找精确位置
        y_c, x_c = py * downsample, px * downsample
        margin = downsample * 2
        
        y_min = max(0, y_c - margin)
        y_max = min(image.shape[0], y_c + margin)
        x_min = max(0, x_c - margin)
        x_max = min(image.shape[1], x_c + margin)
        
        local = smoothed[y_min:y_max, x_min:x_max]
        if local.size == 0:
            continue
            
        local_max_pos = np.unravel_index(np.argmax(local), local.shape)
        y_exact = y_min + local_max_pos[0]
        x_exact = x_min + local_max_pos[1]
        
        # 过滤边缘
        if (y_exact < border_margin or y_exact > h - border_margin or
            x_exact < border_margin or x_exact > w - border_margin):
            continue
            
        if image[y_exact, x_exact] > dynamic_threshold:
            spots.append({
                'y': float(y_exact), 
                'x': float(x_exact), 
                'amplitude': float(image[y_exact, x_exact])
            })
    
    elapsed = (time.time() - start) * 1000
    return spots, elapsed


def refine_centroids_batch(image, spots, box_size=30):
    """
    批量质心精修 - 向量化加速
    """
    start = time.time()
    
    y_coords = np.array([s['y'] for s in spots])
    x_coords = np.array([s['x'] for s in spots])
    
    # 这里简单起见还是循环吧，但去掉了一些多余计算
    for spot in spots:
        y0, x0 = int(spot['y']), int(spot['x'])
        half = box_size // 2
        
        y_min = max(0, y0 - half)
        y_max = min(image.shape[0], y0 + half)
        x_min = max(0, x0 - half)
        x_max = min(image.shape[1], x0 + half) # Fixed bug: x_c -> x0
        
        local = image[y_min:y_max, x_min:x_max].astype(np.float64)
        bg = np.percentile(local, 20)
        local = np.maximum(local - bg, 0)
        
        total = local.sum()
        if total > 0:
            yy, xx = np.mgrid[0:local.shape[0], 0:local.shape[1]]
            spot['y_refined'] = y_min + (yy * local).sum() / total
            spot['x_refined'] = x_min + (xx * local).sum() / total
        else:
            spot['y_refined'] = spot['y']
            spot['x_refined'] = spot['x']
    
    elapsed = (time.time() - start) * 1000
    return spots, elapsed


def quick_check_overlap(region, threshold=30):
    """
    快速检查区域是否可能有重叠光斑
    通过分析形状椭圆率和峰值数量来判断
    """
    # 背景减除
    bg = np.percentile(region, 20)
    region_sub = np.maximum(region.astype(np.float64) - bg, 0)
    
    if region_sub.max() < threshold:
        return False, 0
    
    # 计算二阶矩，判断椭圆率
    h, w = region.shape
    yy, xx = np.mgrid[0:h, 0:w]
    
    total = region_sub.sum()
    if total < 1:
        return False, 1
    
    cy = (yy * region_sub).sum() / total
    cx = (xx * region_sub).sum() / total
    
    myy = ((yy - cy)**2 * region_sub).sum() / total
    mxx = ((xx - cx)**2 * region_sub).sum() / total
    mxy = ((xx - cx) * (yy - cy) * region_sub).sum() / total
    
    # 椭圆长短轴
    temp = np.sqrt((myy - mxx)**2 + 4 * mxy**2)
    a = np.sqrt(max(0, 2 * (myy + mxx + temp)))
    b = np.sqrt(max(0, 2 * (myy + mxx - temp)))
    
    elongation = a / (b + 0.1)
    
    # 椭圆率>1.15说明可能有重叠 (进一步降低以检测轻微拉长的双光斑)
    if elongation > 1.15:
        return True, 2
    
    # 检查是否有多个明显峰值
    smoothed = ndimage.gaussian_filter(region_sub, sigma=3)
    local_max = ndimage.maximum_filter(smoothed, size=15)  # 减小邻域
    peak_mask = (smoothed == local_max) & (smoothed > smoothed.max() * 0.35)  # 降低阈值
    n_peaks = np.sum(peak_mask)
    
    if n_peaks >= 2:
        return True, n_peaks
    
    return False, 1


def split_along_axis(region, threshold=25):
    """
    沿主轴方向分割哑铃形光斑
    
    当形状拉长但峰值检测失败时，使用几何方法分割
    
    Returns:
        list of spots: 分割后的光斑列表 (1或2个)
    """
    bg = np.percentile(region, 20)
    region_sub = np.maximum(region.astype(np.float64) - bg, 0)
    
    if region_sub.max() < threshold:
        return []
    
    h, w = region.shape
    yy, xx = np.mgrid[0:h, 0:w]
    
    total = region_sub.sum()
    if total < 1:
        return []
    
    # 计算质心
    cy = (yy * region_sub).sum() / total
    cx = (xx * region_sub).sum() / total
    
    # 计算二阶矩
    myy = ((yy - cy)**2 * region_sub).sum() / total
    mxx = ((xx - cx)**2 * region_sub).sum() / total
    mxy = ((xx - cx) * (yy - cy) * region_sub).sum() / total
    
    # 计算特征值和特征向量 (主轴方向)
    cov_matrix = np.array([[mxx, mxy], [mxy, myy]])
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    
    # 主轴方向 (最大特征值对应的特征向量)
    major_axis_idx = np.argmax(eigenvalues)
    major_axis = eigenvectors[:, major_axis_idx]  # [dx, dy]
    
    # 沿主轴方向的标准差
    major_sigma = np.sqrt(eigenvalues[major_axis_idx])
    minor_sigma = np.sqrt(eigenvalues[1 - major_axis_idx])
    
    elongation = major_sigma / (minor_sigma + 0.1)
    
    if elongation < 1.2:
        # 不够拉长，返回单个质心
        return [{'y': cy, 'x': cx, 'amplitude': region_sub.max(), 'sigma': 10}]
    
    # 沿主轴方向分割：在质心两侧各偏移一个距离
    offset = major_sigma * 0.5  # 偏移量
    
    # 计算两个中心点
    p1_x = cx + offset * major_axis[0]
    p1_y = cy + offset * major_axis[1]
    p2_x = cx - offset * major_axis[0]
    p2_y = cy - offset * major_axis[1]
    
    # 确保在边界内
    p1_x = np.clip(p1_x, 2, w-3)
    p1_y = np.clip(p1_y, 2, h-3)
    p2_x = np.clip(p2_x, 2, w-3)
    p2_y = np.clip(p2_y, 2, h-3)
    
    # 在两个位置计算局部强度
    amp1 = region_sub[int(p1_y), int(p1_x)] if 0 <= int(p1_y) < h and 0 <= int(p1_x) < w else 0
    amp2 = region_sub[int(p2_y), int(p2_x)] if 0 <= int(p2_y) < h and 0 <= int(p2_x) < w else 0
    
    spots = []
    if amp1 > threshold * 0.5:
        spots.append({'y': p1_y, 'x': p1_x, 'amplitude': amp1, 'sigma': 10})
    if amp2 > threshold * 0.5:
        spots.append({'y': p2_y, 'x': p2_x, 'amplitude': amp2, 'sigma': 10})
    
    if len(spots) == 0:
        # 分割失败，返回质心
        return [{'y': cy, 'x': cx, 'amplitude': region_sub.max(), 'sigma': 10}]
    
    return spots


def analyze_overlap_watershed(region, min_distance=10):
    """
    分水岭分割法分离重叠光斑
    
    优势: 形态学方法，不依赖高斯假设和BIC阈值
    
    流程:
    1. 二值化 → 前景掩码
    2. 距离变换 → 每个像素到边界的距离
    3. 寻找局部最大值作为分水岭种子
    4. 分水岭分割
    5. 计算每个区域的质心
    """
    from skimage.segmentation import watershed
    from skimage.feature import peak_local_max
    
    h, w = region.shape
    region_float = region.astype(np.float64)
    
    # 1. 背景减除 + 归一化
    bg = np.percentile(region, 20)
    img_sub = np.maximum(region_float - bg, 0)
    if img_sub.max() == 0:
        return [{'y': h/2, 'x': w/2, 'amplitude': 0, 'sigma': 10}], 1
    
    img_norm = img_sub / img_sub.max()
    
    # 2. 二值化 (阈值 = 15% 最大值，更低以捕捉更多)
    binary = img_norm > 0.15
    
    # 3. 距离变换
    distance = ndimage.distance_transform_edt(binary)
    
    # 4. 寻找局部最大值作为种子点
    # min_distance 控制最小分离距离
    coords = peak_local_max(distance, min_distance=min_distance, labels=binary)
    
    if len(coords) == 0:
        # 没有找到峰值，计算质心返回
        total = img_sub.sum()
        if total > 0:
            yy, xx = np.mgrid[0:h, 0:w]
            cy = (yy * img_sub).sum() / total
            cx = (xx * img_sub).sum() / total
        else:
            cy, cx = h/2, w/2
        return [{'y': cy, 'x': cx, 'amplitude': img_sub.max(), 'sigma': 10}], 1
    
    if len(coords) == 1:
        # 只有一个峰值
        cy, cx = coords[0]
        return [{'y': float(cy), 'x': float(cx), 'amplitude': img_sub[cy, cx], 'sigma': 10}], 1
    
    # 5. 创建标记图像
    markers = np.zeros_like(distance, dtype=np.int32)
    for i, (y, x) in enumerate(coords):
        markers[y, x] = i + 1
    
    # 6. 分水岭分割 (使用距离变换的负值作为地形)
    labels = watershed(-distance, markers, mask=binary)
    
    # 7. 计算每个区域的质心和强度
    spots = []
    yy, xx = np.mgrid[0:h, 0:w]
    
    for label_id in range(1, labels.max() + 1):
        mask = labels == label_id
        region_pixels = img_sub[mask]
        
        if region_pixels.sum() == 0:
            continue
        
        # 加权质心
        weights = img_sub * mask
        total = weights.sum()
        cy = (yy * weights).sum() / total
        cx = (xx * weights).sum() / total
        amplitude = region_pixels.max()
        
        # 估算 sigma (等效圆半径)
        area = np.sum(mask)
        sigma = np.sqrt(area / np.pi)
        
        spots.append({
            'y': float(cy),
            'x': float(cx),
            'amplitude': float(amplitude),
            'sigma': float(sigma)
        })
    
    if len(spots) == 0:
        return [{'y': h/2, 'x': w/2, 'amplitude': 0, 'sigma': 10}], 1
    
    # 按振幅排序，返回前2个
    spots = sorted(spots, key=lambda s: s['amplitude'], reverse=True)[:2]
    
    return spots, len(spots)


def moffat_2_model_local(params, xx, yy, h, w):
    """双 Moffat 模型辅助函数"""
    bg = params[0]
    model = np.full((h, w), bg, dtype=np.float64)
    for i in range(2):
        idx = 1 + i * 5
        amp, cy, cx, alpha, beta = params[idx:idx+5]
        r2 = (xx - cx)**2 + (yy - cy)**2
        model += amp * (1 + r2 / (alpha**2 + 1e-6))**(-beta)
    return model


def analyze_overlap_fast(region, p1_init, p2_init=None, alpha_init=12, beta_init=2.5):
    """
    简化版光斑定位：使用强度加权质心 + 高斯加权精修
    
    相比复杂的 Moffat 拟合，这个方法：
    1. 更稳定：不依赖非线性优化的收敛
    2. 更快速：纯向量计算
    3. 更准确：直接对准光斑强度中心
    """
    h, w = region.shape
    yy, xx = np.mgrid[0:h, 0:w]
    region_float = region.astype(np.float64)
    
    # 背景估算
    bg = np.percentile(region_float, 20)
    region_sub = np.maximum(region_float - bg, 0)
    
    def compute_weighted_centroid(y_init, x_init, box=7, sigma=2.0):
        """高斯加权质心计算"""
        half = box // 2
        y_min, y_max = max(0, int(y_init) - half), min(h, int(y_init) + half + 1)
        x_min, x_max = max(0, int(x_init) - half), min(w, int(x_init) + half + 1)
        
        sub = region_sub[y_min:y_max, x_min:x_max]
        if sub.sum() < 1:
            return y_init, x_init, 0
        
        yy_sub, xx_sub = np.mgrid[0:sub.shape[0], 0:sub.shape[1]]
        
        # 高斯权重（以初始位置为中心）
        y_center, x_center = y_init - y_min, x_init - x_min
        gauss_weight = np.exp(-((yy_sub - y_center)**2 + (xx_sub - x_center)**2) / (2 * sigma**2))
        weighted = sub * gauss_weight
        total = weighted.sum()
        
        if total < 1:
            return y_init, x_init, sub.max()
        
        cy = (yy_sub * weighted).sum() / total + y_min
        cx = (xx_sub * weighted).sum() / total + x_min
        
        return cy, cx, sub.max()
    
    # 计算第一个光斑的精确位置
    cy1, cx1, amp1 = compute_weighted_centroid(p1_init['y'], p1_init['x'])
    
    # 如果有第二个光斑
    if p2_init:
        cy2, cx2, amp2 = compute_weighted_centroid(p2_init['y'], p2_init['x'])
        
        dist = np.sqrt((cy1 - cy2)**2 + (cx1 - cx2)**2)
        
        # 只有当两个光斑距离足够远且都有一定强度时才返回两个
        if dist > 8 and amp2 > 20:
            return [
                {'y': cy1, 'x': cx1, 'amplitude': amp1, 'sigma': alpha_init},
                {'y': cy2, 'x': cx2, 'amplitude': amp2, 'sigma': alpha_init}
            ], 2
    
    # 返回单光斑
    return [{'y': cy1, 'x': cx1, 'amplitude': amp1, 'sigma': alpha_init}], 1


def analyze_overlap(region, sigma_init=12, max_n=3):
    """
    分析区域内是否有重叠光斑，使用BIC选择最佳模型
    
    Returns:
        spots: 分离后的光斑列表
        best_n: 最佳高斯数量
    """
    h, w = region.shape
    yy, xx = np.mgrid[0:h, 0:w]
    region_flat = region.astype(np.float64).ravel()
    
    results = {}
    
    for n in range(1, max_n + 1):
        bg0 = np.percentile(region, 20)
        
        # 初始参数
        p0 = [bg0]
        lower, upper = [0], [500]
        
        for i in range(n):
            cx = w * (i + 1) / (n + 1)
            cy = h / 2
            amp = (region.max() - bg0) / n
            p0.extend([amp, cy, cx, sigma_init])
            lower.extend([0, 0, 0, 1])
            upper.extend([500, h, w, 50])
        
        def residual(params):
            bg = params[0]
            model = np.full((h, w), bg, dtype=np.float64)
            for i in range(n):
                idx = 1 + i * 4
                amp, cy, cx, sigma = params[idx:idx+4]
                model += amp * np.exp(-((xx - cx)**2 + (yy - cy)**2) / (2 * sigma**2))
            return (model.ravel() - region_flat)
        
        try:
            result = least_squares(residual, p0, bounds=(lower, upper), max_nfev=300)
            
            rss = np.sum(result.fun**2)
            k = 1 + n * 4
            n_pixels = h * w
            bic = n_pixels * np.log(rss / n_pixels + 1e-10) + k * np.log(n_pixels)
            
            spots = []
            for i in range(n):
                idx = 1 + i * 4
                amp, cy, cx, sigma = result.x[idx:idx+4]
                if amp > 5:
                    spots.append({'y': cy, 'x': cx, 'amplitude': amp, 'sigma': sigma})
            
            results[n] = {'spots': spots, 'bic': bic}
        except:
            results[n] = {'spots': [], 'bic': float('inf')}
    
    best_n = min(results.keys(), key=lambda n: results[n]['bic'])
    return results[best_n]['spots'], best_n


# ================== 极限测试 ==================

def test_extreme_overlap():
    """测试极近距离(哑铃形状)的分离能力"""
    
    print("=" * 60)
    print("极限重叠测试 (哑铃形状)")
    print("=" * 60)
    
    # 创建测试图像
    size = (100, 100)
    sigma = 12
    yy, xx = np.mgrid[0:size[0], 0:size[1]]
    
    test_distances = [5, 6, 7, 8, 9, 10]  # 极近距离
    results = []
    
    for dist in test_distances:
        # 生成两个重叠光斑
        center = 50
        image = 5 + 200 * np.exp(-((xx - center + dist/2)**2 + (yy - center)**2) / (2 * sigma**2))
        image += 180 * np.exp(-((xx - center - dist/2)**2 + (yy - center)**2) / (2 * sigma**2))
        image = np.clip(image, 0, 255).astype(np.uint8)
        
        # 尝试分离
        spots, best_n = analyze_overlap(image, sigma_init=sigma)
        
        success = len(spots) == 2
        if success:
            det_dist = abs(spots[0]['x'] - spots[1]['x'])
            error = abs(det_dist - dist)
        else:
            error = None
        
        results.append({'dist': dist, 'success': success, 'best_n': best_n, 'error': error})
        
        status = "✓" if success else "✗"
        err_str = f"{error:.2f}px" if error is not None else "-"
        print(f"  距离={dist}px: {status} 检测={best_n}个高斯, 误差={err_str}")
    
    # 可视化最后一个
    visualize_extreme_overlap(image, spots, test_distances[-1])
    
    return results


def visualize_extreme_overlap(image, spots, dist):
    """可视化极限重叠测试"""
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    ax1 = axes[0]
    im1 = ax1.imshow(image, cmap='hot', origin='lower')
    ax1.set_title(f'哑铃形重叠光斑\n(距离={dist}px, sigma=12)')
    plt.colorbar(im1, ax=ax1)
    
    ax2 = axes[1]
    im2 = ax2.imshow(image, cmap='hot', origin='lower')
    ax2.set_title(f'分离结果: {len(spots)}个')
    
    for s in spots:
        ax2.plot(s['x'], s['y'], 'c+', markersize=15, markeredgewidth=2)
        circle = plt.Circle((s['x'], s['y']), s['sigma'],
                            fill=False, color='cyan', linewidth=2)
        ax2.add_patch(circle)
    plt.colorbar(im2, ax=ax2)
    
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "extreme_overlap_test.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n保存到: {save_path}")
    print(f"\n保存到: {save_path}")
    print(f"\n保存到: {save_path}")
    plt.close()


def refine_peak_local(region, y_int, x_int, box=5):
    """局部质心精修，解决直接使用峰值坐标不准的问题"""
    h, w = region.shape
    half = box // 2
    y_min, y_max = max(0, y_int - half), min(h, y_int + half + 1)
    x_min, x_max = max(0, x_int - half), min(w, x_int + half + 1)
    
    sub = region[int(y_min):int(y_max), int(x_min):int(x_max)].astype(np.float64)
    total = sub.sum()
    if total <= 0:
        return y_int, x_int
        
    yy, xx = np.mgrid[0:sub.shape[0], 0:sub.shape[1]]
    cy = (yy * sub).sum() / total
    cx = (xx * sub).sum() / total
    
    return y_min + cy, x_min + cx


def compute_gauss_centroid(region, y_init, x_init, box=9, sigma=3.0):
    """
    高斯加权质心计算 - 比简单质心更准确
    
    Args:
        region: 区域图像
        y_init, x_init: 初始估计位置
        box: 计算窗口大小
        sigma: 高斯权重的标准差
    
    Returns:
        cy, cx, amplitude: 精修后的坐标和振幅
    """
    h, w = region.shape
    region_float = region.astype(np.float64)
    
    # 背景估算
    bg = np.percentile(region_float, 20)
    region_sub = np.maximum(region_float - bg, 0)
    
    half = box // 2
    y_min = max(0, int(y_init) - half)
    y_max = min(h, int(y_init) + half + 1)
    x_min = max(0, int(x_init) - half)
    x_max = min(w, int(x_init) + half + 1)
    
    sub = region_sub[y_min:y_max, x_min:x_max]
    if sub.sum() < 1:
        return y_init, x_init, 0
    
    yy_sub, xx_sub = np.mgrid[0:sub.shape[0], 0:sub.shape[1]]
    
    # 高斯权重（以初始位置为中心）
    y_center, x_center = y_init - y_min, x_init - x_min
    gauss_weight = np.exp(-((yy_sub - y_center)**2 + (xx_sub - x_center)**2) / (2 * sigma**2))
    weighted = sub * gauss_weight
    total = weighted.sum()
    
    if total < 1:
        return y_init, x_init, sub.max()
    
    cy = (yy_sub * weighted).sum() / total + y_min
    cx = (xx_sub * weighted).sum() / total + x_min
    
    return cy, cx, sub.max()


def find_peaks_in_region(region, threshold=30, min_dist=15):
    """在小区域内寻找局部峰值"""
    # 确保是浮点数并减去背景
    bg = np.percentile(region, 20)
    region_sub = np.maximum(region.astype(np.float64) - bg, 0)
    
    # 动态阈值：区域最大值的 15% (降低以检测更多双光斑)
    dynamic_thresh = max(threshold, region_sub.max() * 0.15)
    
    # 局部最大值滤波 - 使用较小的邻域以检测近距离光斑
    max_filtered = ndimage.maximum_filter(region_sub, size=min_dist)
    mask = (region_sub == max_filtered) & (region_sub > dynamic_thresh)
    
    # 获取坐标
    peaks_y, peaks_x = np.where(mask)
    
    results = []
    for y, x in zip(peaks_y, peaks_x):
        # 边界保护
        if y < 3 or x < 3 or y >= region.shape[0]-3 or x >= region.shape[1]-3:
            continue
        results.append({
            'y': float(y),
            'x': float(x),
            'amplitude': float(region_sub[y, x]),
            'sigma': 12.0
        })
    
    # 按振幅排序
    results.sort(key=lambda s: s['amplitude'], reverse=True)
    return results


def deduplicate_spots(spots, min_dist=5.0):
    """去除重复检测的光斑"""
    if not spots:
        return []
    
    # 按振幅降序排序，优先保留强的
    sorted_spots = sorted(spots, key=lambda s: s['amplitude'], reverse=True)
    kept_spots = []
    
    for spot in sorted_spots:
        is_duplicate = False
        for kept in kept_spots:
            dist = np.sqrt((spot['x'] - kept['x'])**2 + (spot['y'] - kept['y'])**2)
            if dist < min_dist:
                is_duplicate = True
                break
        
        if not is_duplicate:
            kept_spots.append(spot)
            
    return kept_spots


# ================== 主程序 ==================

def process_image(image_path, visualize=True, detect_overlap=True):
    """
    处理单张图像 - 完整流程
    
    Args:
        image_path: 图像路径
        visualize: 是否生成可视化
        detect_overlap: 是否检测重叠
    
    Returns:
        spots: 检测到的光斑列表
        timing: 各阶段耗时
    """
    timing = {}
    
    # 加载图像
    start = time.time()
    image = np.array(Image.open(image_path))
    timing['加载'] = (time.time() - start) * 1000
    
    print(f"图像大小: {image.shape}")
    print(f"加载时间: {timing['加载']:.1f}ms")
    
    # 第一步: 粗检测子眼中心
    spots_coarse, timing['粗检测'] = detect_spots_fast(image, min_distance=150, threshold=50) # 放宽一点，靠后面去重
    print(f"粗检测初始: {len(spots_coarse)} 个子眼区域, 耗时 {timing['粗检测']:.1f}ms")
    
    # 强制粗检测去重 (核心修复: 防止一个透镜出两个中心)
    spots_coarse = deduplicate_spots(spots_coarse, min_dist=100.0)
    print(f"粗检测去重: 剩余 {len(spots_coarse)} 个有效区域")
    
    if detect_overlap:
        # 第二步: 快速筛查 + 选择性高斯拟合
        start = time.time()
        all_spots = []
        lenslet_size = 200
        n_gaussian_fits = 0  # 统计做了多少次高斯拟合
        
        for spot in spots_coarse:
            y_c, x_c = int(spot['y']), int(spot['x'])
            
            half = lenslet_size // 2
            y_min = max(0, y_c - half)
            y_max = min(image.shape[0], y_c + half)
            x_min = max(0, x_c - half)
            x_max = min(image.shape[1], x_c + half)
            
            region = image[y_min:y_max, x_min:x_max]
            
            if region.max() < 30:
                continue
            
            # 使用新的峰值查找逻辑
            peaks = find_peaks_in_region(region, threshold=25, min_dist=15)  # 第一轮：较大间距
            
            valid_spots = []
            
            if len(peaks) == 0:
                continue
            
            # 如果只找到1个峰，检查形状是否是哑铃形
            if len(peaks) == 1:
                is_dumbbell, elongation = quick_check_overlap(region, threshold=25)
                if is_dumbbell:
                    # 形状拉长，用更小的min_dist重新检测
                    peaks = find_peaks_in_region(region, threshold=15, min_dist=5)
            
            # 处理峰值
            if len(peaks) == 0:
                continue
            elif len(peaks) == 1:
                # 再次检查形状，如果是哑铃形，使用轴向分割
                is_dumbbell, elongation = quick_check_overlap(region, threshold=25)
                if is_dumbbell:
                    # 使用几何方法沿主轴分割
                    n_gaussian_fits += 1
                    axis_spots = split_along_axis(region, threshold=20)
                    for s in axis_spots:
                        if s.get('amplitude', 0) > 10:
                            valid_spots.append(s)
                else:
                    # 单光斑：精修质心
                    n_gaussian_fits += 1
                    cy, cx, amp = compute_gauss_centroid(region, peaks[0]['y'], peaks[0]['x'])
                    valid_spots.append({
                        'y': cy, 'x': cx, 
                        'amplitude': amp, 
                        'sigma': peaks[0].get('sigma', 12)
                    })
            else:
                # 多个峰值：检查前两个是否都是有效光斑
                p1, p2 = peaks[0], peaks[1]
                dist = np.sqrt((p1['y']-p2['y'])**2 + (p1['x']-p2['x'])**2)
                amp_ratio = p2['amplitude'] / p1['amplitude']  # 第二个相对第一个的强度
                
                if dist > 10 and amp_ratio > 0.25:
                    # 确实是两个光斑，分别精修
                    n_gaussian_fits += 2
                    for p in [p1, p2]:
                        cy, cx, amp = compute_gauss_centroid(region, p['y'], p['x'])
                        valid_spots.append({
                            'y': cy, 'x': cx,
                            'amplitude': amp,
                            'sigma': p.get('sigma', 12)
                        })
                else:
                    # 距离太近或强度差太大 -> 只保留最强的
                    n_gaussian_fits += 1
                    cy, cx, amp = compute_gauss_centroid(region, p1['y'], p1['x'])
                    valid_spots.append({
                        'y': cy, 'x': cx,
                        'amplitude': amp,
                        'sigma': p1.get('sigma', 12)
                    })
            
            # 转换为全局坐标并添加到列表
            for s in valid_spots:
                all_spots.append({
                    'y': y_min + s['y'],
                    'x': x_min + s['x'],
                    'y_refined': y_min + s['y'],
                    'x_refined': x_min + s['x'],
                    'amplitude': s.get('amplitude', 100),
                    'sigma': s.get('sigma', 12)
                })
        
        # 关键步骤：全局去重，解决远距离光斑被多次检测的问题
        before_dedup = len(all_spots)
        spots = deduplicate_spots(all_spots, min_dist=5.0)
        
        timing['分离检测'] = (time.time() - start) * 1000
        print(f"分离检测: 原始 {before_dedup} -> 去重后 {len(spots)} 个光斑 ({n_gaussian_fits}次高斯拟合), 耗时 {timing['分离检测']:.1f}ms")
    else:
        # 只做质心精修
        spots, timing['精修'] = refine_centroids_batch(image, spots_coarse)
        print(f"质心精修耗时: {timing['精修']:.1f}ms")
    
    # 总时间
    total = sum(timing.values())
    fps = 1000 / total if total > 0 else 0
    
    print(f"\n总处理时间: {total:.1f}ms ({fps:.1f} FPS)")
    
    # 可视化
    if visualize:
        visualize_detection(image, spots, image_path)
    
    return spots, timing


def visualize_detection(image, spots, image_path):
    """可视化检测结果 - 原图与检测结果并排对比"""
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # 降采样显示（大图像）
    downsample = max(1, image.shape[0] // 1200)
    display = image[::downsample, ::downsample]
    
    # 左图: 原始图像
    ax1 = axes[0]
    ax1.imshow(display, cmap='gray', origin='lower')
    ax1.set_title('原始图像', fontsize=14)
    ax1.axis('off')
    
    # 右图: 检测结果叠加
    ax2 = axes[1]
    ax2.imshow(display, cmap='gray', origin='lower')
    ax2.set_title(f'检测结果 ({len(spots)}个光斑)', fontsize=14)
    ax2.axis('off')
    
    # 在右图上标记检测到的光斑
    for s in spots:
        x = s.get('x_refined', s['x']) / downsample
        y = s.get('y_refined', s['y']) / downsample
        ax2.plot(x, y, 'r+', markersize=6, markeredgewidth=1.5)
    
    plt.tight_layout()
    
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    save_path = os.path.join(OUTPUT_DIR, f"{base_name}_result.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"保存到: {save_path}")
    plt.close()


def batch_process(input_dir, output_dir, limit=None):
    """批量处理文件夹中的所有图像"""
    import glob
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 查找所有bmp图像
    patterns = [os.path.join(input_dir, '*.bmp'), os.path.join(input_dir, '*.BMP')]
    image_files = []
    for pattern in patterns:
        image_files.extend(glob.glob(pattern))
    
    if not image_files:
        print(f"错误: 在 {input_dir} 中没有找到BMP图像")
        return
    
    total_files = len(image_files)
    if limit:
        image_files = image_files[:limit]
        print(f"找到 {total_files} 张图像, 将处理前 {len(image_files)} 张 (限制设定)")
    else:
        print(f"找到 {total_files} 张图像")
        
    print(f"输出目录: {output_dir}")
    print("=" * 60)
    
    results = []
    for i, image_path in enumerate(image_files):
        print(f"\n[{i+1}/{len(image_files)}] 处理: {os.path.basename(image_path)}")
        
        try:
            # 临时修改输出目录
            global OUTPUT_DIR
            old_output_dir = OUTPUT_DIR
            OUTPUT_DIR = output_dir
            
            spots, timing = process_image(image_path, visualize=True, detect_overlap=True)
            
            OUTPUT_DIR = old_output_dir
            
            results.append({
                'file': os.path.basename(image_path),
                'spots': len(spots),
                'time_ms': sum(timing.values())
            })
        except Exception as e:
            print(f"  错误: {e}")
            results.append({
                'file': os.path.basename(image_path),
                'spots': 0,
                'time_ms': 0,
                'error': str(e)
            })
    
    # 打印汇总
    print("\n" + "=" * 60)
    print("处理完成! 汇总:")
    print("=" * 60)
    for r in results:
        if 'error' in r:
            print(f"  {r['file']}: 错误 - {r['error']}")
        else:
            print(f"  {r['file']}: {r['spots']}个光斑, {r['time_ms']:.0f}ms")
    
    total_spots = sum(r['spots'] for r in results if 'error' not in r)
    total_time = sum(r['time_ms'] for r in results if 'error' not in r)
    print(f"\n总计: {total_spots}个光斑, 总耗时{total_time/1000:.1f}秒")


def main():
    parser = argparse.ArgumentParser(description='复眼光斑检测与分离')
    parser.add_argument('--image', type=str, help='输入图像路径')
    parser.add_argument('--batch', type=str, help='批量处理文件夹路径')
    parser.add_argument('--output', type=str, default='batch_output', help='批量处理输出目录')
    parser.add_argument('--limit', type=int, help='批量处理的数量限制')
    parser.add_argument('--test', action='store_true', help='运行极限重叠测试')
    parser.add_argument('--benchmark', action='store_true', help='运行速度基准测试')
    
    args = parser.parse_args()
    
    if args.test:
        test_extreme_overlap()
    elif args.batch:
        if os.path.isdir(args.batch):
            batch_process(args.batch, args.output, args.limit)
        else:
            print(f"错误: 找不到目录 {args.batch}")
    elif args.image:
        if os.path.exists(args.image):
            spots, timing = process_image(args.image)
        else:
            print(f"错误: 找不到文件 {args.image}")
    elif args.benchmark:
        # 使用示例图像
        test_path = r"焦距35mm,物距57.2cm1\ME2P-2621-15U3M NIR(GCE22110005)_2025-11-12_17_41_21_218-0.bmp"
        if os.path.exists(test_path):
            spots, timing = process_image(test_path)
        else:
            print("错误: 找不到测试图像")
    else:
        print("使用方法:")
        print("  python spot_detector.py --image <图像路径>  # 处理单张图像")
        print("  python spot_detector.py --batch <文件夹> --output <输出目录>  # 批量处理")
        print("  python spot_detector.py --test             # 运行极限重叠测试")
        print("  python spot_detector.py --benchmark        # 速度基准测试")


if __name__ == "__main__":
    main()
