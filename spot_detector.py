"""
复眼光斑检测与分离 - DAOPHOT算法
================================
清晰版本，核心流程：
  1. 粗检测：降采样找子眼中心
  2. 精细分析：每个子眼内用BIC选择1/2高斯模型
  3. 质心精修：高斯加权质心

使用方法:
    conda activate myenv
    python spot_detector.py --image your_image.bmp
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
matplotlib.rcParams['axes.unicode_minus'] = False
from scipy import ndimage
from scipy.optimize import least_squares
import time
import argparse

# GPU支持（仅用于打印信息）
try:
    import torch
    if torch.cuda.is_available():
        print(f"✓ GPU可用: {torch.cuda.get_device_name(0)}")
except ImportError:
    pass

OUTPUT_DIR = "detection_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
#  第一步：粗检测 - 找子眼中心
# ============================================================

def detect_lenslet_centers(image, min_distance=180, threshold=50):
    """
    在全图中找到所有子眼的大致中心位置。
    
    方法：高斯平滑 → 降采样 → 局部最大值 → 原图精修
    
    Args:
        image: 5120x5120 灰度图像
        min_distance: 子眼间距（像素），约200px
        threshold: 最低亮度阈值
    
    Returns:
        centers: list of (y, x, intensity)
        elapsed_ms: 耗时
    """
    start = time.time()
    h, w = image.shape
    
    # 动态阈值
    img_valid = image[image > 0]
    if img_valid.size > 0:
        dynamic_threshold = max(threshold, np.percentile(img_valid, 99.9) * 0.15)
    else:
        dynamic_threshold = threshold
    
    # 高斯平滑去噪
    smoothed = ndimage.gaussian_filter(image.astype(np.float32), sigma=5)
    
    # 4x降采样加速
    ds = 4
    small = smoothed[::ds, ::ds]
    small_max = ndimage.maximum_filter(small, size=min_distance // ds)
    peak_mask = (small == small_max) & (small > dynamic_threshold)
    
    peaks = np.array(np.where(peak_mask)).T  # (N, 2)
    
    # 在原图精修位置
    centers = []
    border = 150  # 忽略边缘
    
    for py, px in peaks:
        y_c, x_c = py * ds, px * ds
        
        # 边缘过滤
        if y_c < border or y_c > h - border or x_c < border or x_c > w - border:
            continue
        
        # 在原图局部区域找精确峰值
        margin = ds * 2
        y0, y1 = max(0, y_c - margin), min(h, y_c + margin)
        x0, x1 = max(0, x_c - margin), min(w, x_c + margin)
        local = smoothed[y0:y1, x0:x1]
        
        if local.size == 0:
            continue
        
        local_pos = np.unravel_index(np.argmax(local), local.shape)
        y_exact = y0 + local_pos[0]
        x_exact = x0 + local_pos[1]
        
        if image[y_exact, x_exact] > dynamic_threshold:
            centers.append((float(y_exact), float(x_exact), float(image[y_exact, x_exact])))
    
    # 去重：距离太近的只保留最亮的
    centers = _deduplicate(centers, min_dist=100)
    
    elapsed = (time.time() - start) * 1000
    return centers, elapsed


def _deduplicate(centers, min_dist=100):
    """去除重复检测，保留最亮的"""
    if not centers:
        return []
    
    # 按亮度降序
    centers = sorted(centers, key=lambda c: c[2], reverse=True)
    kept = []
    
    for y, x, intensity in centers:
        is_dup = False
        for ky, kx, _ in kept:
            if np.sqrt((y - ky)**2 + (x - kx)**2) < min_dist:
                is_dup = True
                break
        if not is_dup:
            kept.append((y, x, intensity))
    
    return kept


# ============================================================
#  第二步：BIC高斯拟合 - 分离重叠光斑
# ============================================================

def check_needs_separation(region, threshold=25):
    """
    快速判断区域是否可能有多个光斑（通过形状椭圆率）。
    
    Returns:
        needs_fit: True=需要尝试双高斯拟合
        elongation: 椭圆率
    """
    bg = np.percentile(region, 20)
    region_sub = np.maximum(region.astype(np.float64) - bg, 0)
    
    if region_sub.max() < threshold:
        return False, 1.0
    
    h, w = region.shape
    yy, xx = np.mgrid[0:h, 0:w]
    total = region_sub.sum()
    
    if total < 1:
        return False, 1.0
    
    # 质心
    cy = (yy * region_sub).sum() / total
    cx = (xx * region_sub).sum() / total
    
    # 二阶矩 → 椭圆率
    myy = ((yy - cy)**2 * region_sub).sum() / total
    mxx = ((xx - cx)**2 * region_sub).sum() / total
    mxy = ((xx - cx) * (yy - cy) * region_sub).sum() / total
    
    temp = np.sqrt((myy - mxx)**2 + 4 * mxy**2)
    a2 = 2 * (myy + mxx + temp)
    b2 = 2 * (myy + mxx - temp)
    
    a = np.sqrt(max(0, a2))
    b = np.sqrt(max(0, b2))
    elongation = a / (b + 0.1)
    
    return elongation > 1.2, elongation


def fit_1_gaussian(region):
    """
    单高斯拟合。
    
    Returns:
        spots: [{'y', 'x', 'amplitude', 'sigma'}]
        bic: BIC值
    """
    h, w = region.shape
    yy, xx = np.mgrid[0:h, 0:w]
    data = region.astype(np.float64).ravel()
    
    # 初始参数
    bg0 = np.percentile(region, 20)
    amp0 = region.max() - bg0
    
    # 用质心作为初始位置
    sub = np.maximum(region.astype(np.float64) - bg0, 0)
    total = sub.sum()
    if total > 0:
        cy0 = (yy * sub).sum() / total
        cx0 = (xx * sub).sum() / total
    else:
        cy0, cx0 = h / 2, w / 2
    
    p0 = [bg0, amp0, cy0, cx0, 12.0]
    bounds_lo = [0, 0, 0, 0, 2]
    bounds_hi = [300, 500, h, w, 50]
    
    def residual(p):
        bg, amp, cy, cx, sigma = p
        model = bg + amp * np.exp(-((xx - cx)**2 + (yy - cy)**2) / (2 * sigma**2))
        return model.ravel() - data
    
    try:
        result = least_squares(residual, p0, bounds=(bounds_lo, bounds_hi), max_nfev=200)
        rss = np.sum(result.fun**2)
        n_pix = h * w
        bic = n_pix * np.log(rss / n_pix + 1e-10) + 5 * np.log(n_pix)
        
        bg, amp, cy, cx, sigma = result.x
        spots = [{'y': cy, 'x': cx, 'amplitude': amp, 'sigma': sigma}]
        return spots, bic
    except:
        return [{'y': cy0, 'x': cx0, 'amplitude': amp0, 'sigma': 12}], float('inf')


def fit_2_gaussians(region):
    """
    双高斯拟合。
    
    用多组初始值尝试，取最优结果。
    
    Returns:
        spots: [{'y', 'x', 'amplitude', 'sigma'}, ...]
        bic: BIC值
    """
    h, w = region.shape
    yy, xx = np.mgrid[0:h, 0:w]
    data = region.astype(np.float64).ravel()
    bg0 = np.percentile(region, 20)
    amp0 = region.max() - bg0
    
    # 用形状分析获取主轴方向，生成更好的初始猜测
    sub = np.maximum(region.astype(np.float64) - bg0, 0)
    total = sub.sum()
    if total > 0:
        cy_cm = (yy * sub).sum() / total
        cx_cm = (xx * sub).sum() / total
    else:
        cy_cm, cx_cm = h / 2, w / 2
    
    # 计算主轴方向
    myy = ((yy - cy_cm)**2 * sub).sum() / total if total > 0 else 1
    mxx = ((xx - cx_cm)**2 * sub).sum() / total if total > 0 else 1
    mxy = ((xx - cx_cm) * (yy - cy_cm) * sub).sum() / total if total > 0 else 0
    
    cov = np.array([[mxx, mxy], [mxy, myy]])
    eigvals, eigvecs = np.linalg.eig(cov)
    major_idx = np.argmax(eigvals)
    major_dir = eigvecs[:, major_idx]  # (dx, dy)
    major_sigma = np.sqrt(eigvals[major_idx])
    
    # 沿主轴偏移生成两个初始点
    offset = max(major_sigma * 0.5, 10)
    
    # 多组初始值
    init_configs = [
        # 沿主轴方向
        (cy_cm + offset * major_dir[1], cx_cm + offset * major_dir[0],
         cy_cm - offset * major_dir[1], cx_cm - offset * major_dir[0]),
        # 沿水平方向
        (cy_cm, cx_cm - offset, cy_cm, cx_cm + offset),
        # 沿垂直方向
        (cy_cm - offset, cx_cm, cy_cm + offset, cx_cm),
    ]
    
    bounds_lo = [0,  0, 0, 0, 2,  0, 0, 0, 2]
    bounds_hi = [300, 500, h, w, 50, 500, h, w, 50]
    
    best_result = None
    best_bic = float('inf')
    
    for cy1, cx1, cy2, cx2 in init_configs:
        # 限制在区域内
        cy1 = np.clip(cy1, 5, h-5)
        cx1 = np.clip(cx1, 5, w-5)
        cy2 = np.clip(cy2, 5, h-5)
        cx2 = np.clip(cx2, 5, w-5)
        
        p0 = [bg0, amp0/2, cy1, cx1, 10.0, amp0/2, cy2, cx2, 10.0]
        
        def residual(p):
            bg = p[0]
            model = np.full((h, w), bg, dtype=np.float64)
            for i in range(2):
                idx = 1 + i * 4
                amp, cy, cx, sigma = p[idx:idx+4]
                model += amp * np.exp(-((xx - cx)**2 + (yy - cy)**2) / (2 * sigma**2))
            return model.ravel() - data
        
        try:
            result = least_squares(residual, p0, bounds=(bounds_lo, bounds_hi), max_nfev=200)
            rss = np.sum(result.fun**2)
            n_pix = h * w
            bic = n_pix * np.log(rss / n_pix + 1e-10) + 9 * np.log(n_pix)
            
            if bic < best_bic:
                best_bic = bic
                best_result = result.x
        except:
            continue
    
    if best_result is None:
        return [], float('inf')
    
    # 提取光斑
    spots = []
    for i in range(2):
        idx = 1 + i * 4
        amp, cy, cx, sigma = best_result[idx:idx+4]
        if amp > 10:  # 振幅阈值
            spots.append({'y': cy, 'x': cx, 'amplitude': amp, 'sigma': sigma})
    
    # 验证：两个光斑不能太近
    if len(spots) == 2:
        dist = np.sqrt((spots[0]['y'] - spots[1]['y'])**2 + 
                       (spots[0]['x'] - spots[1]['x'])**2)
        if dist < 8:
            # 太近了，合并为一个
            stronger = max(spots, key=lambda s: s['amplitude'])
            spots = [stronger]
    
    return spots, best_bic


def analyze_lenslet(region, sigma_init=12):
    """
    分析单个子眼区域：判断是1个还是2个光斑。
    
    对每个子眼都做1高斯 vs 2高斯的BIC比较，
    选择更优的模型。
    
    Returns:
        spots: [{'y', 'x', 'amplitude', 'sigma'}, ...]
        n_found: 检测到的光斑数
        method: 使用的方法
    """
    bg = np.percentile(region, 20)
    if region.max() - bg < 20:
        return [], 0, 'skip'
    
    # 对全区域做BIC比较（1高斯 vs 2高斯）
    spots_1, bic_1 = fit_1_gaussian(region)
    spots_2, bic_2 = fit_2_gaussians(region)
    
    # BIC判断：2高斯要明显更好才选择
    if len(spots_2) == 2 and bic_2 < bic_1 - 20:
        return spots_2, 2, 'double'
    else:
        return spots_1, 1, 'single'


# ============================================================
#  主处理流程
# ============================================================

def process_image(image_path, visualize=True):
    """
    处理单张图像的完整流程。
    
    Returns:
        spots: 所有检测到的光斑（全局坐标）
        timing: 各阶段耗时
    """
    timing = {}
    
    # 加载图像
    start = time.time()
    image = np.array(Image.open(image_path))
    timing['加载'] = (time.time() - start) * 1000
    print(f"图像大小: {image.shape}")
    print(f"加载时间: {timing['加载']:.1f}ms")
    
    # 第一步：粗检测子眼中心
    centers, timing['粗检测'] = detect_lenslet_centers(image, min_distance=180, threshold=50)
    print(f"粗检测: {len(centers)} 个子眼, 耗时 {timing['粗检测']:.1f}ms")
    
    # 第二步：逐个子眼分析
    start = time.time()
    all_spots = []
    lenslet_size = 200
    stats = {'single': 0, 'double': 0, 'skip': 0}
    
    for y_c, x_c, _ in centers:
        y_c, x_c = int(y_c), int(x_c)
        half = lenslet_size // 2
        
        y0 = max(0, y_c - half)
        y1 = min(image.shape[0], y_c + half)
        x0 = max(0, x_c - half)
        x1 = min(image.shape[1], x_c + half)
        
        region = image[y0:y1, x0:x1]
        
        # 分析这个子眼
        sub_spots, n_found, method = analyze_lenslet(region)
        
        # 统计
        if n_found == 0:
            stats['skip'] += 1
        elif n_found == 1:
            stats['single'] += 1
        else:
            stats['double'] += 1
        
        # 转换为全局坐标
        for s in sub_spots:
            all_spots.append({
                'y': y0 + s['y'],
                'x': x0 + s['x'],
                'y_refined': y0 + s['y'],
                'x_refined': x0 + s['x'],
                'amplitude': s.get('amplitude', 100),
                'sigma': s.get('sigma', 12)
            })
    
    timing['精细分析'] = (time.time() - start) * 1000
    
    # 全局去重
    all_spots = _deduplicate_spots(all_spots, min_dist=8)
    
    print(f"精细分析: {len(all_spots)} 个光斑 "
          f"(单={stats['single']}, 双={stats['double']}, 跳过={stats['skip']}), "
          f"耗时 {timing['精细分析']:.1f}ms")
    
    # 总结
    total = sum(timing.values())
    fps = 1000 / total if total > 0 else 0
    print(f"\n总处理时间: {total:.1f}ms ({fps:.1f} FPS)")
    
    if visualize:
        visualize_detection(image, all_spots, image_path)
    
    return all_spots, timing


def _deduplicate_spots(spots, min_dist=8):
    """去重光斑"""
    if not spots:
        return []
    
    sorted_spots = sorted(spots, key=lambda s: s['amplitude'], reverse=True)
    kept = []
    
    for spot in sorted_spots:
        is_dup = False
        for k in kept:
            dist = np.sqrt((spot['y'] - k['y'])**2 + (spot['x'] - k['x'])**2)
            if dist < min_dist:
                is_dup = True
                break
        if not is_dup:
            kept.append(spot)
    
    return kept


# ============================================================
#  可视化
# ============================================================

def visualize_detection(image, spots, image_path):
    """原图 vs 检测结果对比"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # 降采样显示
    ds = max(1, image.shape[0] // 1200)
    display = image[::ds, ::ds]
    
    # 左：原图
    axes[0].imshow(display, cmap='gray', origin='lower')
    axes[0].set_title('原始图像', fontsize=14)
    axes[0].axis('off')
    
    # 右：检测结果
    axes[1].imshow(display, cmap='gray', origin='lower')
    axes[1].set_title(f'检测结果 ({len(spots)}个光斑)', fontsize=14)
    axes[1].axis('off')
    
    for s in spots:
        x = s.get('x_refined', s['x']) / ds
        y = s.get('y_refined', s['y']) / ds
        axes[1].plot(x, y, 'r+', markersize=6, markeredgewidth=1.5)
    
    plt.tight_layout()
    
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    save_path = os.path.join(OUTPUT_DIR, f"{base_name}_result.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"保存到: {save_path}")
    plt.close()


# ============================================================
#  极限重叠测试
# ============================================================

def test_extreme_overlap():
    """测试极近距离重叠的分离能力"""
    print("=" * 60)
    print("极限重叠测试")
    print("=" * 60)
    
    size = (100, 100)
    sigma = 12
    yy, xx = np.mgrid[0:size[0], 0:size[1]]
    
    for dist in [8, 12, 15, 20, 25, 30]:
        center = 50
        image = 5 + 200 * np.exp(-((xx - center + dist/2)**2 + (yy - center)**2) / (2 * sigma**2))
        image += 180 * np.exp(-((xx - center - dist/2)**2 + (yy - center)**2) / (2 * sigma**2))
        image = np.clip(image, 0, 255).astype(np.uint8)
        
        spots, n_found, method = analyze_lenslet(image, sigma_init=sigma)
        
        success = len(spots) == 2
        if success:
            det_dist = np.sqrt((spots[0]['y']-spots[1]['y'])**2 + (spots[0]['x']-spots[1]['x'])**2)
            error = abs(det_dist - dist)
            err_str = f"{error:.1f}px"
        else:
            err_str = "-"
        
        status = "✓" if success else "✗"
        print(f"  距离={dist:2d}px: {status} 检测={len(spots)}个, 方法={method}, 误差={err_str}")


# ============================================================
#  批量处理
# ============================================================

def batch_process(input_dir, output_dir, limit=None):
    """批量处理文件夹中的所有图像"""
    import glob
    
    os.makedirs(output_dir, exist_ok=True)
    
    image_files = glob.glob(os.path.join(input_dir, '*.bmp')) + \
                  glob.glob(os.path.join(input_dir, '*.BMP'))
    
    if not image_files:
        print(f"错误: 在 {input_dir} 中没有找到BMP图像")
        return
    
    if limit:
        image_files = image_files[:limit]
    
    print(f"找到 {len(image_files)} 张图像")
    print("=" * 60)
    
    global OUTPUT_DIR
    old_dir = OUTPUT_DIR
    OUTPUT_DIR = output_dir
    
    for i, path in enumerate(image_files):
        print(f"\n[{i+1}/{len(image_files)}] {os.path.basename(path)}")
        try:
            process_image(path, visualize=True)
        except Exception as e:
            print(f"  错误: {e}")
    
    OUTPUT_DIR = old_dir
    print("\n处理完成!")


# ============================================================
#  入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='复眼光斑检测与分离 (DAOPHOT)')
    parser.add_argument('--image', type=str, help='输入图像路径')
    parser.add_argument('--batch', type=str, help='批量处理文件夹路径')
    parser.add_argument('--output', type=str, default='batch_output', help='批量处理输出目录')
    parser.add_argument('--limit', type=int, help='批量处理数量限制')
    parser.add_argument('--test', action='store_true', help='运行极限重叠测试')
    
    args = parser.parse_args()
    
    if args.test:
        test_extreme_overlap()
    elif args.batch:
        batch_process(args.batch, args.output, args.limit)
    elif args.image:
        if os.path.exists(args.image):
            process_image(args.image)
        else:
            print(f"错误: 找不到文件 {args.image}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
