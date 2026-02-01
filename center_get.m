clc; clear;
%% ================== 1. 读取单张 BMP 图像 ==================
bmp_fullpath = 'D:\老天爷下刀子吧\2025高阶项目\2026.0120\70，2.40.bmp';
IMG = imread(bmp_fullpath);
IMG = IMG(:,:,1);
[H, W] = size(IMG);

% 计算几何中心
cx = (W + 1) / 2;
cy = (H + 1) / 2;

%% ================== 2-5. 处理过程 (保持原始图像坐标) ==================
level = graythresh(IMG); 
BW = im2bw(IMG, level * 0.8); 
SE = strel('disk',2);
BW = imdilate(imerode(BW,SE),SE);

[L, num_spot] = bwlabel(BW);
if num_spot <= 1 || num_spot >= 400
    error('图像异常，连通域数量不合理');
end

% 计算初次质心
Centroid_raw = zeros(num_spot,3);   
for k = 1:num_spot
    [r,c] = find(L==k);
    idx = sub2ind(size(IMG),r,c);
    I = double(IMG(idx));
    x0 = sum(c .* I) / sum(I);
    y0 = sum(r .* I) / sum(I);
    Centroid_raw(k,:) = [x0, y0, sum(I)];
end

% 高斯精修
Centroid_gauss = zeros(num_spot,2);
for k = 1:num_spot
    Centroid_gauss(k,:) = gaussian_weighted_refine(IMG, L==k, Centroid_raw(k,1:2));
end

%% ================== 6. [修改] 坐标系转换 (原点居中，Y轴向上) ==================
% 转换公式：X_new = X - cx;  Y_new = -(Y - cy)
Centroid_raw_centered = [Centroid_raw(:,1)-cx, -(Centroid_raw(:,2)-cy)];
Centroid_gauss_centered = [Centroid_gauss(:,1)-cx, -(Centroid_gauss(:,2)-cy)];

%% ================== 7-8. 输出与评估 ==================
disp('====== 改后：单高斯拟合质心 (中心原点，Y轴向上) ======')
disp(Centroid_gauss_centered)

result_gauss = evaluate_spacing(Centroid_gauss_centered);
fprintf('平均间距: %.4f, 最大相对偏差: %.4f%%\n', ...
    result_gauss.mean_spacing, result_gauss.max_relative_deviation * 100);

%% ================== 9. [修改] 质心可视化 (纠正失真) ==================
figure('Name', '质心定位对比 (笛卡尔中心坐标系)', 'Color', 'w', 'Position', [100, 200, 1200, 500]);

% 预设绘图范围
x_range = (1:W) - cx;
y_range = -((1:H) - cy); % 对应 Y 轴翻转

% --- 全局视图 ---
subplot(1, 2, 1);
% 注意：flipud(IMG) 是为了让图像像素分布与“Y轴向上”的逻辑对应
imagesc(x_range, sort(y_range), flipud(IMG)); 
colormap gray; hold on;
plot(Centroid_raw_centered(:,1), Centroid_raw_centered(:,2), 'ro');
plot(Centroid_gauss_centered(:,1), Centroid_gauss_centered(:,2), 'b+');
title('全局视图 (Y轴已修正为向上)');
xlabel('X (pixels)'); ylabel('Y (pixels)');
axis image; % 关键：保持比例不失真
set(gca, 'YDir', 'normal'); % 确保 Y 轴从小到大向上排列

% --- 局部放大视图 ---
subplot(1, 2, 2);
zoom_x = Centroid_gauss_centered(1,1);
zoom_y = Centroid_gauss_centered(1,2);
win_size = 20;

imagesc(x_range, sort(y_range), flipud(IMG)); 
colormap gray; hold on;
plot(Centroid_raw_centered(:,1), Centroid_raw_centered(:,2), 'ro', 'MarkerSize', 12, 'LineWidth', 1.5);
plot(Centroid_gauss_centered(:,1), Centroid_gauss_centered(:,2), 'b+', 'MarkerSize', 12, 'LineWidth', 1.5);

% 设置坐标轴并纠正失真
xlim([zoom_x - win_size, zoom_x + win_size]);
ylim([zoom_y - win_size, zoom_y + win_size]);
axis image; % 关键：纠正“圆变椭圆”的问题
grid on;
title('光斑局部放大 (比例已纠正)');
xlabel('X'); ylabel('Y');
set(gca, 'YDir', 'normal');

%% ================== 函数定义 ==================
function centroid = gaussian_weighted_refine(IMG, mask, init_xy)
    sigma = 1.5;
    win = ceil(3*sigma);
    x0i = round(init_xy(1)); y0i = round(init_xy(2));
    xmin = max(x0i-win,1); xmax = min(x0i+win,size(IMG,2));
    ymin = max(y0i-win,1); ymax = min(y0i+win,size(IMG,1));
    ROI = double(IMG(ymin:ymax, xmin:xmax));
    bg = median(ROI(:));
    ROI = max(ROI - bg, 0);
    ROI_mask = mask(ymin:ymax, xmin:xmax);
    ROI = ROI .* double(ROI_mask);
    [X,Y] = meshgrid(xmin:xmax, ymin:ymax);
    W = exp(-((X-init_xy(1)).^2 + (Y-init_xy(2)).^2)/(2*sigma^2));
    Iw = ROI .* W;
    sumI = sum(Iw(:));
    if sumI < 1e-6 || isnan(sumI), centroid = init_xy; return; end
    xg = sum(X(:).*Iw(:)) / sumI;
    yg = sum(Y(:).*Iw(:)) / sumI;
    if hypot(xg-init_xy(1), yg-init_xy(2)) > 1.0, centroid = init_xy; return; end
    centroid = [xg, yg];
end

function result = evaluate_spacing(centroids)
    D = squareform(pdist(centroids));
    D(D==0) = inf;
    nearest_dist = min(D,[],2);
    m_dist = mean(nearest_dist);
    result.mean_spacing = m_dist;
    result.max_relative_deviation = max(abs(nearest_dist - m_dist) / m_dist);
    result.all_nearest_spacing = nearest_dist;
end