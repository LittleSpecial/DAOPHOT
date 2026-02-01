clc; clear;

%% ================== 1. 读取单张 BMP 图像 ==================
bmp_fullpath = 'D:\老天爷下刀子吧\2025高阶项目\2026.0120\70，2.40.bmp';
IMG = imread(bmp_fullpath);
IMG = IMG(:,:,1);

%% ================== 2. 二值化与形态学 ==================
%使用大津法自动计算阈值
level = graythresh(IMG); 
BW = im2bw(IMG, level * 0.8); % 取自动阈值的80%

SE = strel('disk',2);
BW = imdilate(imerode(BW,SE),SE);

%% ================== 3. 连通域 ==================
[L, num_spot] = bwlabel(BW);
if num_spot <= 1 || num_spot >= 400
    error('图像异常，连通域数量不合理');
end

%% ================== 4. 改前：强度加权质心 ==================
Centroid_raw = zeros(num_spot,3);   % [x y 能量]

for k = 1:num_spot
    [r,c] = find(L==k);
    idx = sub2ind(size(IMG),r,c);
    I = double(IMG(idx));

    x0 = sum(c .* I) / sum(I);
    y0 = sum(r .* I) / sum(I);

    Centroid_raw(k,:) = [x0, y0, sum(I)];
end

%% ================== 5. 改后：单高斯精修（带回退） ==================
Centroid_gauss = zeros(num_spot,2);

for k = 1:num_spot
    Centroid_gauss(k,:) = gaussian_weighted_refine( ...
    IMG, L==k, Centroid_raw(k,1:2));

       
end



%% ================== 6. 输出对比 ==================
disp('====== 改前：强度加权质心 ======')
disp(Centroid_raw(:,1:2))

disp('====== 改后：单高斯拟合质心 ======')
disp(Centroid_gauss(:,1:2))

%% ================== 7. 相邻间距精度评估 ==================
result_raw   = evaluate_spacing(Centroid_raw(:,1:2));
result_gauss = evaluate_spacing(Centroid_gauss(:,1:2));

disp('====== 改前（强度加权质心） ======')
disp(result_raw)

disp('====== 改后（单高斯拟合） ======')
disp(result_gauss)
%% ================== 8. 质心可视化 ==================
figure('Name', '质心定位对比可视化', 'Color', 'w', 'NumberTitle', 'off');

% --- 全局视图 ---
subplot(1, 2, 1);
imshow(IMG, []); hold on;
% 绘制初次质心(红色圆圈)
plot(Centroid_raw(:,1), Centroid_raw(:,2), 'ro', 'MarkerSize', 6, 'LineWidth', 1);
% 绘制精修质心(蓝色加号)
plot(Centroid_gauss(:,1), Centroid_gauss(:,2), 'b+', 'MarkerSize', 6, 'LineWidth', 1);
title(['全局视图 (检测到 ', num2str(num_spot), ' 个光斑)']);
legend('强度加权质心', '高斯精修质心', 'Location', 'northeastoutside');
axis on;

% --- 局部放大视图 (选取第一个光斑附近的区域) ---
subplot(1, 2, 2);
% 选取第一个光斑的坐标作为中心
zoom_x = Centroid_gauss(1,1);
zoom_y = Centroid_gauss(1,2);
win_size = 20; % 放大窗口大小

imshow(IMG, []); hold on;
plot(Centroid_raw(:,1), Centroid_raw(:,2), 'ro', 'MarkerSize', 12, 'LineWidth', 1.5);
plot(Centroid_gauss(:,1), Centroid_gauss(:,2), 'b+', 'MarkerSize', 12, 'LineWidth', 1.5);

% 设置坐标轴范围实现放大效果
xlim([zoom_x - win_size, zoom_x + win_size]);
ylim([zoom_y - win_size, zoom_y + win_size]);
grid on;
title('第一个光斑局部放大对比');
legend('强度加权', '高斯精修');

% 调整整体布局
set(gcf, 'Position', [100, 200, 1200, 500]);
function centroid = gaussian_weighted_refine(IMG, mask, init_xy)
% 高斯加权质心（推荐用于阵列光斑）

    %% 参数
    sigma = 1.5;
    win = ceil(3*sigma);

    %% ROI 裁剪
    x0i = round(init_xy(1));
    y0i = round(init_xy(2));

    xmin = max(x0i-win,1);
    xmax = min(x0i+win,size(IMG,2));
    ymin = max(y0i-win,1);
    ymax = min(y0i+win,size(IMG,1));

    ROI = double(IMG(ymin:ymax, xmin:xmax));

    %% 背景扣除
    bg = median(ROI(:));
    ROI = ROI - bg;
    ROI(ROI < 0) = 0;
    ROI_mask = mask(ymin:ymax, xmin:xmax);
    ROI = ROI .* double(ROI_mask);
    %% 坐标网格
    [X,Y] = meshgrid(xmin:xmax, ymin:ymax);

    %% 高斯权重（以粗质心为中心）
    W = exp(-((X-init_xy(1)).^2 + (Y-init_xy(2)).^2)/(2*sigma^2));

    %% 加权质心
    Iw = ROI .* W;

    sumI = sum(Iw(:));
    xg = sum(X(:).*Iw(:)) / sumI;
    yg = sum(Y(:).*Iw(:)) / sumI;
    if hypot(xg-init_xy(1), yg-init_xy(2)) > 1.0
    centroid = [init_xy(1), init_xy(2)];
    return;
end

if sumI < 1e-6 || isnan(sumI)
    centroid = [init_xy(1), init_xy(2)];
    return;
end
    centroid = [xg, yg];
end

function result = evaluate_spacing(centroids)
% 基于最近邻的相邻间距一致性评估

    N = size(centroids,1);

    %% 最近邻距离
    D = squareform(pdist(centroids));
    D(D==0) = inf;

    nearest_dist = min(D,[],2);

    %% 统计指标
    mean_dist = mean(nearest_dist);
    max_relative_deviation = ...
        max(abs(nearest_dist - mean_dist) / mean_dist);

    %% 输出
    result.mean_spacing = mean_dist;
    result.max_relative_deviation = max_relative_deviation;
    result.all_nearest_spacing = nearest_dist;
end
