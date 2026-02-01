function [labeled_centroids, average_depth, std_depth, final_3D_pos, correction_params] = depthCalculate_trycorrect(centroids, H, W, enable_correction)

% 输入:
%   centroids - N×2 质心坐标矩阵 [x, y]
%   H, W      - 图像高度和宽度
%   enable_correction - 是否启用旋转修正 (默认true)
% 输出:
%   labeled_centroids - N×6 标注质心 [bx, by, row, col, ax, ay]
%   average_depth     - 平均深度 (m)
%   std_depth         - 深度标准差 (m)
%   final_3D_pos      - 三维位置结构体
%   correction_params - 修正参数 {theta, center_x, center_y, fit_error}
% =========================================================================

    if nargin < 4
        enable_correction = true;
    end
    
    if ischar(H); H = str2double(H); end
    if ischar(W); W = str2double(W); end
    
    %% ========== 系统参数 ==========
    f = 35e-3;            % 焦距 (m)
    p_size = 2.5e-6;      % 像素尺寸 (m)
    dL = 1e-3;            % 微透镜节距 (m)
    S_avg_theory = 400;   % 理论平均间距 (pixels)
    N_spots = size(centroids, 1);
    
    fprintf('\n========== 开始深度计算 ==========\n');
    fprintf('质心数量: %d\n', N_spots);
    fprintf('旋转修正: %s\n\n', mat2str(enable_correction));
    
    %% ========== 步骤1: 质心编号（行列索引） ==========
    fprintf('[步骤1] 质心行列索引分配...\n');
    
    % X方向排序并分配列号
    [x_sorted, x_idx] = sort(centroids(:,1));
    col_tags = zeros(N_spots, 1);
    curr_col = 1;
    col_tags(x_idx(1)) = curr_col;
    for i = 2:N_spots
        if (x_sorted(i) - x_sorted(i-1)) > S_avg_theory * 0.6
            curr_col = curr_col + 1;
        end
        col_tags(x_idx(i)) = curr_col;
    end
    
    % Y方向排序并分配行号
    [y_sorted, y_idx] = sort(centroids(:,2));
    row_tags = zeros(N_spots, 1);
    curr_row = 1;
    row_tags(y_idx(1)) = curr_row;
    for i = 2:N_spots
        if (y_sorted(i) - y_sorted(i-1)) > S_avg_theory * 0.6
            curr_row = curr_row + 1;
        end
        row_tags(y_idx(i)) = curr_row;
    end
    
    % 构建标注矩阵
    labeled_centroids = zeros(N_spots, 6);
    for k = 1:N_spots
        bx = centroids(k, 1); 
        by = centroids(k, 2);
        rid = row_tags(k); 
        cid = col_tags(k);
        ax = S_avg_theory * cid - S_avg_theory/2;
        ay = S_avg_theory * rid - S_avg_theory/2;
        labeled_centroids(k, :) = [bx, by, rid, cid, ax, ay];
    end
    
    fprintf('  行数: %d, 列数: %d\n', max(row_tags), max(col_tags));
    
    %% ========== 步骤2: 旋转角度标定 ==========
    correction_params = struct('theta', 0, 'center_x', 0, 'center_y', 0, ...
                               'fit_error', 0, 'enabled', enable_correction);
    
    if enable_correction
        fprintf('\n[步骤2] 旋转角度标定...\n');
        
        % 构建理想网格坐标（相对于中心）
        row_center = mean(row_tags);
        col_center = mean(col_tags);
        
        ideal_x = (col_tags - col_center) * S_avg_theory;
        ideal_y = (row_tags - row_center) * S_avg_theory;
        
        % 观测坐标中心
        obs_center_x = mean(centroids(:,1));
        obs_center_y = mean(centroids(:,2));
        
        % 初始猜测（使用PCA快速估计）
        centroids_centered = centroids - [obs_center_x, obs_center_y];
        [coeff, ~, ~] = pca(centroids_centered);
        theta_init = atan2(coeff(2,1), coeff(1,1));
        
        fprintf('  PCA初始估计角度: %.2f°\n', rad2deg(theta_init));
        
        % 优化求解旋转参数
        costFun = @(params) computeRotationError(params, ideal_x, ideal_y, ...
                                                 centroids(:,1), centroids(:,2));
        
        options = optimoptions('lsqnonlin', 'Display', 'off', 'Algorithm', 'levenberg-marquardt');
        params_init = [theta_init, obs_center_x, obs_center_y];
        
        [params_opt, resnorm] = lsqnonlin(costFun, params_init, [], [], options);
        
        correction_params.theta = params_opt(1);
        correction_params.center_x = params_opt(2);
        correction_params.center_y = params_opt(3);
        correction_params.fit_error = sqrt(resnorm / N_spots);
        
        fprintf('  优化后旋转角度: %.2f°\n', rad2deg(correction_params.theta));
        fprintf('  旋转中心: (%.1f, %.1f)\n', correction_params.center_x, correction_params.center_y);
        fprintf('  拟合残差RMS: %.2f pixels\n', correction_params.fit_error);
        
        %% ========== 步骤3: 坐标修正 ==========
        fprintf('\n[步骤3] 应用旋转修正...\n');
        
        theta = correction_params.theta;
        cx = correction_params.center_x;
        cy = correction_params.center_y;
        
        % 逆旋转矩阵
        R_inv = [cos(-theta), -sin(-theta); 
                 sin(-theta),  cos(-theta)];
        
        corrected_centroids = labeled_centroids;
        for k = 1:N_spots
            obs_vec = [labeled_centroids(k,1) - cx; 
                       labeled_centroids(k,2) - cy];
            corr_vec = R_inv * obs_vec;
            
            corrected_centroids(k, 1:2) = corr_vec' + [cx, cy];
        end
        
        % 更新理想位置（基于修正后的坐标重新计算）
        for k = 1:N_spots
            rid = corrected_centroids(k, 3);
            cid = corrected_centroids(k, 4);
            corrected_centroids(k, 5) = S_avg_theory * cid - S_avg_theory/2;
            corrected_centroids(k, 6) = S_avg_theory * rid - S_avg_theory/2;
        end
        
        fprintf('  坐标修正完成\n');
    else
        corrected_centroids = labeled_centroids;
        fprintf('\n[步骤2-3] 跳过旋转修正\n');
    end
    
    %% ========== 步骤4: 深度计算 ==========
    fprintf('\n[步骤4] 视差深度计算...\n');
    
    depths_list = [];
    depth_pairs = []; % 存储 [i, j, Z_ij]
    
    for i = 1:(N_spots - 1)
        for j = (i + 1):N_spots
            % 行列索引差（物理基线）
            Idx1 = labeled_centroids(i, 3:4); 
            Idx2 = labeled_centroids(j, 3:4); 
            B = sqrt((Idx2(1)-Idx1(1))^2 + (Idx2(2)-Idx1(2))^2) * dL;
            
            % 使用修正后坐标计算视差
            dist_pix = sqrt((corrected_centroids(j,1) - corrected_centroids(i,1))^2 + ...
                            (corrected_centroids(j,2) - corrected_centroids(i,2))^2);
            D_meas = dist_pix * p_size;
            
            denominator = D_meas - B;
            
            if abs(denominator) > 1e-10
                Z_ij = (f * B) / denominator;
                if Z_ij > 0 && Z_ij < 50 
                    depths_list = [depths_list; Z_ij];
                    depth_pairs = [depth_pairs; i, j, Z_ij];
                end
            end
        end
    end
    
    average_depth = mean(depths_list);
    std_depth = std(depths_list);
    
    fprintf('  有效点对数: %d\n', length(depths_list));
    fprintf('  平均深度: %.4f m\n', average_depth);
    fprintf('  标准差: %.4f m (%.2f%%)\n', std_depth, 100*std_depth/average_depth);
    
    %% ========== 步骤5: 三维位置计算 ==========
    fprintf('\n[步骤5] 三维空间位置解算...\n');
    
    final_3D_pos = struct('X', NaN, 'Y', NaN, 'Z', average_depth);
    
    if ~isnan(average_depth)
        R = average_depth / (average_depth + f);
        X_pos_list = []; 
        Y_pos_list = [];
        
        for k = 1:N_spots
            xai = corrected_centroids(k, 1) * p_size;
            yai = corrected_centroids(k, 2) * p_size;
            xsi = corrected_centroids(k, 5) * p_size;
            ysi = corrected_centroids(k, 6) * p_size;
            
            X_pos_list = [X_pos_list; (R * xai - xsi) / (R - 1)];
            Y_pos_list = [Y_pos_list; (R * yai - ysi) / (R - 1)];
        end
        
        final_3D_pos.X = mean(X_pos_list);
        final_3D_pos.Y = mean(Y_pos_list);
        
        fprintf('  目标位置: X=%.4f m, Y=%.4f m, Z=%.4f m\n', ...
                final_3D_pos.X, final_3D_pos.Y, final_3D_pos.Z);
    end
    
    %% ========== 步骤6: 可视化 ==========
    fprintf('\n[步骤6] 生成可视化结果...\n');
    
    figure('Color', 'w', 'Position', [100, 100, 1400, 600]);
    
    % ===== 子图1: 原始质心 =====
    subplot(1, 3, 1);
    hold on; box on;
    scatter(labeled_centroids(:,1), labeled_centroids(:,2), 80, 'b', 'filled', 'MarkerFaceAlpha', 0.6);
    
    for k = 1:N_spots
        text(labeled_centroids(k,1)+15, labeled_centroids(k,2)+15, ...
            sprintf('(%d,%d)', labeled_centroids(k,3), labeled_centroids(k,4)), ...
            'FontSize', 7, 'Color', 'r');
    end
    
    if enable_correction
        % 绘制旋转中心
        plot(correction_params.center_x, correction_params.center_y, 'rx', 'MarkerSize', 15, 'LineWidth', 2);
        
        % 绘制旋转方向指示
        arrow_len = 100;
        theta = correction_params.theta;
        cx = correction_params.center_x;
        cy = correction_params.center_y;
        
        quiver(cx, cy, arrow_len*cos(theta), arrow_len*sin(theta), ...
               'r', 'LineWidth', 2, 'MaxHeadSize', 2);
        text(cx + arrow_len*cos(theta)*1.3, cy + arrow_len*sin(theta)*1.3, ...
             sprintf('θ=%.1f°', rad2deg(theta)), 'FontSize', 10, 'Color', 'r', 'FontWeight', 'bold');
    end
    
    title('原始质心分布');
    xlabel('像素 X'); ylabel('像素 Y');
    set(gca, 'YDir', 'reverse');
    axis equal; grid on;
    
    % ===== 子图2: 修正后质心 =====
    subplot(1, 3, 2);
    hold on; box on;
    scatter(corrected_centroids(:,1), corrected_centroids(:,2), 80, 'g', 'filled', 'MarkerFaceAlpha', 0.6);
    
    % 绘制近邻点对深度标注
    for k = 1:size(depth_pairs, 1)
        i = depth_pairs(k, 1);
        j = depth_pairs(k, 2);
        dist_pix = sqrt((corrected_centroids(j,1) - corrected_centroids(i,1))^2 + ...
                        (corrected_centroids(j,2) - corrected_centroids(i,2))^2);
        
        if dist_pix < S_avg_theory * 1.8  % 只显示近邻点对
            p1 = corrected_centroids(i, 1:2);
            p2 = corrected_centroids(j, 1:2);
            
            line([p1(1), p2(1)], [p1(2), p2(2)], 'Color', [0.7 0.7 0.7], 'LineStyle', '--', 'LineWidth', 0.5);
            
            mid_p = (p1 + p2) / 2;
            Z_ij = depth_pairs(k, 3);
            text(mid_p(1), mid_p(2), sprintf('%.3f', Z_ij), ...
                 'FontSize', 6, 'Color', [0.2 0.5 0.2], 'HorizontalAlignment', 'center');
        end
    end
    
    for k = 1:N_spots
        text(corrected_centroids(k,1)+15, corrected_centroids(k,2)+15, ...
            sprintf('(%d,%d)', corrected_centroids(k,3), corrected_centroids(k,4)), ...
            'FontSize', 7, 'Color', 'r');
    end
    
    if enable_correction
        title('修正后质心分布');
    else
        title('质心分布（未修正）');
    end
    xlabel('像素 X'); ylabel('像素 Y');
    set(gca, 'YDir', 'reverse');
    axis equal; grid on;
    
    % ===== 子图3: 深度分布直方图 =====
    subplot(1, 3, 3);
    histogram(depths_list, 30, 'FaceColor', [0.3 0.6 0.9], 'EdgeColor', 'none', 'FaceAlpha', 0.7);
    hold on;
    
    % 绘制统计线
    xline(average_depth, 'r-', 'LineWidth', 2.5, 'Label', sprintf('均值=%.3fm', average_depth));
    xline(average_depth - std_depth, 'r--', 'LineWidth', 1.5, 'Label', '-1σ');
    xline(average_depth + std_depth, 'r--', 'LineWidth', 1.5, 'Label', '+1σ');
    
    title(sprintf('深度分布 (σ/μ = %.2f%%)', 100*std_depth/average_depth));
    xlabel('深度 (m)');
    ylabel('频次');
    grid on;
    
    sgtitle(sprintf('微透镜阵列深度测量结果 | 平均深度: %.4f m ± %.4f m', average_depth, std_depth), ...
            'FontSize', 14, 'FontWeight', 'bold');
    
    fprintf('\n========== 计算完成 ==========\n\n');
end

%% ========== 辅助函数：旋转误差计算 ==========
function residuals = computeRotationError(params, ideal_x, ideal_y, obs_x, obs_y)
    theta = params(1);
    cx = params(2);
    cy = params(3);
    
    % 旋转矩阵
    R = [cos(theta), -sin(theta); 
         sin(theta),  cos(theta)];
    
    N = length(obs_x);
    residuals = zeros(N * 2, 1);
    
    for i = 1:N
        % 理想坐标旋转
        ideal_vec = [ideal_x(i); ideal_y(i)];
        rotated = R * ideal_vec;
        
        % 观测坐标去中心化
        obs_vec = [obs_x(i) - cx; obs_y(i) - cy];
        
        % 残差
        diff = rotated - obs_vec;
        residuals(2*i-1:2*i) = diff;
    end
end