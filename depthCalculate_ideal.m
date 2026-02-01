function [labeled_centroids, average_depth, std_depth, final_3D_pos] = depthCalculate_ideal(centroids, H, W)
    
    if ischar(H); H = str2double(H); end
    if ischar(W); W = str2double(W); end
    
    f = 35e-3;            % 焦距 (m)
    p_size = 2.5e-6;      % 像素尺寸 (m)
    dL = 1e-3;            % 微透镜节距 (m)
    S_avg_theory = 400;   
    N_spots = size(centroids, 1);
    
    %% 1. 编号逻辑
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

    labeled_centroids = zeros(N_spots, 6);
    for k = 1:N_spots
        bx = centroids(k, 1); by = centroids(k, 2);
        rid = row_tags(k); cid = col_tags(k);
        ax = S_avg_theory * cid - S_avg_theory/2;
        ay = S_avg_theory * rid - S_avg_theory/2;
        labeled_centroids(k, :) = [bx, by, rid, cid, ax, ay];
    end

    %% 2. 深度计算与可视化增强
    depths_list = [];
    
    % 创建可视化窗口
    figure('Color', 'w', 'Name', '质心编号与点对深度详情');
    hold on; box on;
    
    % 绘制质心
    h_obs = scatter(labeled_centroids(:,1), labeled_centroids(:,2), 50, 'b', 'filled');
    
    for i = 1:(N_spots - 1)
        for j = (i + 1):N_spots
            Idx1 = labeled_centroids(i, 3:4); 
            Idx2 = labeled_centroids(j, 3:4); 
            
            % 计算物理基线 B
            B = sqrt((Idx2(1)-Idx1(1))^2 + (Idx2(2)-Idx1(2))^2) * dL; 
            % 计算像面像素距离
            dist_pix = sqrt((labeled_centroids(j,1)-labeled_centroids(i,1))^2 + ...
                            (labeled_centroids(j,2)-labeled_centroids(i,2))^2);
            D_meas = dist_pix * p_size;
            denominator = D_meas - B;
            
            if abs(denominator) > 1e-10
                Z_ij = (f * B) / denominator;
                if Z_ij > 0 && Z_ij < 50 
                    depths_list = [depths_list; Z_ij];
                    
                    % --- 可视化每一对质心的深度结果 ---
            
                    if dist_pix < S_avg_theory * 1.5
                        p1 = labeled_centroids(i, 1:2);
                        p2 = labeled_centroids(j, 1:2);
                        % 画连线
                        line([p1(1), p2(1)], [p1(2), p2(2)], 'Color', [0.8 0.8 0.8], 'LineStyle', '--');
                        % 连线中点标注该点对算出的深度
                        mid_p = (p1 + p2) / 2;
                        text(mid_p(1), mid_p(2), sprintf('%.3fm', Z_ij), ...
                             'FontSize', 7, 'Color', [0.3 0.6 0.3], 'HorizontalAlignment', 'center');
                    end
                end
            end
        end
    end
    
    % --- 可视化质心编号 ---
    for k = 1:N_spots
        text(labeled_centroids(k,1)+10, labeled_centroids(k,2)+10, ...
            sprintf('(%d,%d)', labeled_centroids(k,3), labeled_centroids(k,4)), ...
            'FontSize', 8, 'Color', 'r', 'FontWeight', 'bold');
    end

    average_depth = nanmean(depths_list);
    std_depth = nanstd(depths_list);
    
    %% 3. 三维空间方位解算
    final_3D_pos = struct('X', NaN, 'Y', NaN, 'Z', average_depth);
    if ~isnan(average_depth)
        R = average_depth / (average_depth + f); 
        X_pos_list = []; Y_pos_list = [];
        for k = 1:N_spots
            xai = labeled_centroids(k, 1) * p_size;
            yai = labeled_centroids(k, 2) * p_size;
            xsi = labeled_centroids(k, 5) * p_size;
            ysi = labeled_centroids(k, 6) * p_size;
            X_pos_list = [X_pos_list; (R * xai - xsi) / (R - 1)];
            Y_pos_list = [Y_pos_list; (R * yai - ysi) / (R - 1)];
        end
        final_3D_pos.X = mean(X_pos_list);
        final_3D_pos.Y = mean(Y_pos_list);
    end
    title(['质心深度测量可视化 (Avg: ', num2str(average_depth, '%.3f'), 'm)']);
    xlabel('像素 X'); ylabel('像素 Y');
    set(gca, 'YDir', 'reverse'); % 图像坐标习惯
    axis image; grid on;
    
    fprintf('计算完成。平均深度: %.4f m, 标准差: %.4f m\n', average_depth, std_depth);
end