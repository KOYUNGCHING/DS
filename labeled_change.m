%% label 完的資料處理 「根據標籤好的點」生成特徵向量 ball&box 共用
% 輸入: labeled_points_box_timesliced2.csv  -> line8
% 輸出: labeled_features_multiclass_box.csv  -> line 65,66

clear; clc; close all;

% 讀取資料 
T = readtable("labeled_points_box_timesliced.csv");  % 換資料改這個
T.Properties.VariableNames = {'t','angle','range','label'};

% 依角度不連續切 segments
ANGLE_GAP = 0.05;    % 約 3 度
MIN_POINTS = 4;      % 至少 4 點才計算

% 將連續點群組成 segments 
angle_diff = abs(diff(T.angle));
% 尋找角度跳變點 閾值可視情況微調 (0.05 radians ≈ 3 度)
segment_edges = [0; find(angle_diff > ANGLE_GAP); height(T)];
segments = {};
% 根據角度跳變切割成一群一群的 segment。只保留至少 4 個點的 segment，避免雜訊
for i = 1:length(segment_edges)-1
    idx = (segment_edges(i)+1):segment_edges(i+1);
    if numel(idx) >= MIN_POINTS
        segments{end+1} = T(idx, :);
    end
end

fprintf("共有 %d 個 segments\n", numel(segments));

% 特徵提取
feat_rows = [];              % [n,sigma,width,sc,rc,sl,dtheta,arc_chord_ratio,linearity,flatness]
label_rows = [];
pos_rows   = [];             % [x_c, y_c, d]  (非必要，方便檢視/除錯)

for k = 1:length(segments)
    S = segments{k};
    try
        [f, pos, ~] = extract_features(S);
        lbl = mode(S.label);

        feat_rows = [feat_rows; f];
        label_rows = [label_rows; lbl];
        pos_rows   = [pos_rows; pos];
    catch ME
        fprintf("Segment %d skipped: %s\n", k, ME.message);
    end
end


% 建立表格並輸出 
T_out = array2table([feat_rows, label_rows, pos_rows], ...
    'VariableNames', {'n','sigma','width','sc','rc','sl','dtheta','arc_chord_ratio','linearity','flatness', ...
                      'label', 'x','y','d'});

% n: 點數（物體掃到的點數；與距離/表面反射有關）
% sigma: 平面離散程度（點群擴散度；越大越分散）
% width: X 方向寬度（箱子常較寬；球視角度而定）
% sc: 圓擬合殘差（越小越像圓）
% rc: 擬合圓半徑（僅作輔助；小弧/大半徑時需搭配其他特徵）
% sl: 線擬合殘差（越小越像直線 → 偏箱）
% dtheta: 角度跨度（弧段越大越有助於辨識圓弧）
% arc_chord_ratio: 弧長/弦長比（>1 顯著為弧；≈1 偏直線）
% linearity, flatness: PCA 形狀比（線性度/扁平度）

writetable(T_out, 'labeled_features_multiclass_box.csv'); % 換資料改這個
disp('已生成: labeled_features_multiclass_box.csv'); % 換資料改這個

%% 特徵提取函數 
function [feats, pos, extras] = extract_features(segment)
    ang = segment.angle; 
    rng = segment.range;
    X = [rng.*cos(ang), rng.*sin(ang)];
    n = size(X,1);

    mu = mean(X,1);
    sigma = sqrt(mean(sum((X - mu).^2, 2)));
    width = max(X(:,1)) - min(X(:,1));

    % --- 圓擬合 ---
    [xc, yc, rc] = circle_fit_ls(X);
    ri = sqrt((X(:,1)-xc).^2 + (X(:,2)-yc).^2);
    sc = sum((rc - ri).^2);

    % --- 線擬合 ---
    p = polyfit(X(:,1), X(:,2), 1);       % y = p1*x + p2
    y_hat = polyval(p, X(:,1));
    sl = sum((X(:,2) - y_hat).^2);        % 線擬合殘差

    % --- 角度跨度與弧/弦 ---
    dtheta = max(ang) - min(ang);
    L_arc  = sum( sqrt(sum(diff(X).^2,2)) );
    chord  = sqrt(sum((X(end,:) - X(1,:)).^2));
    arc_chord_ratio = L_arc / max(chord, 1e-6);

    % --- PCA 形狀比 ---
    C = cov(X);
    e = sort(eig(C), 'descend');          % λ1 ≥ λ2
    lambda1 = e(1); lambda2 = e(2);
    linearity = (lambda1 - lambda2) / max(lambda1, 1e-9);
    flatness  = lambda2 / max(lambda1, 1e-9);

    % --- 位置 ---
    x_c = mu(1); y_c = mu(2); d = hypot(x_c, y_c);

    feats  = [n, sigma, width, sc, rc, sl, dtheta, arc_chord_ratio, linearity, flatness];
    pos    = [x_c, y_c, d];
    extras = struct('sl',sl,'dtheta',dtheta,'arc_chord_ratio',arc_chord_ratio, ...
                    'linearity',linearity,'flatness',flatness, ...
                    'xc',xc,'yc',yc);
end

%% === 最小平方擬合圓 ===
function [xc, yc, r] = circle_fit_ls(P)
    x = P(:,1); y = P(:,2);
    A = [-2*x, -2*y, ones(size(x))];
    b = -(x.^2 + y.^2);
    sol = A\b;
    xc = sol(1); yc = sol(2);
    r = sqrt(max(xc^2 + yc^2 - sol(3), 0));
end
