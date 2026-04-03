%% 執行sop:
% 機器人：laser_dump.py蒐集資料
% 分別用 label_ball_timeslices.m ＆ label_box_timeslices.m 逐秒標記
% 把標記完的 csv 丟到 label_change
% 再把 labeled_features_multiclass_box.csv ＆ labeled_features_multiclass_ball.csv 丟到這個檔案訓練
% 再把訓練完的數據套到 ros_predictor.py 給機器人

% AdaBoost 訓練 
clear; clc; close all;

% 檔案設定 
box_file  = "labeled_features_multiclass_box.csv";   % 1=箱,0=其他
ball_file = "labeled_features_multiclass_ball.csv";  % 1=球,0=其他

% 特徵定義 
features  = {'n','sigma','width','sc','rc'}; % 5個特徵

% --- AdaBoost 參數 ---
T_adaboost = 50;         % 弱分類器數量
TEST_SPLIT_RATIO = 0.2;  % 分割 20% 資料用於測試評估

%% ------------------- 訓練 BOX 模型 -------------------
fprintf("\n==================================\n");
fprintf("           BOX ADA BOOST TRAINING\n");
fprintf("==================================\n");

[acc_tr_box, acc_te_box, CM_te_box, model_box] = ...
    train_adaboost(box_file, features, T_adaboost, TEST_SPLIT_RATIO, 'box');

fprintf('\n--- BOX AdaBoost 參數 ---\n');
fprintf('ALPHA_BOX:\n'); disp(model_box.alpha');
fprintf('STUMPS_BOX (j, theta, s):\n'); disp([cell2mat({model_box.stump.j})', cell2mat({model_box.stump.theta})', cell2mat({model_box.stump.s})']);
fprintf('-------------------------\n');

%% ------------------- 訓練 BALL 模型 -------------------
fprintf("\n==================================\n");
fprintf("           BALL ADA BOOST TRAINING\n");
fprintf("==================================\n");

[acc_tr_ball, acc_te_ball, CM_te_ball, model_ball] = ...
    train_adaboost(ball_file, features, T_adaboost, TEST_SPLIT_RATIO, 'ball');

fprintf('\n--- BALL AdaBoost 參數 ---\n');
fprintf('ALPHA_BALL:\n'); disp(model_ball.alpha');
fprintf('STUMPS_BALL (j, theta, s):\n'); disp([cell2mat({model_ball.stump.j})', cell2mat({model_ball.stump.theta})', cell2mat({model_ball.stump.s})']);
fprintf('-------------------------\n');

%% 1. AdaBoost 訓練函數 
function [acc_tr, acc_te, CM_te, model] = ...
    train_adaboost(file, features, T, test_ratio, obj_name)
    
    fprintf("\n[ADA] Training %s model (T=%d)\n", obj_name, T);
    
    % 讀資料
    Tbl = readtable(file);
    X_all = table2array(Tbl(:, features));
    y_all = Tbl.label(:);   % 1=物體, 0=其他

    m_all = length(y_all);
    rng(42); % 固定隨機數種子
    
    % 生成一個隨機排列的索引
    idx = randperm(m_all);
    
    % 計算測試集大小
    m_te = round(m_all * test_ratio);
    
    % 分割索引
    idx_te = idx(1:m_te);
    idx_tr = idx(m_te+1:end);
    
    Xtr = X_all(idx_tr,:);
    ytr = y_all(idx_tr);
    Xte = X_all(idx_te,:);
    yte = y_all(idx_te);
    
    % 轉標籤到 ±1 形式
    ytr_pm = 2*ytr - 1;   % 0→−1, 1→+1
    yte_pm = 2*yte - 1;
    [m, d] = size(Xtr);
    
    % 初始化訓練樣本分佈 D_1
    D = ones(m,1) / m;
    
    alpha = zeros(T,1);
    stump = struct('j',[],'theta',[],'s',[]);
    
    % main loop
    for t = 1:T
        best_err = inf;
        best = struct('j',1,'theta',0,'s',1);
        
        for j = 1:d
            xj = Xtr(:,j);
            vals = unique(xj);
            if numel(vals)==1
                thetas = vals + 1e-6;    
            else
                thetas = (vals(1:end-1) + vals(2:end))/2;
            end
            
            for th = thetas.'
                for sgn = [+1, -1]
                    pred = stump_predict_column(xj, th, sgn);
                    err = sum(D .* (pred ~= ytr_pm));
                    if err < best_err
                        best_err = err;
                        best.j = j; best.theta = th; best.s = sgn;
                    end
                end
            end
        end
        
        h_t = stump_predict_column(Xtr(:,best.j), best.theta, best.s);
        
        eps_t = max(min(best_err, 1-1e-9), 1e-9);
        alpha(t) = 0.5 * log( (1 - eps_t) / eps_t );
        
        D = D .* exp( -alpha(t) .* (ytr_pm .* h_t) );
        D = D / sum(D); 
        stump(t) = best;  
        
        if mod(t, 10)==0
            fprintf("  Iter %3d/%d | Err=%.4f | Alpha=%.4f\n", t, T, best_err, alpha(t));
        end
    end
    
    % 強分類器輸出（訓練/測試）: sign(F(x))
    F_tr = zeros(m,1);
    F_te = zeros(size(Xte,1),1);
    for t = 1:T
        F_tr = F_tr + alpha(t)*stump_predict_column(Xtr(:,stump(t).j), stump(t).theta, stump(t).s);
        F_te = F_te + alpha(t)*stump_predict_column(Xte(:,stump(t).j), stump(t).theta, stump(t).s);
    end
    
    % 轉換回 0/1 標籤
    yhat_tr_pm = sign(F_tr); yhat_tr_pm(yhat_tr_pm==0) = 1;
    yhat_te_pm = sign(F_te); yhat_te_pm(yhat_te_pm==0) = 1;
    yhat_tr = (yhat_tr_pm + 1)/2;
    yhat_te = (yhat_te_pm + 1)/2;
    
    acc_tr = mean(yhat_tr == ytr);
    acc_te = mean(yhat_te == yte);
    
    fprintf('\n[ADA] Training accuracy = %.2f%%\n', 100*acc_tr);
    fprintf('[ADA] Testing  accuracy = %.2f%%\n',  100*acc_te);
    
    % Confusion Matrix (測試集)
    TN_te = sum( (yte==0) & (yhat_te==0) );
    FP_te = sum( (yte==0) & (yhat_te==1) );
    FN_te = sum( (yte==1) & (yhat_te==0) );
    TP_te = sum( (yte==1) & (yhat_te==1) );
    CM_te = [TN_te FP_te; FN_te TP_te];
    
    fprintf("--- AdaBoost %s: Testing Confusion Matrix ---\n", obj_name);
    disp(array2table(CM_te, 'VariableNames',{'pred_0','pred_1'}, ...
                              'RowNames',    {'true_0','true_1'}));
    
    % 儲存模型參數
    model.alpha = alpha;   
    model.stump = stump;   
end

%% 決策樹樁預測函數
function pred = stump_predict_column(xcol, theta, s)
    pred = ones(size(xcol));
    pred( s*(xcol - theta) < 0 ) = -1;
end
