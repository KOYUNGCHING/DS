% 逐秒標註（左鍵逐點；Enter 完成；r=沿用；s/Esc=跳過；q=結束）
% 箱子=1，其餘=0
close all; clear; clc;

%% 參數
DATA_FN = 'laser_dump_box.dat';           % 來源 t, angle(rad), range(m)
SAVE_FN = 'labeled_points_box_timesliced.csv';
DT = 1.0;                                 % 每窗秒數
MAX_SHOW = 5e4;                           % 每窗最多顯示點數（僅顯示用，計算仍用全量）

%% 讀檔 + 前處理
assert(isfile(DATA_FN),'找不到 %s',DATA_FN);
T = readmatrix(DATA_FN,'FileType','text','CommentStyle','#');
assert(size(T,2)>=3,'需至少三欄 t, angle, range');

t = T(:,1); ang = T(:,2); r = T(:,3);
ok = isfinite(t)&isfinite(ang)&isfinite(r)&r>0;
t=t(ok); ang=ang(ok); r=r(ok);

[~,ord] = sort(t,'ascend'); t=t(ord); ang=ang(ord); r=r(ord);

% 單精度以降記憶體/提順暢度（計算足夠）
t = single(t); ang = single(ang); r = single(r);
x = r.*cos(ang); y = r.*sin(ang);

% 時間分箱（一次完成，避免每窗 find）
edges = single(floor(min(t))):single(DT):single(ceil(max(t)));
if edges(end) < max(t), edges(end+1) = max(t); end
bin = discretize(t, edges);
nWin = numel(edges) - 1;

idxCell = cell(nWin,1);
for k = 1:nWin, idxCell{k} = find(bin==k); end

% 準備輸出檔
fid = fopen(SAVE_FN,'w'); assert(fid>0);
fprintf(fid,'# columns: t,angle,range,label  (label: 1=box, 0=other)\n');
fclose(fid);

%% 輕量 GUI
fig = figure('Color','w','Name','[箱子標註] 左鍵逐點；Enter 完成；r=沿用；s/Esc=跳過；q=結束', ...
    'MenuBar','none','ToolBar','none','DockControls','off', ...
    'Renderer','opengl','GraphicsSmoothing','off','Position',[100 100 820 620]);
ax  = axes('Parent',fig); hold(ax,'on'); grid(ax,'on'); axis(ax,'equal');
xlabel(ax,'X (m)'); ylabel(ax,'Y (m)');
plot(ax,0,0,'ko','MarkerSize',5,'LineWidth',1,'HitTest','off'); % 原點
set(fig,'WindowKeyPressFcn',@(s,e) setappdata(s,'lastKey',e.Key)); % 擷取快捷鍵

% 點雲用 line（比 scatter 輕），多邊形線
hPts = line(ax,'XData',nan,'YData',nan,'LineStyle','none','Marker','.', ...
    'MarkerSize',6,'Color',[0 0.447 0.741],'HitTest','off');
hPoly = line(ax,'XData',nan,'YData',nan,'Color',[0.2 0.8 0.2], ...
    'LineWidth',1.2,'HitTest','off');

Pprev = []; % 上一窗多邊形（Nx2）

%% main loop
for k = 1:nWin
    if ~ishandle(fig), break; end
    I = idxCell{k}; if isempty(I), continue; end

    % 顯示（裁切顯示點，重繪節流）
    Ishow = I(1:min(numel(I),MAX_SHOW));
    set(hPts,'XData',x(Ishow),'YData',y(Ishow));
    set(hPoly,'XData',nan,'YData',nan);
    title(ax,sprintf('[箱] Window %d/%d | t∈[%.3f, %.3f) | 點數=%d', ...
        k, nWin, edges(k), edges(k+1), numel(I)), 'Interpreter','none');
    setappdata(fig,'lastKey',''); drawnow limitrate;

    % 預設本窗全 0（非箱）
    in = false(numel(I),1);

    % 先讀快捷鍵
    pause(0.01); key = getappdata(fig,'lastKey');
    if strcmp(key,'q'), break; end
    if strcmp(key,'s') || strcmp(key,'escape')
        % 跳過 → 維持全 0
    elseif strcmp(key,'r') && ~isempty(Pprev) && size(Pprev,1)>=3
        % 沿用上一窗（先 bbox 過濾再 inpolygon，加速）
        xmin=min(Pprev(:,1)); xmax=max(Pprev(:,1));
        ymin=min(Pprev(:,2)); ymax=max(Pprev(:,2));
        boxmask = (x(I)>=xmin & x(I)<=xmax & y(I)>=ymin & y(I)<=ymax);
        J = find(boxmask);
        if ~isempty(J)
            inJ = inpolygon(x(I(J)), y(I(J)), Pprev(:,1), Pprev(:,2));
            in(J) = inJ;
        end
        set(hPoly,'XData',[Pprev(:,1);Pprev(1,1)],'YData',[Pprev(:,2);Pprev(1,2)]);
        drawnow limitrate;
    else
        % 互動畫：左鍵逐點、Enter 完成（Esc 跳過）
        xp = []; yp = [];
        while true
            [xi,yi,btn] = ginput(1);
            if isempty(btn) || btn==13, break;       % Enter 完成
            elseif btn==27, xp=[]; yp=[]; break;     % Esc 跳過
            else
                xp(end+1)=xi; yp(end+1)=yi;          
                set(hPoly,'XData',[xp xp(1)],'YData',[yp yp(1)]);
                drawnow limitrate;
            end
        end
        if numel(xp)>=3
            Pprev = [xp(:) yp(:)];
            xmin=min(xp); xmax=max(xp); ymin=min(yp); ymax=max(yp);
            boxmask = (x(I)>=xmin & x(I)<=xmax & y(I)>=ymin & y(I)<=ymax);
            J = find(boxmask);
            if ~isempty(J)
                inJ = inpolygon(x(I(J)), y(I(J)), xp, yp);
                in(J) = inJ;
            end
        end
    end

    % 寫檔：t,angle,range,label（label: 1=box, 0=other）
    out = [double(t(I)) double(ang(I)) double(r(I)) double(in)];
    writematrix(out, SAVE_FN, 'WriteMode','append');
    fprintf(' [%.2f–%.2f) 共 %d 筆已存\n',edges(k),edges(k+1),numel(I));
end

disp(['完成，輸出：', SAVE_FN]);