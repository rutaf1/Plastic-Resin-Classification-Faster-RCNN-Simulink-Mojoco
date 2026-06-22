function [keepIdx, decScores] = matrixNMS(boxes, scores, sigma, scoreThr, maxKeep)
% matrixNMS - Matrix NMS (SOLOv2, Wang et al. NeurIPS 2020), box-adapted.
%   Sesuai jurnal Wang et al. (Metals 2021) "matrix NMS": parallel, satu pass,
%   tanpa loop greedy. Skor tiap box di-DECAY berdasarkan IoU dengan box
%   ber-skor lebih tinggi (Gaussian), bukan dibuang keras.
%
% Input:
%   boxes     - [N x 4] format [xmin ymin xmax ymax] (xyxy)
%   scores    - [N x 1]
%   sigma     - lebar kernel Gaussian (default 0.5)
%   scoreThr  - ambang skor SETELAH decay untuk dipertahankan (default 0.05)
%   maxKeep   - maksimum box dikembalikan (default 300)
%
% Output:
%   keepIdx   - indeks (ke baris `boxes` masukan) yang dipertahankan, urut skor turun
%   decScores - skor setelah decay untuk box yang dipertahankan

    if nargin < 3 || isempty(sigma),    sigma    = 0.5;  end
    if nargin < 4 || isempty(scoreThr), scoreThr = 0.05; end
    if nargin < 5 || isempty(maxKeep),  maxKeep  = 300;  end

    keepIdx = []; decScores = [];
    if isempty(boxes), return; end

    scores = scores(:);
    [sScores, order] = sort(scores, 'descend');   % urut skor turun
    b = boxes(order, :);
    N = size(b, 1);

    % --- IoU matrix berpasangan (xyxy) ---
    x1 = b(:,1); y1 = b(:,2); x2 = b(:,3); y2 = b(:,4);
    area = max(x2 - x1, 0) .* max(y2 - y1, 0);
    xx1 = max(x1, x1.');  yy1 = max(y1, y1.');
    xx2 = min(x2, x2.');  yy2 = min(y2, y2.');
    inter = max(0, xx2 - xx1) .* max(0, yy2 - yy1);
    iou = inter ./ (area + area.' - inter + eps('single'));

    % --- Matrix NMS decay (Gaussian, SOLOv2) ---
    iou = triu(iou, 1);                       % hanya pengaruh box ber-skor lebih tinggi (i<j)
    iouCmax = max(iou, [], 1);                % [1 x N]: utk tiap kolom j, IoU maks thd box i<j
    iouCmaxMat = repmat(iouCmax.', 1, N);     % [N x N]: elemen (i,j) = iouCmax(i)
    decay = exp(-(iou.^2 - iouCmaxMat.^2) / sigma);   % [N x N]
    decayCoef = min(decay, [], 1).';          % [N x 1]: min atas i untuk tiap j
    decayCoef(1) = 1;                         % box ber-skor tertinggi tak di-decay

    newScores = sScores .* decayCoef;

    % --- Threshold + cap, lalu petakan balik ke indeks asli ---
    keepMask = newScores >= scoreThr;
    keepIdx   = order(keepMask);
    decScores = newScores(keepMask);
    [decScores, o2] = sort(decScores, 'descend');
    keepIdx = keepIdx(o2);
    if numel(keepIdx) > maxKeep
        keepIdx   = keepIdx(1:maxKeep);
        decScores = decScores(1:maxKeep);
    end
end
