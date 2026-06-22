function [proposals, scores] = applyNMS(bboxes, scores, varargin)
% applyNMS - Region-proposal suppression via MATRIX NMS (sesuai jurnal).
%
% Input:
%   bboxes - [N x 4] [xmin ymin xmax ymax]  (xyxy)
%   scores - [N x 1] objectness scores
%
% Optional:
%   'ScoreThreshold' - ambang skor (default 0.05) untuk filter awal & pasca-decay
%   'MaxProposals'   - maks proposal keluaran (default 2000)
%   'PreNMSTopN'     - top-N sebelum NMS (default 6000)
%   'Sigma'          - lebar kernel Gaussian Matrix NMS (default 0.5)
%   'OverlapThreshold' - diterima untuk kompatibilitas API (tidak dipakai Matrix NMS)

    p = inputParser;
    p.KeepUnmatched = true;   % terima 'OverlapThreshold' lama tanpa error
    addParameter(p, 'ScoreThreshold', 0.05);
    addParameter(p, 'MaxProposals',   2000);
    addParameter(p, 'PreNMSTopN',     6000);
    addParameter(p, 'Sigma',          0.5);
    parse(p, varargin{:});

    scoreThreshold = p.Results.ScoreThreshold;
    maxProposals   = p.Results.MaxProposals;
    preNMSTopN     = p.Results.PreNMSTopN;
    sigma          = p.Results.Sigma;

    % Filter skor awal
    validIdx = scores >= scoreThreshold;
    bboxes   = bboxes(validIdx, :);
    scores   = scores(validIdx);
    if isempty(bboxes)
        proposals = []; scores = []; return;
    end

    % Top-N sebelum NMS (batasi biaya matriks IoU N x N)
    [scores, sortIdx] = sort(scores, 'descend');
    if numel(sortIdx) > preNMSTopN
        sortIdx = sortIdx(1:preNMSTopN);
        scores  = scores(1:preNMSTopN);
    end
    bboxes = bboxes(sortIdx, :);

    % Matrix NMS (box xyxy) -> indeks dipertahankan + skor ter-decay
    [keepIdx, decScores] = matrixNMS(bboxes, scores, sigma, scoreThreshold, maxProposals);
    proposals = bboxes(keepIdx, :);
    scores    = decScores;
end
