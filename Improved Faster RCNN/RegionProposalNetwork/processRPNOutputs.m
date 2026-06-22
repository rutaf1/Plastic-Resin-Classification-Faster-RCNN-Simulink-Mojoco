function [allProposals, allScores] = processRPNOutputs(rpnOutputs, anchorBoxes, imageSize, varargin)
% processRPNOutputs - Process RPN outputs dan generate final proposals
%
% Input:
%   rpnOutputs  - Cell array {scoresP2, boxDeltasP2, scoresP3, boxDeltasP3,
%                              scoresP4, boxDeltasP4, scoresP5, boxDeltasP5}
%   anchorBoxes - Cell array anchor boxes dari generateAnchorBoxes()
%   imageSize   - [height, width] ukuran gambar asli
%
% Optional:
%   'NMSThreshold'   - IoU threshold NMS (default: 0.7)
%   'ScoreThreshold' - Minimum score (default: 0.05)
%   'MaxProposals'   - Max proposals output (default: 2000)
%
% Output:
%   allProposals - [N x 4] proposals [x, y, width, height]
%   allScores    - [N x 1] scores

    p = inputParser;
    addParameter(p, 'NMSThreshold',   0.7);
    addParameter(p, 'ScoreThreshold', 0.05);
    addParameter(p, 'MaxProposals',   2000);
    parse(p, varargin{:});

    % Hitung featureMapSizes otomatis dari imageSize [H W]
    featureMapSizes = {
        [floor(imageSize(1)/4),  floor(imageSize(2)/4)],  ...  % P2 stride 4
        [floor(imageSize(1)/8),  floor(imageSize(2)/8)],  ...  % P3 stride 8
        [floor(imageSize(1)/16), floor(imageSize(2)/16)], ...  % P4 stride 16
        [floor(imageSize(1)/32), floor(imageSize(2)/32)]       % P5 stride 32
    };
    allProposals = [];
    allScores    = [];

    for level = 1:4
        clsScores  = rpnOutputs{(level-1)*2 + 1};
        bboxDeltas = rpnOutputs{(level-1)*2 + 2};

        [proposals, scores] = generateProposalsPerLevel(...
            clsScores, bboxDeltas, anchorBoxes{level}, ...
            featureMapSizes{level}, imageSize);

        allProposals = [allProposals; proposals];
        allScores    = [allScores;    scores];
    end

    % Clip ke batas gambar SEBELUM NMS (urutan benar: decode -> clip -> NMS),
    % supaya IoU Matrix NMS dihitung pada box yang sudah valid.
    allProposals = clipBoxes(allProposals, imageSize);

    % Matrix NMS lintas semua level (sesuai jurnal)
    [allProposals, allScores] = applyNMS(allProposals, allScores, ...
        'ScoreThreshold', p.Results.ScoreThreshold, ...
        'MaxProposals',   p.Results.MaxProposals,   ...
        'Sigma',          0.5);
end