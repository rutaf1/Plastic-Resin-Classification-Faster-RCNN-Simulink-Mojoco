function [proposals, scores] = generateProposalsPerLevel(clsScores, bboxDeltas, anchors, featureMapSize, imageSize)
% generateProposalsPerLevel - Generate proposals untuk satu pyramid level
%
% Input:
%   clsScores      - [H x W x numAnchors*2]
%   bboxDeltas     - [H x W x numAnchors*4]
%   anchors        - [numAnchors x 2] [width, height]
%   featureMapSize - [height, width]
%   imageSize      - [height, width]
%
% Output:
%   proposals - [N x 4] [xmin, ymin, xmax, ymax]
%   scores    - [N x 1]

    fmHeight = featureMapSize(1);
    fmWidth  = featureMapSize(2);
    stride   = [imageSize(1)/fmHeight, imageSize(2)/fmWidth];

    numAnchorsPerLoc = size(clsScores, 3) / 2;
    anchorsUsed      = anchors(1:min(numAnchorsPerLoc, size(anchors,1)), :);
    [anchorGrid, ~]  = generateAnchorGrid(anchorsUsed, featureMapSize, stride);

    %% Reshape [H x W x C] -> [N x C], ANCHOR-MAJOR to match generateAnchorGrid
    %  (all locations of anchor 1, then anchor 2). Channel layout per anchor:
    %  scores [c0 c1], deltas [dx dy dw dh]; locations column-major (row fastest).
    csParts = cell(1, numAnchorsPerLoc); bdParts = cell(1, numAnchorsPerLoc);
    for a = 1:numAnchorsPerLoc
        csParts{a} = reshape(clsScores(:,:,(a-1)*2 + (1:2)), [], 2);   % [H*W,2]
        bdParts{a} = reshape(bboxDeltas(:,:,(a-1)*4 + (1:4)), [], 4);  % [H*W,4]
    end
    clsScores  = vertcat(csParts{:});
    bboxDeltas = vertcat(bdParts{:});

    %% Sesuaikan ukuran
    minN       = min([size(anchorGrid,1), size(clsScores,1), size(bboxDeltas,1)]);
    anchorGrid = anchorGrid(1:minN, :);
    clsScores  = clsScores(1:minN,  :);
    bboxDeltas = bboxDeltas(1:minN, :);

    %% Apply bbox deltas → [xmin ymin xmax ymax]
    scores    = clsScores(:, 2);
    proposals = applyBBoxDeltas(anchorGrid, bboxDeltas);

    %% Filter box tidak valid (xmax > xmin dan ymax > ymin)
    validIdx  = (proposals(:,3) > proposals(:,1)) & ...
                (proposals(:,4) > proposals(:,2));
    proposals = proposals(validIdx, :);
    scores    = scores(validIdx);
end
