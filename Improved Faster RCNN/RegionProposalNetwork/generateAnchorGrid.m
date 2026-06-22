function [anchorGrid, numAnchors] = generateAnchorGrid(anchors, featureMapSize, stride)
% generateAnchorGrid - Generate grid anchor boxes untuk satu feature map
%
% Input:
%   anchors        - [numAnchorsPerLoc x 2] [width, height]
%   featureMapSize - [height, width]
%   stride         - [strideH, strideW]
%
% Output:
%   anchorGrid - [N x 4] [xmin, ymin, xmax, ymax]

    fmHeight         = featureMapSize(1);
    fmWidth          = featureMapSize(2);
    numAnchorsPerLoc = size(anchors, 1);

    [gridX, gridY] = meshgrid(0:fmWidth-1, 0:fmHeight-1);
    centerX = gridX(:) * stride(2) + stride(2)/2;
    centerY = gridY(:) * stride(1) + stride(1)/2;

    numLocations = fmHeight * fmWidth;
    anchorGrid   = zeros(numLocations * numAnchorsPerLoc, 4, 'single');

    for i = 1:numAnchorsPerLoc
        startIdx = (i-1) * numLocations + 1;
        endIdx   =  i    * numLocations;
        w = anchors(i, 1);
        h = anchors(i, 2);
        anchorGrid(startIdx:endIdx, 1) = centerX - w/2;   % xmin
        anchorGrid(startIdx:endIdx, 2) = centerY - h/2;   % ymin
        anchorGrid(startIdx:endIdx, 3) = centerX + w/2;   % xmax
        anchorGrid(startIdx:endIdx, 4) = centerY + h/2;   % ymax
    end

    numAnchors = size(anchorGrid, 1);
end
