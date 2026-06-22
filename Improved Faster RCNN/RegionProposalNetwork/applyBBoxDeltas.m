function decodedBoxes = applyBBoxDeltas(anchors, deltas)
% applyBBoxDeltas - Apply predicted deltas ke anchor boxes
%
% Input:
%   anchors      - [N x 4] [xmin, ymin, xmax, ymax]
%   deltas       - [N x 4] [dx, dy, dw, dh]
%
% Output:
%   decodedBoxes - [N x 4] [xmin, ymin, xmax, ymax]

    anchorW = anchors(:,3) - anchors(:,1) + 1e-6;
    anchorH = anchors(:,4) - anchors(:,2) + 1e-6;
    anchorCX = anchors(:,1) + anchorW/2;
    anchorCY = anchors(:,2) + anchorH/2;

    dx = deltas(:,1);
    dy = deltas(:,2);
    dw = min(deltas(:,3), 4);   % clamp agar exp tidak explode
    dh = min(deltas(:,4), 4);

    predCX = dx .* anchorW  + anchorCX;
    predCY = dy .* anchorH + anchorCY;
    predW  = exp(dw) .* anchorW;
    predH  = exp(dh) .* anchorH;

    decodedBoxes = zeros(size(anchors), 'single');
    decodedBoxes(:,1) = predCX - predW/2;   % xmin
    decodedBoxes(:,2) = predCY - predH/2;   % ymin
    decodedBoxes(:,3) = predCX + predW/2;   % xmax
    decodedBoxes(:,4) = predCY + predH/2;   % ymax
end
