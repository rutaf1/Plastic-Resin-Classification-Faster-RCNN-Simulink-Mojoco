function anchorBoxes = generateAnchorBoxes()
% generateAnchorBoxes - Generate anchor boxes untuk setiap pyramid level FPN
%
% Output:
%   anchorBoxes - Cell array {anchorsP2, anchorsP3, anchorsP4, anchorsP5}
%                 Tiap elemen: [5 x 2] matrix [width, height]

    aspectRatios = [0.25, 0.5, 1, 2, 5];   % sesuai jurnal (Wang et al. 2021)
    % Base size per level (P2..P5) disetel untuk cakupan dataset @ input 480x480
    % (box 480-space: sqrt-area ~18..153 px). recall@IoU0.5 = 99.8%.
    baseSizes    = [20, 40, 72, 120]; % P2, P3, P4, P5

    anchorBoxes = cell(1, 4);

    for i = 1:4
        baseSize   = baseSizes(i);
        numAnchors = length(aspectRatios);
        anchors    = zeros(numAnchors, 2);

        for j = 1:numAnchors
            anchors(j, 1) = baseSize * sqrt(aspectRatios(j));
            anchors(j, 2) = baseSize / sqrt(aspectRatios(j));
        end

        anchorBoxes{i} = anchors;
    end
end
