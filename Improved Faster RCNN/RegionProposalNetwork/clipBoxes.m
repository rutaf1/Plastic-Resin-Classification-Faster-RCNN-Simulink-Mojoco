function clippedBoxes = clipBoxes(boxes, imageSize)
% clipBoxes - Clip bounding boxes agar tidak keluar batas gambar
%
% Input:
%   boxes     - [N x 4] [xmin, ymin, xmax, ymax]
%   imageSize - [height, width]

    if isempty(boxes) || size(boxes,2) < 4
        clippedBoxes = boxes;
        return;
    end

    height = imageSize(1);
    width  = imageSize(2);

    boxes(:,1) = max(0,     min(boxes(:,1), width  - 1));   % xmin
    boxes(:,2) = max(0,     min(boxes(:,2), height - 1));   % ymin
    boxes(:,3) = max(1,     min(boxes(:,3), width));         % xmax
    boxes(:,4) = max(1,     min(boxes(:,4), height));        % ymax

    % Pastikan xmax > xmin dan ymax > ymin
    boxes(:,3) = max(boxes(:,3), boxes(:,1) + 1);
    boxes(:,4) = max(boxes(:,4), boxes(:,2) + 1);

    clippedBoxes = boxes;
end
