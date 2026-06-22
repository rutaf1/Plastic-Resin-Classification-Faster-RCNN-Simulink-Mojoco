function selectedIdx = customNMS(bboxes, scores, overlapThreshold)
% customNMS - Implementasi Non-Maximum Suppression
%
% Input:
%   bboxes           - [N x 4] [x, y, width, height]
%   scores           - [N x 1] scores
%   overlapThreshold - IoU threshold

    x1 = bboxes(:, 1);
    y1 = bboxes(:, 2);
    x2 = bboxes(:, 1) + bboxes(:, 3);
    y2 = bboxes(:, 2) + bboxes(:, 4);

    areas = (x2 - x1 + 1) .* (y2 - y1 + 1);

    [~, order] = sort(scores, 'descend');

    selectedIdx = [];

    while ~isempty(order)
        idx         = order(1);
        selectedIdx = [selectedIdx; idx];

        if length(order) == 1
            break;
        end

        rest = order(2:end);

        xx1 = max(x1(idx), x1(rest));
        yy1 = max(y1(idx), y1(rest));
        xx2 = min(x2(idx), x2(rest));
        yy2 = min(y2(idx), y2(rest));

        w = max(0, xx2 - xx1 + 1);
        h = max(0, yy2 - yy1 + 1);

        intersection = w .* h;
        iou          = intersection ./ (areas(idx) + areas(rest) - intersection);

        keepIdx = find(iou <= overlapThreshold);
        order   = rest(keepIdx);
    end
end
