classdef AnnotateBboxSystem < matlab.System
% AnnotateBboxSystem - Visualisasi bbox + tripwire (koordinat 480x480)
%
%   Warna bbox per kelas (sebelum tripwire):
%     HDPE (idx 1) - Cyan     [0,   210, 210]
%     LDPE (idx 2) - Kuning   [255, 210,   0]
%     Other(idx 3) - Magenta  [210,   0, 210]
%     PET  (idx 4) - Oranye   [255, 130,   0]
%     PP   (idx 5) - Biru     [30,  144, 255]
%     PS   (idx 6) - Putih    [255, 255, 255]
%     PVC  (idx 7) - Ungu     [148,  60, 255]
%
%   Saat melewati tripwire:
%     HIJAU [0,255,0] = kelas TARGET
%     MERAH [255,0,0] = kelas NON-TARGET
%
%   Input:
%     frameOrig   - uint8  [H x W x 3]
%     bboxes      - double [N x 4]  variable-size
%     scores      - double [N x 1]  variable-size
%     labelIdx    - uint8  [N x 1]  variable-size
%     targetClass - uint8  scalar   (resin code: 0=semua, 1=PET, ..., 7=Other)
%
%   Output:
%     frameAnnotated - uint8 [H x W x 3]
%
% =========================================================================

    properties (Access = private)
        TripwireY
        FrameH
        FrameW
        BboxLineWidth
        LabelFontSize
        TextBoxOpacity
        TripwireWidth
        ClassNames
        ClassColors
    end

    methods (Access = protected)

        function setupImpl(obj)
            obj.TripwireY     = 370;    % = 400*0.75+60, koordinat 480x480
            obj.FrameH        = 480;
            obj.FrameW        = 480;
            obj.BboxLineWidth = 1;
            obj.LabelFontSize = 10;
            obj.TextBoxOpacity = 0.45;
            obj.TripwireWidth = 1;
            obj.ClassNames    = {'HDPE (2)', 'LDPE (4)', 'Other (7)', 'PET (1)', 'PP (5)', 'PS (6)', 'PVC (3)'};
            obj.ClassColors   = [
                  0, 210, 210;   % idx 1 HDPE  - Cyan
                255, 210,   0;   % idx 2 LDPE  - Kuning
                210,   0, 210;   % idx 3 Other - Magenta
                255, 130,   0;   % idx 4 PET   - Oranye
                 30, 144, 255;   % idx 5 PP    - Biru
                255, 255, 255;   % idx 6 PS    - Putih
                148,  60, 255;   % idx 7 PVC   - Ungu
            ];
        end

        function flag = isInputSizeMutableImpl(~, ~)
            flag = true;
        end

        function frameAnnotated = stepImpl(obj, frameOrig, bboxes, scores, labelIdx, targetClass)

            frameAnnotated = uint8(frameOrig);
            nDet           = size(bboxes, 1);

            % Konversi resin code → class index
            % Resin: 0=Semua, 1=PET, 2=HDPE, 3=PVC, 4=LDPE, 5=PP, 6=PS, 7=Other
            RESIN_TO_IDX = [0, 4, 1, 7, 2, 5, 6, 3];
            rc = double(targetClass);
            if rc == 0
                target = 0;
            elseif rc >= 1 && rc <= 7
                target = RESIN_TO_IDX(rc + 1);
            else
                target = 0;
            end

            %% Tentukan status tripwire tiap deteksi (trigger di 1/2 bbox)
            anyTarget       = false;
            anyNonTarget    = false;
            crossesTripwire = false(nDet, 1);
            isTargetDet     = false(nDet, 1);

            for i = 1:nDet
                triggerPoint = bboxes(i, 2) + 0.5 * bboxes(i, 4);
                if triggerPoint >= obj.TripwireY
                    crossesTripwire(i) = true;
                    kelasIni = double(labelIdx(i));
                    if (target == 0) || (kelasIni == target)
                        isTargetDet(i) = true;
                        anyTarget      = true;
                    else
                        anyNonTarget   = true;
                    end
                end
            end

            %% Gambar tripwire
            if anyNonTarget
                lineColor = [255,   0,   0];
            elseif anyTarget
                lineColor = [  0, 255,   0];
            else
                lineColor = [255, 255,   0];
            end
            y1 = max(1, obj.TripwireY - floor(obj.TripwireWidth / 2));
            y2 = min(obj.FrameH, obj.TripwireY + floor(obj.TripwireWidth / 2));
            frameAnnotated(y1:y2, :, 1) = lineColor(1);
            frameAnnotated(y1:y2, :, 2) = lineColor(2);
            frameAnnotated(y1:y2, :, 3) = lineColor(3);

            if nDet == 0, return; end

            %% Tentukan warna final tiap bbox — matrix Nx3 uint8
            colorsMat = zeros(nDet, 3, 'uint8');
            for i = 1:nDet
                if crossesTripwire(i)
                    if isTargetDet(i)
                        colorsMat(i,:) = uint8([0, 255, 0]);
                    else
                        colorsMat(i,:) = uint8([255, 0, 0]);
                    end
                else
                    idx = double(labelIdx(i));
                    if idx >= 1 && idx <= size(obj.ClassColors, 1)
                        colorsMat(i,:) = uint8(obj.ClassColors(idx, :));
                    else
                        colorsMat(i,:) = uint8([200, 200, 200]);
                    end
                end
            end

            %% Gambar bbox
            frameAnnotated = insertShape(frameAnnotated, 'Rectangle', bboxes, ...
                'Color',     colorsMat, ...
                'LineWidth', obj.BboxLineWidth);

            %% Gambar label teks di pojok kiri atas tiap bbox
            labels  = obj.buildLabels(labelIdx, scores);
            textPos = [bboxes(:,1), bboxes(:,2)];
            frameAnnotated = insertText(frameAnnotated, textPos, labels, ...
                'FontSize',   obj.LabelFontSize, ...
                'BoxColor',   colorsMat, ...
                'BoxOpacity', obj.TextBoxOpacity, ...
                'TextColor',  'black');
        end

        function labels = buildLabels(obj, idxArr, scoresArr)
            n = numel(idxArr);
            labels = cell(n, 1);
            for i = 1:n
                idx = double(idxArr(i));
                if idx >= 1 && idx <= numel(obj.ClassNames)
                    cname = obj.ClassNames{idx};
                else
                    cname = 'unknown';
                end
                labels{i} = sprintf('%s %.0f%%', cname, scoresArr(i) * 100);
            end
        end

        function s = getOutputSizeImpl(obj),   s = propagatedInputSize(obj, 1);   end
        function t = getOutputDataTypeImpl(~), t = 'uint8'; end
        function c = isOutputComplexImpl(~),   c = false;   end
        function f = isOutputFixedSizeImpl(~), f = true;    end
        function n = getNumInputsImpl(~),      n = 5;       end
        function n = getNumOutputsImpl(~),     n = 1;       end

    end
end
