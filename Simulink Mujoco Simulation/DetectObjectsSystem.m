classdef DetectObjectsSystem < matlab.System
% DetectObjectsSystem - Faster R-CNN detector sebagai MATLAB System block
%
%   ModelType  : 'baseline'    - loads detector dari Baseline.mat
%                'pruned_fp32' - loads detectorRetrained dari Pruned_fp32.mat
%                'pruned_int8' - dequantisasi int8Weights ke struktur Pruned_fp32
%
%   Frame input di-resize ke DetectSize sebelum masuk ke detect(), lalu
%   bbox hasilnya di-scale balik ke koordinat frame asli (misal 224x224).
%
%   Input  : frame    - uint8 [H x W x 3]
%   Output : bboxes   - double [N x 4]   koordinat frame asli
%            scores   - double [N x 1]
%            labelIdx - uint8  [N x 1]   index kelas 1-based
%
% =========================================================================

    properties (Access = private)
        Detector
        ClassNames
    end

    properties (Nontunable)
        ModelType   = 'baseline'   % 'baseline' | 'pruned_fp32' | 'pruned_int8'
        ModelFolder = 'Model'
        Threshold   = 0.5
        MaxDet      = 100
        DetectSize  = [480, 480]   % [H W] ukuran frame saat masuk ke detect()
    end

    methods (Access = protected)

        function setupImpl(obj)
            switch lower(strtrim(obj.ModelType))
                case 'baseline'
                    p = fullfile(obj.ModelFolder, 'Baseline.mat');
                    fprintf('[DetectObjects] Loading baseline: %s\n', p);
                    d = load(p, 'detector');
                    obj.Detector = d.detector;

                case 'pruned_fp32'
                    p = fullfile(obj.ModelFolder, 'Pruned_fp32.mat');
                    fprintf('[DetectObjects] Loading pruned fp32: %s\n', p);
                    d = load(p, 'detectorRetrained');
                    obj.Detector = d.detectorRetrained;

                case 'pruned_int8'
                    fprintf('[DetectObjects] Membangun pruned int8 dari fp32 base...\n');
                    obj.Detector = obj.buildInt8Detector();

                otherwise
                    error('[DetectObjects] ModelType tidak dikenal: "%s". Gunakan baseline / pruned_fp32 / pruned_int8.', obj.ModelType);
            end

            obj.ClassNames = obj.Detector.ClassNames;
            fprintf('[DetectObjects] Model "%s" siap. Classes: %s\n', ...
                obj.ModelType, strjoin(obj.ClassNames, ', '));
        end

        function [bboxes, scores, labelIdx] = stepImpl(obj, frame)
            frame = uint8(frame);

            % Resize ke DetectSize agar memenuhi minimum ukuran model,
            % lalu scale bbox hasilnya balik ke koordinat frame asli.
            [origH, origW, ~] = size(frame);
            dH = obj.DetectSize(1);
            dW = obj.DetectSize(2);
            if origH ~= dH || origW ~= dW
                frameDetect = imresize(frame, [dH, dW]);
                scaleX = origW / dW;
                scaleY = origH / dH;
            else
                frameDetect = frame;
                scaleX = 1;
                scaleY = 1;
            end

            [bboxRaw, scoreRaw, labelRaw] = detect( ...
                obj.Detector, frameDetect, 'Threshold', obj.Threshold);

            % Scale bbox kembali ke koordinat frame asli
            if (scaleX ~= 1 || scaleY ~= 1) && ~isempty(bboxRaw)
                bboxRaw(:, 1) = bboxRaw(:, 1) * scaleX;  % x
                bboxRaw(:, 2) = bboxRaw(:, 2) * scaleY;  % y
                bboxRaw(:, 3) = bboxRaw(:, 3) * scaleX;  % w
                bboxRaw(:, 4) = bboxRaw(:, 4) * scaleY;  % h
            end

            %% Cast semua output ke tipe yang dideklarasikan
            if isempty(bboxRaw)
                bboxes   = zeros(0, 4);
                scores   = zeros(0, 1);
                labelIdx = uint8(zeros(0, 1));
                return;
            end

            % Potong ke MaxDet
            n = min(size(bboxRaw, 1), obj.MaxDet);
            bboxRaw   = bboxRaw(1:n, :);
            scoreRaw  = scoreRaw(1:n);
            labelRaw  = labelRaw(1:n);

            % Cast eksplisit — detect() bisa return single
            bboxes = double(bboxRaw);
            scores = double(scoreRaw);

            % Konversi categorical label → uint8 index
            labelIdx = uint8(zeros(n, 1));
            for i = 1:n
                lname = char(labelRaw(i));
                for c = 1:numel(obj.ClassNames)
                    if strcmp(lname, obj.ClassNames{c})
                        labelIdx(i) = uint8(c);
                        break;
                    end
                end
            end

        end

        function [s1, s2, s3] = getOutputSizeImpl(obj)
            s1 = [obj.MaxDet, 4];
            s2 = [obj.MaxDet, 1];
            s3 = [obj.MaxDet, 1];
        end

        function [t1, t2, t3] = getOutputDataTypeImpl(~)
            t1 = 'double';     % bboxes
            t2 = 'double';     % scores  ← cocok dengan cast di stepImpl
            t3 = 'uint8';      % labelIdx
        end

        function [c1, c2, c3] = isOutputComplexImpl(~)
            c1 = false; c2 = false; c3 = false;
        end

        function [v1, v2, v3] = isOutputFixedSizeImpl(~)
            v1 = false; v2 = false; v3 = false;
        end

        function n = getNumInputsImpl(~),  n = 1; end
        function n = getNumOutputsImpl(~), n = 3; end

    end

    methods (Access = private)

        function det = buildInt8Detector(obj)
            fp32Path = fullfile(obj.ModelFolder, 'Pruned_fp32.mat');
            int8Path = fullfile(obj.ModelFolder, 'Pruned_int8.mat');
            basePath = fullfile(obj.ModelFolder, 'Baseline.mat');

            % Load fp32 sebagai struktur dasar
            fp32 = load(fp32Path, 'detectorRetrained');
            det  = fp32.detectorRetrained;

            % Load data kuantisasi
            q = load(int8Path, 'int8Weights', 'scaleWeights', 'biasData');

            % Ambil network dan learnables-nya
            net = det.Network;
            if ~isa(net, 'dlnetwork')
                net = dlnetwork(net);
            end
            L = net.Learnables;

            % Dequantisasi dan ganti weight layer per layer
            layerNames = fieldnames(q.int8Weights);
            nW = 0; nB = 0;
            for i = 1:numel(layerNames)
                lname = layerNames{i};

                % W_fp32 = scale * double(W_int8)
                W_dq = single(double(q.int8Weights.(lname)) .* double(q.scaleWeights.(lname)));
                idxW = strcmp(L.Layer, lname) & strcmp(L.Parameter, 'Weights');
                if any(idxW)
                    L.Value{idxW} = dlarray(W_dq);
                    nW = nW + 1;
                end

                % Bias sudah fp32, langsung ganti
                if isfield(q.biasData, lname)
                    B    = single(q.biasData.(lname));
                    idxB = strcmp(L.Layer, lname) & strcmp(L.Parameter, 'Bias');
                    if any(idxB)
                        L.Value{idxB} = dlarray(B);
                        nB = nB + 1;
                    end
                end
            end
            net = setLearnables(net, L);
            fprintf('[Int8] Weight diganti: %d Weights, %d Bias dari %d layer.\n', nW, nB, numel(layerNames));

            % Rekonstruksi fasterRCNNObjectDetector dengan network yang sudah diperbarui
            % Anchor boxes diambil dari Baseline.mat (sama untuk semua model)
            base = load(basePath, 'anchorBoxes');
            try
                det = fasterRCNNObjectDetector(net, det.ClassNames, base.anchorBoxes);
                fprintf('[Int8] Detector berhasil direkonstruksi.\n');
            catch ME
                error(['[Int8] Rekonstruksi detector gagal: %s\n' ...
                       'Jalankan script buildInt8DetectorOffline.m sekali dari Command Window,\n' ...
                       'lalu set ModelType ke ''pruned_int8_cached''.'], ME.message);
            end
        end

    end
end