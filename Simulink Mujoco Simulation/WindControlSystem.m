classdef WindControlSystem < matlab.System
% WindControlSystem - Aktivasi solenoid valve (koordinat 480x480)
%
%   Bbox langsung dari DetectObjectsSystem (480x480), tidak perlu rescale.
%
%   Letterbox 640x480 → 480x480:
%     scale  = 0.75
%     padTop = 60px, padLeft = 0px
%
%   TripwireY di 480x480 = 365px
%
%   Trigger: aktif saat TENGAH bbox dalam zona simetris ±50% bboxH dari TripwireY.
%     Zona lebar memastikan valve mulai extend saat objek masih di atas valve,
%     sehingga valve sudah fully extended saat objek tiba di posisinya.
%     aktif = TripwireY - bboxH*0.5 <= tengah <= TripwireY + bboxH*0.5
%
%   Aktivasi: valve yang posisi X-nya overlap dengan titik tengah X bbox.
%
%   Input:
%     bboxes      - double [N x 4]  variable-size  [x y w h] koordinat 480x480
%     labelIdx    - uint8  [N x 1]  variable-size
%     targetClass - uint8  scalar
%     rotaryVel   - double scalar
%
%   Output:
%     u           - double [36x1]
%                   u(1)     = rotaryVel
%                   u(2..36) = valve1..35
%
% =========================================================================

    properties (Nontunable)
        ExtendPos       = 3.5
        RetractPos      = 0.0
        MaxActiveValves = 40    % batas valve aktif sekaligus
        HoldSteps       = 2   % step tambahan valve tetap extend setelah objek keluar zona
        RotaryVelocity  = 0.6  % kecepatan konstan rotary gate (rad/s)
        DebugMode       = false
    end

    properties (Access = private)
        ValveRanges
        TripwireY
        HoldCounters    % 35x1, hitung mundur setelah objek keluar zona
    end

    methods (Access = protected)

        function setupImpl(obj)
            obj.TripwireY    = 370;
            obj.HoldCounters = zeros(35, 1);

            obj.ValveRanges = [
               116, 107;   % valve1
               122, 116;   % valve2
               131, 122;   % valve3
               137, 131;   % valve4
               146, 139;   % valve5
               152, 146;   % valve6
               161, 154;   % valve7
               169, 163;   % valve8
               176, 169;   % valve9
               184, 178;   % valve10
               193, 184;   % valve11
               199, 193;   % valve12
               208, 199;   % valve13
               214, 208;   % valve14
               223, 216;   % valve15
               231, 225;   % valve16
               238, 231;   % valve17
               246, 240;   % valve18
               255, 246;   % valve19
               261, 255;   % valve20
               270, 261;   % valve21
               279, 270;   % valve22
               285, 279;   % valve23
               294, 285;   % valve24
               300, 294;   % valve25
               309, 302;   % valve26
               317, 309;   % valve27
               324, 317;   % valve28
               332, 326;   % valve29
               341, 332;   % valve30
               347, 341;   % valve31
               356, 347;   % valve32
               362, 356;   % valve33
               371, 364;   % valve34
               375, 371;   % valve35
            ];

            fprintf('[WindControl] setupImpl. TripwireY=%d | zona ±50%% | hold=%d steps\n', obj.TripwireY, obj.HoldSteps);
        end

        function u = stepImpl(obj, bboxes, labelIdx, targetClass, rotaryVel)

            u    = zeros(36, 1);
            u(1) = obj.RotaryVelocity;

            nDet = size(bboxes, 1);

            % Konversi resin code → class index
            RESIN_TO_IDX = [0, 4, 1, 7, 2, 5, 6, 3];
            rc = double(targetClass);
            if rc == 0
                target = 0;
            elseif rc >= 1 && rc <= 7
                target = RESIN_TO_IDX(rc + 1);
            else
                target = 0;
            end

            valveTriggered = false(35, 1);

            for i = 1:nDet

                kelasIni    = double(labelIdx(i));
                targetMatch = (target == 0) || (kelasIni == target);
                if ~targetMatch, continue; end

                bboxY = bboxes(i, 2);
                bboxH = bboxes(i, 4);

                % Zona simetris lebar: ±50% bboxH dari TripwireY.
                % Valve mulai extend saat objek masih di atas valve → fully extended
                % saat objek tiba di posisi valve.
                trigger_tengah = bboxY + 0.5 * bboxH;
                band           = 0.5 * bboxH;

                if trigger_tengah < obj.TripwireY - band, continue; end
                if trigger_tengah > obj.TripwireY + band, continue; end

                bboxCenterX = bboxes(i, 1) + 0.5 * bboxes(i, 3);
                for k = 1:35
                    lo = obj.ValveRanges(k, 2);
                    hi = obj.ValveRanges(k, 1);
                    if lo <= bboxCenterX && hi >= bboxCenterX
                        valveTriggered(k) = true;
                        if obj.DebugMode
                            fprintf('[WindControl] Valve %d AKTIF | centerX=%.1f\n', k, bboxCenterX);
                        end
                    end
                end
            end

            %% Hold counter — one-shot per valve: counter hanya dimulai jika sudah 0.
            % Selama counter berjalan (>0), trigger baru diabaikan — tidak bisa di-reset.
            % Counter harus habis dulu baru valve bisa dipicu lagi.
            for k = 1:35
                if valveTriggered(k) && obj.HoldCounters(k) == 0
                    obj.HoldCounters(k) = obj.HoldSteps;   % mulai hanya jika belum berjalan
                end
                if obj.HoldCounters(k) > 0
                    valveTriggered(k)   = true;
                    obj.HoldCounters(k) = obj.HoldCounters(k) - 1;
                end
            end

            %% Batasi valve aktif sekaligus
            if sum(valveTriggered) > obj.MaxActiveValves
                idx  = find(valveTriggered);
                mid  = round(median(idx));
                half = floor(obj.MaxActiveValves / 2);
                keep = max(1, mid-half) : min(35, mid-half+obj.MaxActiveValves-1);
                valveTriggered(:)    = false;
                valveTriggered(keep) = true;
            end

            for k = 1:35
                if valveTriggered(k)
                    u(k + 1) = obj.ExtendPos;
                else
                    u(k + 1) = obj.RetractPos;
                end
            end

        end

        function s = getOutputSizeImpl(~),          s = [36, 1]; end
        function t = getOutputDataTypeImpl(~),      t = 'double'; end
        function c = isOutputComplexImpl(~),        c = false;   end
        function f = isOutputFixedSizeImpl(~),      f = true;    end
        function flag = isInputSizeMutableImpl(~, ~), flag = true; end
        function n = getNumInputsImpl(~),           n = 4; end
        function n = getNumOutputsImpl(~),          n = 1; end

    end
end
