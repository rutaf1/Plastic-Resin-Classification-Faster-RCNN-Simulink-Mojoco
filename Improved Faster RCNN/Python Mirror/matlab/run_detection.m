% run_detection.m
% Deteksi cacat plastik di MATLAB menggunakan model ONNX "Improved Faster R-CNN"
% (PET/HDPE/PVC/LDPE/PP/PS/Other) lewat ONNX Runtime (Python), TANPA mengimpor
% graph-nya ke MATLAB.
%
% Kenapa lewat ONNX Runtime, bukan importNetworkFromONNX?
%   Graph berisi GridSample (dari DCN), RoiAlign, dan NonMaxSuppression yang
%   tidak didukung importer ONNX MATLAB (jadi placeholder layer kosong). ONNX
%   Runtime mendukung semua operator itu, jadi kita jalankan model apa adanya.
%
% PRASYARAT (sekali saja):
%   1) Punya file ONNX hasil training: python_alternatif/outputs/model_best.onnx
%      (dibuat otomatis di akhir `python train.py`, atau `python export_onnx.py`).
%   2) Python 3.10-3.12 dengan paket: onnxruntime, numpy, pillow, torch
%      (torch dipakai oleh data/transforms.py untuk letterbox).
%   3) Arahkan MATLAB ke Python tsb (sekali per sesi), lihat PYEXE di bawah.
% ---------------------------------------------------------------------------

% ===== KONFIGURASI (sesuaikan path) =====
PYEXE     = "C:\Users\lkoaj104\AppData\Local\Programs\Python\Python310\python.exe";
HELPERDIR = "D:\Fatur2026\fasterRCNNModified\python_alternatif\matlab";  % berisi onnx_detect.py
ONNXFILE  = "D:\Fatur2026\fasterRCNNModified\python_alternatif\outputs\model_best.onnx";
IMAGEFILE = "D:\Fatur2026\fasterRCNNModified\dataset\images_aug\frame_000081.png";
SCORETHR  = 0.5;   % ambang skor deteksi

% ===== 1) Arahkan MATLAB ke Python (OutOfProcess: aman dari crash native) =====
pe = pyenv;
if pe.Status ~= "Loaded" || pe.Executable ~= PYEXE
    try
        pyenv(Version=PYEXE, ExecutionMode="OutOfProcess");
    catch
        % Python sudah ter-load di sesi ini; restart MATLAB bila ingin ganti exe.
        pyenv(ExecutionMode="OutOfProcess");
    end
end

% ===== 2) Tambahkan folder helper ke path Python =====
if count(py.sys.path, HELPERDIR) == 0
    insert(py.sys.path, int32(0), HELPERDIR);
end

% ===== 3) Jalankan deteksi =====
res = py.onnx_detect.detect(IMAGEFILE, ONNXFILE, SCORETHR);

boxes  = double(res{'boxes'});                 % [N x 4] xyxy, koordinat ASLI gambar
scores = double(res{'scores'});                % [N x 1]
labels = double(res{'labels'});                % [N x 1] id kelas 1..7
namesC = cell(res{'names'});                   % {N} nama kelas (py.str)
names  = cellfun(@char, namesC, UniformOutput=false);

N = numel(scores);
fprintf("%d deteksi (skor >= %.2f)\n", N, SCORETHR);

% ===== 4) Gambar hasil =====
im = imread(IMAGEFILE);
if N > 0
    % xyxy -> [x y w h] untuk insertObjectAnnotation
    bb = [boxes(:,1), boxes(:,2), boxes(:,3)-boxes(:,1), boxes(:,4)-boxes(:,2)];
    lab = strings(N,1);
    for i = 1:N
        lab(i) = sprintf("%s %.2f", names{i}, scores(i));
    end
    im = insertObjectAnnotation(im, "rectangle", bb, cellstr(lab), ...
                                LineWidth=2, Color="yellow", TextColor="black");
end
figure; imshow(im); title(sprintf("Deteksi ONNX (%d objek)", N));
