%% =========================================================================
%  QUANTIZATION INT8 - Faster R-CNN ResNet-50 Setelah Pruning
%  =========================================================================
%  Alur:
%    1. Load model pruned (retrainedNet = dlnetwork backbone)
%    2. Buat calibration datastore dari validation set
%    3. dlquantizer → calibrate → quantize
%    4. Evaluasi perbandingan: float32 vs int8
%    5. Simpan ke .mat
%  =========================================================================
%  SYARAT:
%    - Deep Learning Toolbox Model Compression Library (wajib)
%    - File hasil pruning: FasterRCNN_ResNet50_PRUNED_*.mat
%  =========================================================================

clear; clc; close all;

%% ============================================================
%% KONFIGURASI
%% ============================================================

% File hasil pruning (ganti dengan nama file aktual kamu)
prunedModelFile = 'FasterRCNN_ResNet50_PRUNED_*.mat';  % wildcard OK

% Dataset untuk kalibrasi
datasetFolder  = 'D:\Fatur2026\dataset\images_aug_224';
annotationFile = 'D:\Fatur2026\dataset\annotations_aug.xml';

TARGET_SIZE = [224 224];
ORIG_W      = 956;
ORIG_H      = 720;

% Jumlah gambar untuk kalibrasi (lebih banyak = lebih akurat, lebih lambat)
numCalibrationImages = 100;

%% ============================================================
%% 1. Load Model Pruned
%% ============================================================
fprintf('================================================\n');
fprintf(' QUANTIZATION INT8 - Faster R-CNN ResNet-50\n');
fprintf('================================================\n\n');

listing = dir(prunedModelFile);
if isempty(listing)
    error('File pruned model tidak ditemukan: %s', prunedModelFile);
end
[~, idxSort] = sort([listing.datenum], 'descend');
loadFile     = listing(idxSort(1)).name;

fprintf('Memuat model pruned: %s\n', loadFile);
loaded      = load(loadFile);
prunedNet   = loaded.retrainedNet;   % dlnetwork backbone
classNames  = loaded.classNames;
anchorBoxes = loaded.anchorBoxes;

if isstring(classNames),      classNames = cellstr(classNames); end
if iscategorical(classNames), classNames = cellstr(classNames); end

fprintf('Tipe network: %s\n', class(prunedNet));
fprintf('Layer: %d\n\n', numel(prunedNet.Layers));

%% ============================================================
%% 2. Siapkan Calibration Datastore
%% ============================================================
fprintf('Menyiapkan calibration datastore...\n');

% Baca semua file gambar dari folder
imgFiles = [dir(fullfile(datasetFolder,'*.png')); ...
            dir(fullfile(datasetFolder,'*.jpg'))];

if isempty(imgFiles)
    error('Tidak ada gambar di: %s', datasetFolder);
end

% Batasi jumlah gambar kalibrasi
numCal    = min(numCalibrationImages, numel(imgFiles));
calPaths  = fullfile(datasetFolder, {imgFiles(1:numCal).name}');

fprintf('Gambar kalibrasi: %d\n\n', numCal);

% Buat imageDatastore dengan preprocessing
calDS = imageDatastore(calPaths, ...
    'ReadFcn', @(path) preprocessCalibImage(path, TARGET_SIZE));

%% ============================================================
%% 3. Ukuran Model Sebelum Quantization
%% ============================================================
fprintf('Mengukur model SEBELUM quantization...\n');
try
    metricsFloat = estimateNetworkMetrics(prunedNet);
    floatParams  = sum(metricsFloat.NumberOfLearnables);
    floatMemMB   = sum(metricsFloat.("ParameterMemory (MB)"));
catch
    floatParams = 0;
    floatMemMB  = 0;
end
fprintf('Parameter  : %d\n', floatParams);
fprintf('Ukuran     : %.2f MB\n\n', floatMemMB);

%% ============================================================
%% 4. Buat dlquantizer dan Kalibrasi
%% ============================================================
fprintf('================================================\n');
fprintf('KALIBRASI INT8\n');
fprintf('================================================\n\n');

% Buat quantizer dengan ExecutionEnvironment MATLAB
% (tidak perlu hardware khusus, bisa eksplorasi di MATLAB)
quantObj = dlquantizer(prunedNet, 'ExecutionEnvironment', 'MATLAB');

fprintf('Menjalankan kalibrasi dengan %d gambar...\n', numCal);
fprintf('(Proses ini mengumpulkan dynamic range tiap layer)\n\n');

tic;
try
    calResults = calibrate(quantObj, calDS);
    fprintf('Kalibrasi selesai dalam %.1f detik\n\n', toc);

    % Tampilkan ringkasan kalibrasi
    fprintf('Ringkasan kalibrasi (10 layer pertama):\n');
    disp(calResults(1:min(10, height(calResults)), :));
catch ME
    error('Kalibrasi gagal: %s\n\nPastikan Deep Learning Toolbox Model Compression Library terinstall.\nCek: ver(''deeplearning'')', ME.message);
end

%% ============================================================
%% 5. Quantize ke INT8
%% ============================================================
fprintf('================================================\n');
fprintf('QUANTIZE KE INT8\n');
fprintf('================================================\n\n');

tic;
try
    quantizedNet = quantize(quantObj);
    fprintf('Quantization selesai dalam %.1f detik\n\n', toc);
catch ME
    error('Quantization gagal: %s', ME.message);
end

% Inspeksi hasil quantization
fprintf('Detail quantization:\n');
try
    qDetails = quantizationDetails(quantizedNet);
    fprintf('  Layer terkuantisasi: %d\n', height(qDetails));
    disp(qDetails(1:min(5, height(qDetails)), :));
catch
    fprintf('  (quantizationDetails tidak tersedia)\n');
end

%% ============================================================
%% 6. Ukuran Model Setelah Quantization
%% ============================================================
fprintf('\nMengukur model SESUDAH quantization...\n');
try
    metricsInt8 = estimateNetworkMetrics(quantizedNet);
    int8Params  = sum(metricsInt8.NumberOfLearnables);
    int8MemMB   = sum(metricsInt8.("ParameterMemory (MB)"));
catch
    % estimateNetworkMetrics mungkin tidak support quantized network
    % Estimasi manual: int8 = float32/4
    int8Params = floatParams;
    int8MemMB  = floatMemMB / 4;
    fprintf('  (estimasi manual: INT8 = FP32/4)\n');
end

%% ============================================================
%% 7. Validasi Akurasi INT8
%% ============================================================
fprintf('\nValidasi akurasi INT8...\n');

% Buat validation datastore kecil
numValImages = min(50, numel(imgFiles));
valPaths     = fullfile(datasetFolder, {imgFiles(end-numValImages+1:end).name}');
valDS        = imageDatastore(valPaths, ...
    'ReadFcn', @(path) preprocessCalibImage(path, TARGET_SIZE));

% Custom metric: hitung rata-rata output magnitude sebagai proxy
try
    quantOpts = dlquantizationOptions('MetricFcn', ...
        {@(net) computeProxyMetric(net, valDS, TARGET_SIZE)});
    valResults = validate(quantObj, valDS, quantOpts);
    fprintf('Metric float32 : %.4f\n', valResults.MetricResults.Result{1,1});
    fprintf('Metric int8    : %.4f\n', valResults.MetricResults.Result{2,1});
catch ME
    fprintf('Validasi formal dilewati: %s\n', ME.message);
    fprintf('(Validasi manual tetap dilakukan di bawah)\n');
end

% Validasi manual: bandingkan output float vs int8
fprintf('\nValidasi manual (forward pass 5 gambar)...\n');
reset(valDS);
diffList = zeros(1,5);
for ii = 1:5
    if ~hasdata(valDS), break; end
    imgVal = read(valDS);
    if iscell(imgVal), imgVal = imgVal{1}; end
    X = dlarray(single(imgVal), 'SSCB');

    try
        outFloat = predict(prunedNet, X);
        outInt8  = predict(quantizedNet, X);
        if iscell(outFloat), outFloat = outFloat{1}; end
        if iscell(outInt8),  outInt8  = outInt8{1};  end
        diffList(ii) = mean(abs(extractdata(outFloat(:)) - extractdata(outInt8(:))));
    catch
        diffList(ii) = NaN;
    end
end
validDiff = diffList(~isnan(diffList));
if ~isempty(validDiff)
    fprintf('Rata-rata perbedaan output float32 vs int8: %.6f\n', mean(validDiff));
    fprintf('(Nilai kecil = quantization error rendah)\n\n');
end

%% ============================================================
%% 8. Ringkasan Perbandingan
%% ============================================================
fprintf('=====================================================\n');
fprintf('  PERBANDINGAN: FP32 (Pruned) vs INT8 (Quantized)\n');
fprintf('=====================================================\n');
fprintf('%-28s %12s %12s\n', 'Metrik', 'FP32 Pruned', 'INT8');
fprintf('%-28s %12d %12d\n', 'Jumlah Parameter', floatParams, int8Params);
fprintf('%-28s %12.2f %12.2f\n', 'Ukuran (MB)', floatMemMB, int8MemMB);
fprintf('%-28s %12.1f%% %11.1f%%\n', 'Reduksi Ukuran', 0, ...
        max(0, 100*(1 - int8MemMB/max(floatMemMB,1))));
fprintf('=====================================================\n\n');

%% ============================================================
%% 9. Simpan Model INT8
%% ============================================================
saveName = sprintf('FasterRCNN_ResNet50_INT8_%s.mat', datestr(now,'yyyymmdd_HHMMSS'));

save(saveName, ...
    'quantizedNet', ...   % dlnetwork INT8
    'quantObj', ...       % dlquantizer object (berisi calibration data)
    'calResults', ...     % tabel kalibrasi dynamic range
    'classNames', ...     % nama kelas
    'anchorBoxes', ...    % anchor boxes untuk rekonstruksi detector
    'TARGET_SIZE', ...    % ukuran input
    'floatParams', 'floatMemMB', ...
    'int8Params',  'int8MemMB');

fprintf('Model INT8 tersimpan: %s\n\n', saveName);

% Simpan juga quantizer object terpisah (untuk code generation)
quantizerFile = sprintf('dlquantizer_INT8_%s.mat', datestr(now,'yyyymmdd_HHMMSS'));
save(quantizerFile, 'quantObj', 'calResults');
fprintf('Quantizer object tersimpan: %s\n', quantizerFile);
fprintf('(File ini dibutuhkan untuk code generation ke CPU/GPU)\n\n');

%% ============================================================
%% 10. Visualisasi
%% ============================================================
figure('Name','Quantization Summary','Position',[100 100 900 450]);
tl = tiledlayout(1, 3, 'TileSpacing','compact','Padding','normal');
sgtitle(tl, 'Ringkasan Quantization INT8 - Backbone ResNet-50', ...
        'FontSize', 13, 'FontWeight', 'bold');

% Ukuran memori
ax1 = nexttile(tl);
vals = [floatMemMB, int8MemMB];
bh   = bar(ax1, vals, 0.5);
bh.FaceColor = 'flat';
bh.CData     = [0.2 0.6 0.9; 0.9 0.4 0.1];
bh.FaceAlpha = 0.85;
xticklabels(ax1, {'FP32 Pruned','INT8'});
ylabel(ax1, 'Ukuran (MB)');
title(ax1, 'Ukuran Model', 'FontWeight','bold');
grid(ax1, 'on');
for kk = 1:2
    text(ax1, kk, vals(kk)*1.03, sprintf('%.1f MB', vals(kk)), ...
         'HorizontalAlignment','center','FontWeight','bold','FontSize',10);
end

% Parameter
ax2 = nexttile(tl);
pvals = [floatParams/1e6, int8Params/1e6];
bh2   = bar(ax2, pvals, 0.5);
bh2.FaceColor = 'flat';
bh2.CData     = [0.2 0.6 0.9; 0.9 0.4 0.1];
bh2.FaceAlpha = 0.85;
xticklabels(ax2, {'FP32 Pruned','INT8'});
ylabel(ax2, 'Parameter (M)');
title(ax2, 'Jumlah Parameter', 'FontWeight','bold');
grid(ax2, 'on');

% Pie reduksi memori
ax3  = nexttile(tl);
redM = max(0, 100*(1 - int8MemMB/max(floatMemMB,1)));
p    = pie(ax3, [100-redM, redM]);
p(1).FaceColor = [0.2 0.6 0.9];
p(3).FaceColor = [0.9 0.4 0.1];
legend(ax3, {sprintf('Tersisa %.1f%%',100-redM), sprintf('Tereduksi %.1f%%',redM)}, ...
       'Location','southoutside','FontSize',9);
title(ax3, 'Reduksi Ukuran', 'FontWeight','bold');

saveas(gcf, 'quantization_summary.png');
fprintf('Gambar tersimpan: quantization_summary.png\n');

fprintf('\n=================================\n');
fprintf(' QUANTIZATION INT8 SELESAI!\n');
fprintf('=================================\n');
fprintf(' Model INT8   : %s\n', saveName);
fprintf(' Ukuran FP32  : %.2f MB\n', floatMemMB);
fprintf(' Ukuran INT8  : %.2f MB\n', int8MemMB);
fprintf(' Reduksi      : %.1f%%\n', max(0, 100*(1-int8MemMB/max(floatMemMB,1))));
fprintf('=================================\n\n');


%% ============================================================
%% LOCAL FUNCTIONS
%% ============================================================

function imgOut = preprocessCalibImage(imgPath, targetSize)
    % Baca dan preprocess gambar untuk kalibrasi/validasi
    img = imread(imgPath);
    if size(img,3) == 1, img = repmat(img,[1 1 3]); end
    if size(img,3) == 4, img = img(:,:,1:3); end
    img    = imresize(img, targetSize);
    imgOut = single(img) / 255;
end

function metric = computeProxyMetric(net, ds, targetSize)
    % Custom metric function untuk dlquantizationOptions
    % Menghitung mean output magnitude sebagai proxy akurasi
    totalMag = 0;
    count    = 0;
    reset(ds);
    while hasdata(ds)
        img = read(ds);
        if iscell(img), img = img{1}; end
        X   = dlarray(single(img), 'SSCB');
        try
            out = predict(net, X);
            if iscell(out), out = out{1}; end
            totalMag = totalMag + mean(abs(extractdata(out(:))));
            count    = count + 1;
        catch
        end
    end
    if count == 0
        metric = 0;
    else
        metric = totalMag / count;
    end
end
