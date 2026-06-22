%% =========================================================================
%  PRUNING Faster R-CNN ResNet-50 - Taylor Pruning (FIXED - SIZE PERMANEN)
%  =========================================================================
%  PERBAIKAN UTAMA:
%    1. Sinkronisasi dlnetwork -> layerGraph menggunakan NumFilters eksplisit
%    2. injectAndFixGraph dibangun ulang: tidak hanya replaceLayer, tapi
%       juga rebuild conv layer dengan NumFilters yang benar
%    3. fixDownstreamConvs memperbaiki RPN/head secara rekursif
%    4. Verifikasi parameter count sebelum & sesudah pruning
%    5. BatchNormalization channel disesuaikan otomatis
%
%  FITUR BARU - SAMPLE DETECTION VISUALIZATION:
%    Section 10: Deteksi sampel 3-in-1 (Baseline | Pruned | Retrained)
%       - Memilih N_SAMPLE_IMAGES gambar acak dari validation set
%       - Menjalankan detect() pada ketiga model secara berurutan
%       - Menampilkan figure grid dengan bounding box & confidence score
%       - Menyimpan figure ke PNG otomatis
%    Section 11: Tabel ringkasan perbandingan kuantitatif
%       - AP, jumlah parameter, compression ratio, inference time
% =========================================================================
clear; clc; close all;

%% ============================================================
%% KONFIGURASI
%% ============================================================
modelFile      = 'D:\Fatur2026\FasterRCNN_ResNet50_v2_20260615_141454.mat';
datasetFolder  = 'D:\Fatur2026\dataset\images_aug';
annotationFile = 'D:\Fatur2026\dataset\annotations_aug.xml';
resizedFolder  = 'D:\Fatur2026\dataset\images_aug_480';

TARGET_SIZE    = [480 480];
ORIG_W         = 956;
ORIG_H         = 720;

% --- Konfigurasi Sample Detection Visualization ---
N_SAMPLE_IMAGES  = 6;      % Jumlah gambar sampel yang ditampilkan (kelipatan 3 disarankan)
DETECT_THRESHOLD = 0.3;    % Confidence threshold untuk deteksi
DETECT_SAVE_FIG  = true;   % true = simpan figure ke file PNG otomatis

maxPruningIterations = 5;
maxToPrune           = 32;
numMinibatchUpdates  = 30;
learnRatePrune       = 1e-4;
momentum             = 0.9;
l2Reg                = 5e-4;
miniBatchSizePrune   = 4;

numRetrainEpochs     = 1;
learnRateRetrain     = 1e-4;
learnRateDropEpoch   = 8;
learnRateDropFactor  = 0.1;
miniBatchSizeTrain   = 3;

%% ============================================================
%% 1. LOAD MODEL & DATASET ANOTASI
%% ============================================================
%  ARSITEKTUR DATASET (imageDatastore-based):
%  ┌─────────────────────────────────────────────────────────┐
%  │  XML Anotasi  ──►  readAnnotationsAug()                 │
%  │       │                                                 │
%  │       ▼                                                 │
%  │  Resize + Letterbox Padding (offline, simpan ke disk)   │
%  │       │                                                 │
%  │       ▼                                                 │
%  │  imageFilenames{}  +  bboxesList{}  +  labelsList{}     │
%  │       │                                                 │
%  │       ▼                                                 │
%  │  gTruth (table: imageFilename | kelas1 | kelas2 | ...)  │
%  │       │                                                 │
%  │       ▼  split 80/20                                    │
%  │  trainingData (table)   validationData (table)          │
%  │       │                       │                         │
%  │       ▼                       ▼                         │
%  │  imageDatastore(train)   imageDatastore(val)            │
%  │  + boxLabelDatastore     + boxLabelDatastore            │
%  │       │                       │                         │
%  │       ▼                       ▼                         │
%  │  minibatchqueue          evaluateDetectorAP()           │
%  │  (pruning loop)          (pakai imageDatastore)         │
%  └─────────────────────────────────────────────────────────┘
%
%  CATATAN PENTING:
%    - imageDatastore dipakai di: pruning loop, evaluasi AP,
%      retraining, sample detection visualization
%    - Table (gTruth/trainingData/validationData) hanya untuk
%      boxLabelDatastore dan trainFasterRCNNObjectDetector
%    - Semua pemanggilan detect() dan evaluateDetectionPrecision()
%      menggunakan imageDatastore, BUKAN cell array filename
%% ============================================================
fprintf('============================================\n');
fprintf('MEMUAT MODEL & PRE-PROCESSING DATASET\n');
fprintf('============================================\n');
loaded     = load(modelFile);
detector   = loaded.detector;
classNames = cellstr(loaded.classNames);
numClasses = numel(classNames);

if ~exist(resizedFolder, 'dir')
    mkdir(resizedFolder);
end

% ------------------------------------------------------------------
% TAHAP 1A: Baca anotasi dari XML
% ------------------------------------------------------------------
fprintf('[1/4] Membaca anotasi XML...\n');
frameAnnotations = readAnnotationsAug(annotationFile, ORIG_W, ORIG_H);
fprintf('      Total frame dalam anotasi: %d\n', length(frameAnnotations));

% ------------------------------------------------------------------
% TAHAP 1B: Resize + Letterbox Padding (simpan ke disk)
%           Kumpulkan imageFilenames, bboxesList, labelsList
% ------------------------------------------------------------------
fprintf('[2/4] Resize + Letterbox Padding (offline ke %dx%d)...\n', ...
    TARGET_SIZE(1), TARGET_SIZE(2));

imageFilenames = {}; bboxesList = {}; labelsList = {}; validCount = 0;
skipCount = 0;

for fIdx = 1:length(frameAnnotations)
    % Lewati frame tanpa anotasi atau filename
    if isempty(frameAnnotations(fIdx).bboxes),   skipCount = skipCount+1; continue; end
    if ~isfield(frameAnnotations, 'filename') || ...
            isempty(frameAnnotations(fIdx).filename), skipCount = skipCount+1; continue; end

    fname   = frameAnnotations(fIdx).filename;
    imgPath = fullfile(datasetFolder, fname);
    if ~isfile(imgPath), skipCount = skipCount+1; continue; end

    [~, fn, ext] = fileparts(imgPath);
    resizedPath  = fullfile(resizedFolder, [fn ext]);
    rawBboxes    = frameAnnotations(fIdx).bboxes;
    rawLabels    = frameAnnotations(fIdx).labels;

    try
        if ~isfile(resizedPath)
            % Gambar belum di-resize: baca, resize, simpan
            img = imread(imgPath);
            if size(img,3) == 1, img = repmat(img, [1 1 3]); end
            if size(img,3) == 4, img = img(:,:,1:3); end
            [imgResized, bboxScaled] = resizeWithPadding( ...
                img, rawBboxes, TARGET_SIZE, ORIG_W, ORIG_H);
            imwrite(imgResized, resizedPath);
        else
            % Gambar sudah ada di disk: hitung ulang bbox saja (tanpa baca gambar)
            [~, bboxScaled] = resizeWithPadding( ...
                zeros(ORIG_H, ORIG_W, 3, 'uint8'), rawBboxes, TARGET_SIZE, ORIG_W, ORIG_H);
        end
    catch ME_resize
        warning('[Skip] Gagal resize %s: %s', fname, ME_resize.message);
        skipCount = skipCount + 1;
        continue;
    end

    if isempty(bboxScaled), skipCount = skipCount+1; continue; end

    % Pastikan jumlah label konsisten dengan jumlah bbox valid
    nValid = size(bboxScaled, 1);
    rawLabelsValid = rawLabels(1:min(numel(rawLabels), nValid));
    if numel(rawLabelsValid) < nValid
        % Pad label jika kurang (pakai label pertama sebagai default)
        while numel(rawLabelsValid) < nValid
            rawLabelsValid{end+1} = rawLabels{1}; %#ok<AGROW>
        end
    end

    validCount = validCount + 1;
    imageFilenames{validCount} = resizedPath;          % path absolut gambar 224x224
    bboxesList{validCount}     = bboxScaled;            % [x y w h] sudah discale
    labelsList{validCount}     = rawLabelsValid(1:nValid)';
end

fprintf('      Gambar valid (224x224): %d  |  Dilewati: %d\n', validCount, skipCount);
if validCount == 0
    error('Tidak ada gambar valid. Periksa path dataset dan anotasi XML.');
end

% ------------------------------------------------------------------
% TAHAP 1C: Bangun Ground Truth Table
%           Format: imageFilename | kelas1 | kelas2 | ...
%           (kolom bbox per kelas, cell of [N x 4] double)
% ------------------------------------------------------------------
fprintf('[3/4] Membangun Ground Truth Table...\n');

% Inisialisasi kolom bbox per kelas (cell N x numClasses berisi [0x4 double])
colData = cell(validCount, numClasses);
for i = 1:validCount
    for c = 1:numClasses
        colData{i,c} = zeros(0, 4);
    end
end

% Isi colData dengan bbox yang sesuai kelas
for i = 1:validCount
    bb  = bboxesList{i};
    lbl = labelsList{i};
    for k = 1:min(size(bb,1), numel(lbl))
        cIdx = find(strcmp(classNames, char(lbl{k})), 1);
        if ~isempty(cIdx)
            colData{i,cIdx} = [colData{i,cIdx}; double(bb(k,:))];
        end
    end
end

% Susun tabel (imageFilename + satu kolom per kelas)
gTruth = table(imageFilenames', 'VariableNames', {'imageFilename'});
for c = 1:numClasses
    gTruth.(classNames{c}) = colData(:, c);
end

% ------------------------------------------------------------------
% TAHAP 1D: Split Train/Val (80:20) dan bangun imageDatastore
%           imageDatastore adalah SATU-SATUNYA interface untuk
%           detect() dan evaluateDetectionPrecision()
% ------------------------------------------------------------------
fprintf('[4/4] Split Train/Val dan membangun imageDatastore...\n');

rng(42);
shuffleIdx = randperm(height(gTruth));
numTrain   = round(0.8 * height(gTruth));

% Index train dan val
trainIdx = shuffleIdx(1:numTrain);
valIdx   = shuffleIdx(numTrain+1:end);

% Table (untuk boxLabelDatastore & trainFasterRCNNObjectDetector)
trainingData   = gTruth(trainIdx, :);
validationData = gTruth(valIdx,   :);

% ---- imageDatastore TRAINING ----
%   Dipakai oleh: minibatchqueue (pruning loop)
imdsTrain_full = imageDatastore(trainingData.imageFilename);
imdsTrain_full.ReadFcn = @readAndNormalize224;   % normalisasi di sini

% ---- imageDatastore VALIDASI ----
%   Dipakai oleh: evaluateDetectorAP, sample visualization
%   PENTING: ReadFcn default (imread) agar detect() dapat gambar uint8
imdsVal = imageDatastore(validationData.imageFilename);
% (ReadFcn dibiarkan default = imread, karena detect() butuh uint8 asli)

% ---- boxLabelDatastore (hanya untuk trainFasterRCNNObjectDetector) ----
bldsVal   = boxLabelDatastore(validationData(:, 2:end));
bldsTrain = boxLabelDatastore(trainingData(:,  2:end));

fprintf('Training : %d gambar  |  Validasi: %d gambar\n\n', ...
    height(trainingData), height(validationData));

%% ============================================================
%% 2. EVALUASI AP BASELINE
%% ============================================================
fprintf('============================================\n');
fprintf('EVALUASI AP BASELINE\n');
fprintf('============================================\n');
% Panggil evaluateDetectorAP dengan:
%   arg1 = detector (fasterRCNNObjectDetector)
%   arg2 = imdsVal  (imageDatastore - untuk detect())
%   arg3 = bldsVal  (boxLabelDatastore - untuk evaluateDetectionPrecision())
apBefore = evaluateDetectorAP(detector, imdsVal, bldsVal);
fprintf('AP Baseline: %.2f%%\n\n', apBefore*100);

%% ============================================================
%% 3. EKSTRAK BACKBONE MURNI (DAG TRACING)
%% ============================================================
fprintf('============================================\n');
fprintf('EKSTRAK BACKBONE\n');
fprintf('============================================\n');
lgraphFull = layerGraph(detector.Network);
conns      = lgraphFull.Connections;

rpnConnIdx = find(contains(conns.Destination, 'rpnConv3x3'), 1);
if ~isempty(rpnConnIdx)
    featureLayerName = split(conns.Source{rpnConnIdx}, '/');
    featureLayerName = featureLayerName{1};
else
    featureLayerName = 'activation_40_relu';
end

layersToKeep = {featureLayerName}; queue = {featureLayerName};
while ~isempty(queue)
    curr = queue{1}; queue(1) = [];
    idxConn = find(strcmp(conns.Destination, curr) | contains(conns.Destination, [curr '/']));
    for k = 1:numel(idxConn)
        srcName = split(conns.Source{idxConn(k)}, '/'); srcName = srcName{1};
        if ~ismember(srcName, layersToKeep)
            layersToKeep{end+1} = srcName; %#ok<SAGROW>
            queue{end+1}        = srcName; %#ok<SAGROW>
        end
    end
end

lgraphBackbone = removeLayers(lgraphFull, setdiff({lgraphFull.Layers.Name}, layersToKeep));
backboneNet    = dlnetwork(lgraphBackbone);
prunableNet    = taylorPrunableNetwork(backboneNet);
maxPrunableFilters = prunableNet.NumPrunables;
fprintf('Backbone Prunable Filters: %d\n\n', maxPrunableFilters);

%% ============================================================
%% 4. PRUNING LOOP
%% ============================================================
fprintf('============================================\n');
fprintf('MULAI PRUNING TAYLOR\n');
fprintf('============================================\n');

% ------------------------------------------------------------------
% Siapkan minibatchqueue untuk pruning loop.
%   - imdsTrain_full : imageDatastore dengan ReadFcn normalisasi
%                      (sudah dibuat di Section 1)
%   - bldsTrain      : boxLabelDatastore (sudah dibuat di Section 1)
%
%   KENAPA PAKAI imageDatastore (bukan cell/table langsung)?
%   => minibatchqueue mensyaratkan datastore sebagai input;
%      imageDatastore memastikan gambar dibaca + dinormalisasi
%      secara konsisten tanpa error saat iterasi.
%
%   KENAPA ReadFcn @readAndNormalize224?
%   => Gambar sudah 224x224 di disk (hasil resize Section 1),
%      ReadFcn hanya konversi ke single [0,1] agar siap masuk
%      ke backbone dlnetwork tanpa transform tambahan.
% ------------------------------------------------------------------
mbqTrain = minibatchqueue( ...
    combine(imdsTrain_full, bldsTrain), 2, ...
    'MiniBatchSize',     miniBatchSizePrune, ...
    'MiniBatchFcn',      @prepBatchFast, ...
    'MiniBatchFormat',   ["SSCB", ""], ...
    'OutputEnvironment', "auto");

globalIter = 0;
for pruneIter = 1:maxPruningIterations
    shuffle(mbqTrain); velocity = []; localIter = 0;
    while hasdata(mbqTrain) && localIter < numMinibatchUpdates
        globalIter = globalIter + 1; localIter = localIter + 1;
        [X, ~] = next(mbqTrain);
        [lossVal, pruneGrads, netGrads, pruneActs, newState] = dlfeval(@modelLossTaylor, prunableNet, X);
        prunableNet.State = newState;
        netGrads    = dlupdate(@(g,w) g + l2Reg*w, netGrads, prunableNet.Learnables);
        [prunableNet, velocity] = sgdmupdate(prunableNet, netGrads, velocity, learnRatePrune, momentum);
        prunableNet = updateScore(prunableNet, pruneActs, pruneGrads);
    end
    prunableNet = updatePrunables(prunableNet, 'MaxToPrune', maxToPrune);
    pctPruned   = 100 * (maxPrunableFilters - prunableNet.NumPrunables) / maxPrunableFilters;
    fprintf('  [Iter %3d] Filter: %d (%.1f%% dipruning)\n', pruneIter, prunableNet.NumPrunables, pctPruned);
end
fprintf('\nPruning selesai!\n\n');

%% ============================================================
%% 5. VERIFIKASI PARAMETER SEBELUM MUTASI GRAPH
%%    (Memastikan pruning benar-benar mengubah arsitektur)
%% ============================================================
fprintf('============================================\n');
fprintf('VERIFIKASI PARAMETER COUNT\n');
fprintf('============================================\n');
paramsBefore = countParameters(lgraphFull);
fprintf('Parameter model ASLI    : %d\n', paramsBefore);

prunedBackboneNet = dlnetwork(prunableNet);
paramsBackbonePruned = countParameters(layerGraph(prunedBackboneNet));
paramsBackboneOrig   = countParameters(lgraphBackbone);
fprintf('Parameter backbone asli : %d\n', paramsBackboneOrig);
fprintf('Parameter backbone pruned: %d (%.1f%% berkurang)\n', ...
    paramsBackbonePruned, 100*(1 - paramsBackbonePruned/paramsBackboneOrig));

if paramsBackbonePruned >= paramsBackboneOrig
    warning('PRUNING TIDAK EFEKTIF! Cek konfigurasi maxToPrune dan maxPruningIterations.');
end

%% ============================================================
%% 6. SURGICAL GRAPH MUTATION (FIXED)
%% ============================================================
fprintf('============================================\n');
fprintf('MUTASI LAYER GRAPH (FIXED)\n');
fprintf('============================================\n');
lgraphMutated = injectAndFixGraph(detector.Network, prunableNet, featureLayerName);

paramsAfter = countParameters(lgraphMutated);
fprintf('Parameter model SETELAH pruning: %d (%.1f%% berkurang)\n\n', ...
    paramsAfter, 100*(1 - paramsAfter/paramsBefore));

if paramsAfter >= paramsBefore
    error(['injectAndFixGraph gagal mengurangi ukuran model. ' ...
           'Periksa dimensi layer backbone dan downstream conv layers.']);
end

%% ============================================================
%% 7. RETRAINING (END-TO-END)
%% ============================================================
fprintf('============================================\n');
fprintf('RETRAINING AKHIR (MEMAKAI GPU)\n');
fprintf('============================================\n');
retrainOptions = trainingOptions('sgdm', ...
    'MaxEpochs',           numRetrainEpochs, ...
    'MiniBatchSize',       miniBatchSizeTrain, ...
    'InitialLearnRate',    learnRateRetrain, ...
    'LearnRateSchedule',   'piecewise', ...
    'LearnRateDropFactor', learnRateDropFactor, ...
    'LearnRateDropPeriod', learnRateDropEpoch, ...
    'Momentum',            momentum, ...
    'L2Regularization',    l2Reg, ...
    'Shuffle',             'every-epoch', ...
    'Verbose',             true, ...
    'Plots',               'training-progress', ...
    'ExecutionEnvironment','auto');

[detectorRetrained, retrainInfo] = trainFasterRCNNObjectDetector( ...
    trainingData, lgraphMutated, retrainOptions);

%% ============================================================
%% 8. VERIFIKASI SIZE AKHIR SETELAH RETRAINING
%% ============================================================
fprintf('============================================\n');
fprintf('VERIFIKASI SIZE MODEL AKHIR\n');
fprintf('============================================\n');
lgraphFinal   = layerGraph(detectorRetrained.Network);
paramsFinal   = countParameters(lgraphFinal);
fprintf('Parameter model ASLI    : %d\n', paramsBefore);
fprintf('Parameter sesudah prune : %d\n', paramsAfter);
fprintf('Parameter setelah retrain: %d\n', paramsFinal);
if paramsFinal <= paramsAfter * 1.01
    fprintf('✓ Size model TETAP KECIL setelah retraining (selisih < 1%%)\n\n');
else
    warning('Size model bertambah setelah retraining. Kemungkinan ada layer yang di-reinisialisasi.');
end

%% ============================================================
%% 9. EVALUASI AP AKHIR & SIMPAN
%% ============================================================
fprintf('============================================\n');
fprintf('EVALUASI AP AKHIR\n');
fprintf('============================================\n');
% Gunakan imdsVal + bldsVal (imageDatastore) agar konsisten dengan baseline
% imdsVal  : imageDatastore path gambar 224x224 (ReadFcn default = imread)
% bldsVal  : boxLabelDatastore ground-truth bbox per kelas
apAfter = evaluateDetectorAP(detectorRetrained, imdsVal, bldsVal);
fprintf('AP Baseline  : %.2f%%\n', apBefore*100);
fprintf('AP Setelah   : %.2f%%\n\n', apAfter*100);

saveName = sprintf('FasterRCNN_ResNet50_PRUNED_FIXED_%s.mat', datestr(now,'yyyymmdd_HHMMSS'));
save(saveName, 'detectorRetrained', 'classNames', 'TARGET_SIZE', ...
    'apBefore', 'apAfter', 'retrainInfo', 'paramsBefore', 'paramsFinal');
fprintf('Selesai! Model disimpan: %s\n', saveName);

%% ============================================================
%% 10. SAMPLE DETECTION - PERBANDINGAN VISUAL 3 MODEL
%%     Baseline | After Pruning (sebelum retrain) | After Retrain
%% ============================================================
fprintf('============================================\n');
fprintf('SAMPLE DETECTION VISUALIZATION\n');
fprintf('============================================\n');

% ---- Buat detector sementara dari lgraphMutated untuk model pruned-only ----
% Kita pakai detectorRetrained untuk "after retrain" dan detector asli untuk baseline.
% Untuk "pruned before retrain", kita buat detector dari lgraphMutated.
fprintf('Membangun detector pruned (sebelum retrain) untuk visualisasi...\n');
try
    % Latih 1 epoch saja dengan lr sangat kecil agar bisa dipakai detect()
    optVisualize = trainingOptions('sgdm', ...
        'MaxEpochs',           1, ...
        'MiniBatchSize',       miniBatchSizeTrain, ...
        'InitialLearnRate',    1e-7, ...
        'Shuffle',             'never', ...
        'Verbose',             false, ...
        'Plots',               'none', ...
        'ExecutionEnvironment','auto');
    detectorPrunedOnly = trainFasterRCNNObjectDetector( ...
        trainingData(1:min(10,height(trainingData)),:), lgraphMutated, optVisualize);
    hasPrunedDetector = true;
catch ME
    warning('Tidak bisa membuat detectorPrunedOnly: %s\nMemakai detectorRetrained sebagai pengganti.', ME.message);
    detectorPrunedOnly = detectorRetrained;
    hasPrunedDetector  = false;
end

% ---- Pilih gambar sampel dari validation set secara acak ----
%
%   CATATAN imageDatastore untuk visualisasi:
%   - sampleData (table) dipakai untuk: iterasi baris, akses imageFilename,
%     dan akses ground-truth bbox per kelas (kolom classNames{c})
%   - detect() di bawah memakai imread(imgPath) langsung karena hanya
%     satu gambar per iterasi — lebih simpel daripada membuat imageDatastore
%     baru per-gambar. Gambar yang dibaca adalah uint8 224x224 dari disk.
%
rng(7);
nSample   = min(N_SAMPLE_IMAGES, height(validationData));
sampleIdx = randperm(height(validationData), nSample);
sampleData = validationData(sampleIdx, :);

% ---- Warna per kelas (palette kontras) ----
palette = lines(numClasses);  % MATLAB built-in: warna berbeda per kelas
classColorMap = containers.Map(classNames, num2cell(palette, 2));

% ---- Label kolom ----
if hasPrunedDetector
    colTitles = {'Baseline (Original)', 'After Pruning (before retrain)', 'After Retrain'};
    detectors = {detector, detectorPrunedOnly, detectorRetrained};
else
    colTitles = {'Baseline (Original)', 'After Pruning + Retrain', ''};
    detectors = {detector, detectorRetrained, []};
end

% ---- Buat figure grid: nSample baris x 3 kolom ----
figW = 1400; figH = 320 * nSample;
hFig = figure('Name', 'Sample Detection Comparison', ...
    'Position', [50, 50, figW, figH], ...
    'Color', [0.12 0.12 0.12]);

nCols = 3;
for r = 1:nSample
    imgPath = sampleData.imageFilename{r};
    img     = imread(imgPath);
    if size(img,3)==1, img = repmat(img,[1 1 3]); end

    % Ambil ground truth bboxes untuk gambar ini (semua kelas)
    gtBoxes  = [];
    gtLabels = {};
    for c = 1:numClasses
        bb = sampleData.(classNames{c}){r};
        if ~isempty(bb)
            gtBoxes  = [gtBoxes;  bb]; %#ok<AGROW>
            gtLabels = [gtLabels; repmat(classNames(c), size(bb,1), 1)]; %#ok<AGROW>
        end
    end

    for col = 1:nCols
        spIdx = (r-1)*nCols + col;
        ax = subplot(nSample, nCols, spIdx);
        imshow(img, 'Parent', ax);
        hold(ax, 'on');

        titleStr = colTitles{col};

        if col == 1
            % ---- Kolom 1: Baseline detection ----
            try
                [bboxes, scores, labels] = detect(detectors{col}, img, ...
                    'Threshold', DETECT_THRESHOLD, 'ExecutionEnvironment', 'auto');
                drawDetections(ax, bboxes, scores, labels, classColorMap, classNames, palette);
                nDet = size(bboxes,1);
            catch
                nDet = 0;
            end
            titleStr = sprintf('\\color{cyan}%s\n\\color{white}Deteksi: %d | GT: %d', ...
                colTitles{col}, nDet, size(gtBoxes,1));

        elseif col == 2 && ~isempty(detectors{col})
            % ---- Kolom 2: After pruning (sebelum retrain) ----
            try
                [bboxes, scores, labels] = detect(detectors{col}, img, ...
                    'Threshold', DETECT_THRESHOLD, 'ExecutionEnvironment', 'auto');
                drawDetections(ax, bboxes, scores, labels, classColorMap, classNames, palette);
                nDet = size(bboxes,1);
            catch
                nDet = 0;
            end
            color = '\color{yellow}';
            if ~hasPrunedDetector, color = '\color{green}'; end
            titleStr = sprintf('%s%s\n\\color{white}Deteksi: %d | GT: %d', ...
                color, colTitles{col}, nDet, size(gtBoxes,1));

        elseif col == 3 && ~isempty(detectors{col})
            % ---- Kolom 3: After retrain ----
            try
                [bboxes, scores, labels] = detect(detectors{col}, img, ...
                    'Threshold', DETECT_THRESHOLD, 'ExecutionEnvironment', 'auto');
                drawDetections(ax, bboxes, scores, labels, classColorMap, classNames, palette);
                nDet = size(bboxes,1);
            catch
                nDet = 0;
            end
            titleStr = sprintf('\\color{lime}After Retrain\n\\color{white}Deteksi: %d | GT: %d', ...
                nDet, size(gtBoxes,1));

        else
            % Kolom kosong (jika hasPrunedDetector=false)
            text(0.5, 0.5, 'N/A', 'Units','normalized', 'HorizontalAlignment','center', ...
                'Color','white', 'FontSize',14);
        end

        % Gambar ground truth (garis putus-putus putih)
        if ~isempty(gtBoxes)
            for g = 1:size(gtBoxes,1)
                bb = gtBoxes(g,:);
                rectangle(ax, 'Position', bb, 'EdgeColor', [1 1 1], ...
                    'LineStyle','--', 'LineWidth', 1.2);
            end
        end

        % Nomor gambar di pojok kiri bawah
        [~, fn, ext] = fileparts(imgPath);
        text(ax, 5, size(img,1)-5, [fn ext], 'Color',[0.8 0.8 0.8], ...
            'FontSize', 7, 'Interpreter','none', 'VerticalAlignment','bottom');

        title(ax, titleStr, 'FontSize', 8, 'Interpreter','tex', 'Color','white');
        set(ax, 'Color',[0.08 0.08 0.08], 'XColor','none', 'YColor','none');
        hold(ax, 'off');
    end
end

% Tambahkan legenda kelas di bawah figure
annotation(hFig, 'textbox', [0.0, 0.0, 1.0, 0.025], ...
    'String', buildLegendString(classNames, palette), ...
    'Interpreter','tex', 'FontSize',8, 'Color','white', ...
    'BackgroundColor',[0.12 0.12 0.12], 'EdgeColor','none', ...
    'HorizontalAlignment','center', 'VerticalAlignment','middle');

% Tambahkan judul utama
annotation(hFig, 'textbox', [0.0, 0.975, 1.0, 0.025], ...
    'String', 'Detection Comparison — Baseline vs Pruned vs Retrained  (putus-putus putih = Ground Truth)', ...
    'Interpreter','none', 'FontSize',10, 'FontWeight','bold', ...
    'Color','white', 'BackgroundColor',[0.12 0.12 0.12], ...
    'EdgeColor','none', 'HorizontalAlignment','center', 'VerticalAlignment','middle');

% Simpan figure
if DETECT_SAVE_FIG
    figName = sprintf('SampleDetection_Comparison_%s.png', datestr(now,'yyyymmdd_HHMMSS'));
    exportgraphics(hFig, figName, 'Resolution', 150);
    fprintf('Figure sampel deteksi disimpan: %s\n', figName);
end

%% ============================================================
%% 11. TABEL RINGKASAN PERBANDINGAN KUANTITATIF
%% ============================================================
fprintf('\n============================================\n');
fprintf('TABEL RINGKASAN PERBANDINGAN\n');
fprintf('============================================\n');

% Ukur inference time pada gambar pertama sampel
testImg = imread(sampleData.imageFilename{1});
if size(testImg,3)==1, testImg = repmat(testImg,[1 1 3]); end

NWARMUP = 3; NREPEAT = 10;

% Baseline
for k = 1:NWARMUP
    detect(detector, testImg, 'Threshold', DETECT_THRESHOLD, 'ExecutionEnvironment','auto');
end
tStart = tic;
for k = 1:NREPEAT
    detect(detector, testImg, 'Threshold', DETECT_THRESHOLD, 'ExecutionEnvironment','auto');
end
timeBaseline = toc(tStart)/NREPEAT * 1000;  % ms

% After retrain
for k = 1:NWARMUP
    detect(detectorRetrained, testImg, 'Threshold', DETECT_THRESHOLD, 'ExecutionEnvironment','auto');
end
tStart = tic;
for k = 1:NREPEAT
    detect(detectorRetrained, testImg, 'Threshold', DETECT_THRESHOLD, 'ExecutionEnvironment','auto');
end
timeRetrain = toc(tStart)/NREPEAT * 1000;  % ms

compressionRatio = paramsBefore / paramsFinal;
speedupRatio     = timeBaseline / timeRetrain;

fprintf('\n%-30s  %15s  %15s  %15s\n', 'Metrik', 'Baseline', 'Pruned+Retrain', 'Perubahan');
fprintf('%s\n', repmat('-',1,80));
fprintf('%-30s  %15d  %15d  %14.1f%%\n', 'Jumlah Parameter', ...
    paramsBefore, paramsFinal, -100*(1-paramsFinal/paramsBefore));
fprintf('%-30s  %15.2f  %15.2f  %14.1f%%\n', 'AP (%)', ...
    apBefore*100, apAfter*100, (apAfter-apBefore)/apBefore*100);
fprintf('%-30s  %13.2fms  %13.2fms  %14.1fx\n', 'Inference Time (per img)', ...
    timeBaseline, timeRetrain, speedupRatio);
fprintf('%-30s  %15s  %15.2f  %s\n', 'Compression Ratio', '1.00x', compressionRatio, ...
    sprintf('%.2fx lebih kecil', compressionRatio));
fprintf('%s\n', repmat('-',1,80));
fprintf('\n');

% Tampilkan bar chart ringkasan
hFigSummary = figure('Name','Pruning Summary', 'Position',[100 100 900 380], ...
    'Color',[0.12 0.12 0.12]);

subplot(1,3,1);
barData = [paramsBefore; paramsFinal] / 1e6;
hb = bar(barData, 0.5, 'FaceColor','flat');
hb.CData = [0.3 0.6 0.9; 0.2 0.85 0.5];
set(gca, 'XTickLabel', {'Baseline','Pruned+Retrain'}, 'Color',[0.15 0.15 0.15], ...
    'XColor','white','YColor','white','GridColor',[0.3 0.3 0.3],'YGrid','on');
ylabel('Juta Parameter','Color','white');
title('Ukuran Model','Color','white','FontSize',9);
text(1, barData(1)*0.5, sprintf('%.1fM',barData(1)), 'HorizontalAlignment','center', ...
    'Color','white','FontWeight','bold');
text(2, barData(2)*0.5, sprintf('%.1fM',barData(2)), 'HorizontalAlignment','center', ...
    'Color','white','FontWeight','bold');

subplot(1,3,2);
apData = [apBefore*100; apAfter*100];
hb2 = bar(apData, 0.5, 'FaceColor','flat');
hb2.CData = [0.3 0.6 0.9; 0.2 0.85 0.5];
set(gca, 'XTickLabel', {'Baseline','Pruned+Retrain'}, 'Color',[0.15 0.15 0.15], ...
    'XColor','white','YColor','white','GridColor',[0.3 0.3 0.3],'YGrid','on', ...
    'YLim',[0 100]);
ylabel('AP (%)','Color','white');
title('Average Precision','Color','white','FontSize',9);
text(1, apData(1)/2, sprintf('%.1f%%',apData(1)), 'HorizontalAlignment','center', ...
    'Color','white','FontWeight','bold');
text(2, apData(2)/2, sprintf('%.1f%%',apData(2)), 'HorizontalAlignment','center', ...
    'Color','white','FontWeight','bold');

subplot(1,3,3);
timeData = [timeBaseline; timeRetrain];
hb3 = bar(timeData, 0.5, 'FaceColor','flat');
hb3.CData = [0.3 0.6 0.9; 0.2 0.85 0.5];
set(gca, 'XTickLabel', {'Baseline','Pruned+Retrain'}, 'Color',[0.15 0.15 0.15], ...
    'XColor','white','YColor','white','GridColor',[0.3 0.3 0.3],'YGrid','on');
ylabel('ms / gambar','Color','white');
title('Inference Time','Color','white','FontSize',9);
text(1, timeData(1)/2, sprintf('%.1fms',timeData(1)), 'HorizontalAlignment','center', ...
    'Color','white','FontWeight','bold');
text(2, timeData(2)/2, sprintf('%.1fms',timeData(2)), 'HorizontalAlignment','center', ...
    'Color','white','FontWeight','bold');

sgtitle(sprintf('Pruning Summary  |  Compression: %.2fx  |  Speedup: %.2fx', ...
    compressionRatio, speedupRatio), 'Color','white', 'FontSize',11);

if DETECT_SAVE_FIG
    summaryName = sprintf('PruningSummary_%s.png', datestr(now,'yyyymmdd_HHMMSS'));
    exportgraphics(hFigSummary, summaryName, 'Resolution', 150);
    fprintf('Figure summary disimpan: %s\n', summaryName);
end

fprintf('\n=== PIPELINE SELESAI ===\n');

%% ============================================================
%% FUNGSI LOKAL
%% ============================================================

% ------------------------------------------------------------------
% drawDetections: gambar bounding box + label confidence di atas axes
% ------------------------------------------------------------------
function drawDetections(ax, bboxes, scores, labels, colorMap, classNames, palette)
    if isempty(bboxes), return; end
    labels = cellstr(labels);
    for d = 1:size(bboxes,1)
        bb    = bboxes(d,:);
        score = scores(d);
        lbl   = labels{d};

        % Cari warna untuk kelas ini
        cIdx = find(strcmp(classNames, lbl), 1);
        if isempty(cIdx), clr = [1 0.4 0]; else, clr = palette(cIdx,:); end

        % Kotak deteksi
        rectangle(ax, 'Position', bb, 'EdgeColor', clr, 'LineWidth', 2);

        % Label background
        txtStr = sprintf('%s %.0f%%', lbl, score*100);
        txtPos = [bb(1), max(1, bb(2)-16)];
        text(ax, txtPos(1)+2, txtPos(2)+14, txtStr, ...
            'Color', 'white', ...
            'FontSize', 7, ...
            'FontWeight', 'bold', ...
            'BackgroundColor', [clr, 0.75], ...
            'Margin', 1, ...
            'Interpreter', 'none', ...
            'VerticalAlignment', 'bottom');
    end
end

% ------------------------------------------------------------------
% buildLegendString: buat string legenda kelas untuk annotation
% ------------------------------------------------------------------
function s = buildLegendString(classNames, palette)
    parts = {'Kelas: '};
    for c = 1:numel(classNames)
        clr  = palette(c,:);
        hex  = sprintf('#%02X%02X%02X', round(clr(1)*255), round(clr(2)*255), round(clr(3)*255));
        parts{end+1} = sprintf('\\color[rgb]{%.2f,%.2f,%.2f}■ %s  ', clr(1), clr(2), clr(3), classNames{c}); %#ok<AGROW>
    end
    parts{end+1} = '\color{white}  |  \color[rgb]{1,1,1}-- -- Ground Truth';
    s = strjoin(parts, '');
end

% ------------------------------------------------------------------
% prepBatchFast: normalisasi ringan, tanpa imresize (gambar sudah 224)
% ------------------------------------------------------------------
function [X, T] = prepBatchFast(images, ~)
    imgs = cellfun(@(im) single(im)/255, images, 'UniformOutput', false);
    X    = dlarray(cat(4, imgs{:}), 'SSCB');
    T    = zeros(1,1,1,numel(images));
end

% ------------------------------------------------------------------
% modelLossTaylor: loss untuk Taylor importance scoring
% ------------------------------------------------------------------
function [loss, pruneGrads, netGrads, pruneActs, state] = modelLossTaylor(prunableNet, X)
    [YPred, state, pruneActs] = forward(prunableNet, X);
    if iscell(YPred), YPred = YPred{1}; end
    YNorm = (YPred - mean(YPred(:))) / (std(YPred(:)) + 1e-8);
    loss  = -mean(YNorm(:).^2) + 1e-6 * mean(YPred(:).^2);
    [netGrads, pruneGrads] = dlgradient(loss, prunableNet.Learnables, pruneActs);
end

% ------------------------------------------------------------------
% countParameters: hitung total bobot yang dapat dilatih
% ------------------------------------------------------------------
function n = countParameters(lgraph)
    n = 0;
    for i = 1:numel(lgraph.Layers)
        lyr = lgraph.Layers(i);
        if isprop(lyr, 'Weights') && ~isempty(lyr.Weights)
            n = n + numel(lyr.Weights);
        end
        if isprop(lyr, 'Bias') && ~isempty(lyr.Bias)
            n = n + numel(lyr.Bias);
        end
    end
end

% ------------------------------------------------------------------
% getOutputChannels: dapatkan jumlah channel output suatu layer
%   dengan tracing ke upstream jika perlu
% ------------------------------------------------------------------
function outCh = getOutputChannels(net, layerName)
    outCh = [];
    visited = {};
    queue   = {layerName};
    while ~isempty(queue)
        curr = queue{1}; queue(1) = [];
        if ismember(curr, visited), continue; end
        visited{end+1} = curr; %#ok<AGROW>

        idx = find(strcmp({net.Layers.Name}, curr), 1);
        if isempty(idx), continue; end
        lyr = net.Layers(idx);

        if isa(lyr, 'nnet.cnn.layer.Convolution2DLayer') && ~isempty(lyr.Weights)
            outCh = size(lyr.Weights, 4);
            return;
        elseif isa(lyr, 'nnet.cnn.layer.BatchNormalizationLayer')
            if ~isempty(lyr.TrainedMean)
                outCh = numel(lyr.TrainedMean);
                return;
            end
        elseif isa(lyr, 'nnet.cnn.layer.GroupedConvolution2DLayer') && ~isempty(lyr.Weights)
            outCh = size(lyr.Weights, 4) * lyr.NumGroups;
            return;
        end

        % Telusuri ke upstream
        if isprop(net, 'Connections')
            srcRows = net.Connections(strcmp(net.Connections.Destination, curr), :);
        else
            srcRows = table();
        end
        for k = 1:height(srcRows)
            srcName = split(srcRows.Source{k}, '/'); srcName = srcName{1};
            queue{end+1} = srcName; %#ok<AGROW>
        end
    end
end

% ------------------------------------------------------------------
% rebuildConvLayer: buat ulang Convolution2DLayer dengan NumFilters baru
%   sambil mempertahankan semua properti aslinya
% ------------------------------------------------------------------
function newLyr = rebuildConvLayer(origLyr, newNumFilters, newInputCh)
    fs   = origLyr.FilterSize;
    pad  = origLyr.PaddingSize;
    str  = origLyr.Stride;
    dil  = origLyr.DilationFactor;
    nm   = origLyr.Name;
    hasBias = ~isempty(origLyr.Bias);

    newLyr = convolution2dLayer(fs, newNumFilters, ...
        'Padding',          pad, ...
        'Stride',           str, ...
        'DilationFactor',   dil, ...
        'Name',             nm, ...
        'WeightsInitializer','glorot', ...
        'BiasInitializer',   'zeros');

    % Salin bobot jika dimensi cocok; jika tidak, reinisialisasi (akan dilatih ulang)
    if nargin >= 3 && ~isempty(origLyr.Weights)
        origW  = origLyr.Weights;
        origB  = origLyr.Bias;
        nF_old = size(origW, 4);
        nC_old = size(origW, 3);
        nF_new = newNumFilters;
        nC_new = newInputCh;

        % Jumlah filter yang bisa disalin
        nF_copy = min(nF_old, nF_new);
        nC_copy = min(nC_old, nC_new);

        newW = zeros(fs(1), fs(2), nC_new, nF_new, 'single');
        newW(:,:,1:nC_copy,1:nF_copy) = origW(:,:,1:nC_copy,1:nF_copy);

        if hasBias
            newB = zeros(1,1,nF_new,'single');
            newB(1,1,1:nF_copy) = origB(1,1,1:nF_copy);
            newLyr.Bias = newB;
        end
        newLyr.Weights = newW;
    end
end

% ------------------------------------------------------------------
% rebuildBNLayer: buat ulang BatchNormalizationLayer dengan NumChannels baru
% ------------------------------------------------------------------
function newLyr = rebuildBNLayer(origLyr, newNumCh)
    newLyr = batchNormalizationLayer('Name', origLyr.Name);

    nCopy = min(origLyr.NumChannels, newNumCh);

    % Salin parameter yang ada
    if ~isempty(origLyr.TrainedMean)
        newMean    = zeros(1,1,newNumCh,'single');
        newVar     = ones(1,1,newNumCh,'single');
        newMean(1,1,1:nCopy) = origLyr.TrainedMean(1,1,1:nCopy);
        newVar(1,1,1:nCopy)  = origLyr.TrainedVariance(1,1,1:nCopy);
        newLyr.TrainedMean     = newMean;
        newLyr.TrainedVariance = newVar;
    end
    if ~isempty(origLyr.Offset)
        newOff  = zeros(1,1,newNumCh,'single');
        newSc   = ones(1,1,newNumCh,'single');
        newOff(1,1,1:nCopy) = origLyr.Offset(1,1,1:nCopy);
        newSc(1,1,1:nCopy)  = origLyr.Scale(1,1,1:nCopy);
        newLyr.Offset = newOff;
        newLyr.Scale  = newSc;
    end
end

% ------------------------------------------------------------------
% fixDownstreamConvs: perbaiki layer conv & BN setelah featureLayer
%   secara rekursif mengikuti aliran graph (RPN, ROI head, dsb.)
% ------------------------------------------------------------------
function lgraphOut = fixDownstreamConvs(lgraphOut, startName, inputCh)
    conns   = lgraphOut.Connections;
    queue   = {startName};
    visited = {};

    while ~isempty(queue)
        curr = queue{1}; queue(1) = [];
        if ismember(curr, visited), continue; end
        visited{end+1} = curr; %#ok<AGROW>

        % Cari semua layer yang menerima output dari curr
        srcMask  = strcmp(conns.Source, curr) | startsWith(conns.Source, [curr '/']);
        childRows = conns(srcMask, :);

        for i = 1:height(childRows)
            dstRaw  = childRows.Destination{i};
            dstName = split(dstRaw, '/'); dstName = dstName{1};
            if ismember(dstName, visited), continue; end

            idx = find(strcmp({lgraphOut.Layers.Name}, dstName), 1);
            if isempty(idx), continue; end
            lyr = lgraphOut.Layers(idx);

            if isa(lyr, 'nnet.cnn.layer.Convolution2DLayer')
                currInputCh = size(lyr.Weights, 3);
                if isempty(lyr.Weights), currInputCh = inputCh; end

                if currInputCh ~= inputCh
                    fprintf('    Fix conv: %s  (%d -> %d input ch, %d filters)\n', ...
                        dstName, currInputCh, inputCh, lyr.NumFilters);
                    newLyr   = rebuildConvLayer(lyr, lyr.NumFilters, inputCh);
                    lgraphOut = replaceLayer(lgraphOut, dstName, newLyr);
                end
                % Conv layer mengubah channel; jangan teruskan inputCh
                % (output channel-nya berbeda dari inputCh)
                % Teruskan traversal dengan output channel layer ini
                nextCh = lyr.NumFilters;
                queue{end+1} = dstName; %#ok<AGROW>
                % Tandai dengan inputCh baru untuk layer berikutnya
                % (dicatat melalui rekursi terpisah)
                lgraphOut = fixDownstreamConvs(lgraphOut, dstName, nextCh);
                visited{end+1} = dstName; %#ok<AGROW>

            elseif isa(lyr, 'nnet.cnn.layer.BatchNormalizationLayer')
                currCh = lyr.NumChannels;
                if ~isempty(lyr.TrainedMean)
                    currCh = numel(lyr.TrainedMean);
                end
                if currCh ~= inputCh
                    fprintf('    Fix BN  : %s  (%d -> %d ch)\n', dstName, currCh, inputCh);
                    newLyr    = rebuildBNLayer(lyr, inputCh);
                    lgraphOut = replaceLayer(lgraphOut, dstName, newLyr);
                end
                queue{end+1} = dstName; %#ok<AGROW>

            elseif isa(lyr, 'nnet.cnn.layer.FullyConnectedLayer')
                % FC layer di head — jangan ubah, biarkan MATLAB handle
                % dimensi saat retraining (akan di-reinisialisasi oleh trainer)
                continue;

            else
                % Layer lain (ReLU, pooling, dll.) — teruskan traversal
                queue{end+1} = dstName; %#ok<AGROW>
            end
        end
    end
end

% ------------------------------------------------------------------
% injectAndFixGraph: VERSI FIXED
%   - Rebuild conv layer dengan NumFilters eksplisit (bukan sekadar replaceLayer)
%   - Perbaiki BN yang bergantung pada conv yang dipruning
%   - Perbaiki downstream layer (RPN, head) secara rekursif
% ------------------------------------------------------------------
function lgraphOut = injectAndFixGraph(origNetwork, prunableNet, featureLayerName)
    lgraphOut = layerGraph(origNetwork);

    % Dapatkan backbone yang sudah dipruning sebagai dlnetwork bersih
    prunedBackboneNet = dlnetwork(prunableNet);
    prunedLayers      = prunedBackboneNet.Layers;

    fprintf('  Menyuntikkan %d layer backbone yang dipruning...\n', numel(prunedLayers));

    for i = 1:numel(prunedLayers)
        lyr      = prunedLayers(i);
        lyrName  = lyr.Name;

        % Lewati layer yang tidak ada di graph penuh
        if ~any(strcmp({lgraphOut.Layers.Name}, lyrName))
            continue;
        end

        origIdx = find(strcmp({lgraphOut.Layers.Name}, lyrName), 1);
        origLyr = lgraphOut.Layers(origIdx);

        if isa(lyr, 'nnet.cnn.layer.Convolution2DLayer')
            newNumFilters = size(lyr.Weights, 4);
            newInputCh    = size(lyr.Weights, 3);
            origNumFilters = origLyr.NumFilters;

            if newNumFilters ~= origNumFilters
                fprintf('    Rebuild conv: %s  filters %d -> %d\n', ...
                    lyrName, origNumFilters, newNumFilters);
                newLyr = rebuildConvLayer(origLyr, newNumFilters, newInputCh);
                % Salin bobot pruned langsung (dimensi sudah pasti cocok)
                newLyr.Weights = lyr.Weights;
                if ~isempty(lyr.Bias), newLyr.Bias = lyr.Bias; end
            else
                % Dimensi sama, cukup copy bobot
                newLyr = lyr;
            end
            lgraphOut = replaceLayer(lgraphOut, lyrName, newLyr);

        elseif isa(lyr, 'nnet.cnn.layer.BatchNormalizationLayer')
            newNumCh  = [];
            if ~isempty(lyr.TrainedMean)
                newNumCh = numel(lyr.TrainedMean);
            end
            if ~isempty(newNumCh)
                origNumCh = origLyr.NumChannels;
                if newNumCh ~= origNumCh
                    fprintf('    Rebuild BN  : %s  ch %d -> %d\n', ...
                        lyrName, origNumCh, newNumCh);
                    newLyr = rebuildBNLayer(origLyr, newNumCh);
                    newLyr.TrainedMean     = lyr.TrainedMean;
                    newLyr.TrainedVariance = lyr.TrainedVariance;
                    if ~isempty(lyr.Offset)
                        newLyr.Offset = lyr.Offset;
                        newLyr.Scale  = lyr.Scale;
                    end
                    lgraphOut = replaceLayer(lgraphOut, lyrName, newLyr);
                else
                    lgraphOut = replaceLayer(lgraphOut, lyrName, lyr);
                end
            end

        else
            % Layer lain (ReLU, pooling, add, dsb.): langsung replace
            try
                lgraphOut = replaceLayer(lgraphOut, lyrName, lyr);
            catch
                % Beberapa layer mungkin tidak bisa di-replace langsung; lewati
            end
        end
    end

    % ----------------------------------------------------------------
    % Perbaiki downstream dari featureLayer (RPN conv, head conv, dsb.)
    % ----------------------------------------------------------------
    fprintf('  Memperbaiki downstream conv & BN dari featureLayer...\n');
    outCh = getOutputChannels(prunedBackboneNet, featureLayerName);
    if isempty(outCh)
        warning('Tidak bisa menentukan output channel dari %s. Lewati fixDownstream.', featureLayerName);
    else
        fprintf('  Output channel featureLayer "%s": %d\n', featureLayerName, outCh);
        lgraphOut = fixDownstreamConvs(lgraphOut, featureLayerName, outCh);
    end

    fprintf('  injectAndFixGraph selesai.\n');
end

% ------------------------------------------------------------------
% evaluateDetectorAP: evaluasi Average Precision pada validation set
%
%   SIGNATURE: evaluateDetectorAP(det, imdsVal, bldsVal)
%
%   det     : fasterRCNNObjectDetector
%   imdsVal : imageDatastore (path gambar 224x224, ReadFcn = imread default)
%             → dipakai oleh detect() untuk inferensi
%   bldsVal : boxLabelDatastore (ground-truth bbox per kelas)
%             → dipakai oleh evaluateDetectionPrecision() untuk metrik AP
%
%   KENAPA PISAH imds & blds (bukan satu table)?
%   - detect() mensyaratkan imageDatastore sebagai input agar bisa
%     membaca gambar secara batch dan mengembalikan tabel hasil deteksi
%     dengan schema {Boxes, Scores, Labels} — tidak bisa dari table biasa.
%   - evaluateDetectionPrecision() membandingkan hasil deteksi dengan
%     ground-truth dari boxLabelDatastore (bukan table mentah).
%   - Memisahkan keduanya memastikan tidak ada type-mismatch yang menjadi
%     sumber error umum saat validasi ("Unrecognized variable name 'Boxes'").
% ------------------------------------------------------------------
function ap = evaluateDetectorAP(det, imdsVal, bldsVal)
    ap = 0;
    try
        % Jalankan deteksi pada seluruh validation set menggunakan imageDatastore
        % detect() membaca gambar via ReadFcn (imread) dan mengembalikan tabel:
        %   columns: Boxes (cell), Scores (cell), Labels (cell)
        results = detect(det, imdsVal, ...
            'MiniBatchSize',      1, ...
            'Threshold',          0.3, ...
            'ExecutionEnvironment', 'auto');

        % Cek apakah ada hasil deteksi yang tidak kosong
        if height(results) == 0
            warning('evaluateDetectorAP: detect() mengembalikan 0 baris hasil.');
            return;
        end
        hasDetection = any(cellfun(@(x) ~isempty(x), results.Boxes));
        if ~hasDetection
            warning('evaluateDetectorAP: Semua deteksi kosong (threshold terlalu tinggi?).');
            return;
        end

        % evaluateDetectionPrecision(detResults, groundTruth)
        %   detResults  : tabel dari detect()
        %   groundTruth : boxLabelDatastore (BUKAN table, BUKAN imageDatastore)
        %   → Mengembalikan AP per kelas dalam tabel atau array numerik
        [apVals, ~, ~] = evaluateDetectionPrecision(results, bldsVal);

        % Ekstrak nilai numerik AP (bisa jadi tabel atau array)
        if isnumeric(apVals)
            apNum = apVals(isfinite(apVals));
        elseif istable(apVals)
            col = apVals{:, end};
            if iscell(col), col = cell2mat(col); end
            apNum = col(isfinite(col));
        else
            apNum = [];
        end

        if isempty(apNum)
            ap = 0;
        else
            ap = double(mean(apNum(:)));
        end

    catch ME
        warning('evaluateDetectorAP error: %s\n  (file: %s, line: %d)', ...
            ME.message, ME.stack(1).file, ME.stack(1).line);
        ap = 0;
    end
end

% ------------------------------------------------------------------
% readAndNormalize224: ReadFcn untuk imageDatastore training (pruning loop)
%
%   Dipakai sebagai: imdsTrain_full.ReadFcn = @readAndNormalize224
%
%   MENGAPA FUNGSI INI DIPERLUKAN?
%   - Gambar di disk sudah berukuran 224x224 (hasil resize Section 1),
%     sehingga tidak perlu imresize() lagi → hemat memori & waktu.
%   - minibatchqueue + prepBatchFast mengharapkan input single [0,1],
%     bukan uint8 [0,255]. ReadFcn ini melakukan konversi itu.
%   - Untuk detect() dan evaluateDetectorAP kita TIDAK menggunakan
%     ReadFcn ini (imdsVal memakai ReadFcn default = imread) karena
%     detect() membutuhkan gambar uint8 asli — bukan normalized float.
%
%   PENTING:
%   - Hanya diset pada imdsTrain_full (untuk pruning loop via minibatchqueue)
%   - JANGAN set ReadFcn ini pada imdsVal, karena detect() akan error
%     jika menerima gambar single [0,1] sebagai input.
% ------------------------------------------------------------------
function img = readAndNormalize224(filename)
    % Baca gambar dari disk (sudah 224x224 karena hasil resize Section 1)
    img = imread(filename);

    % Pastikan 3 channel (RGB)
    if size(img, 3) == 1
        img = repmat(img, [1 1 3]);
    elseif size(img, 3) == 4
        img = img(:, :, 1:3);
    end

    % Konversi ke single [0, 1] untuk input backbone dlnetwork
    img = single(img) / 255;

    % Pastikan ukuran tepat 224x224 (antisipasi edge case)
    if size(img,1) ~= 224 || size(img,2) ~= 224
        img = imresize(img, [224 224]);
    end
end

% ------------------------------------------------------------------
% readAnnotationsAug: baca anotasi dari XML (CVAT format)
% ------------------------------------------------------------------
function frameAnnotations = readAnnotationsAug(annotationFile, ORIG_W, ORIG_H)
    xmlData          = xmlread(annotationFile);
    frameAnnotations = struct('bboxes',{},'labels',{},'filename',{});

    trackNodes = xmlData.getElementsByTagName('track');
    if trackNodes.getLength > 0
        for t = 0 : trackNodes.getLength - 1
            track      = trackNodes.item(t);
            trackLabel = strtrim(char(track.getAttribute('label')));
            boxNodes   = track.getElementsByTagName('box');
            for b = 0 : boxNodes.getLength - 1
                box = boxNodes.item(b);
                if str2double(char(box.getAttribute('outside'))) == 1, continue; end
                frameNum = str2double(char(box.getAttribute('frame')));
                xtl = max(0, str2double(char(box.getAttribute('xtl'))));
                ytl = max(0, str2double(char(box.getAttribute('ytl'))));
                xbr = min(ORIG_W, str2double(char(box.getAttribute('xbr'))));
                ybr = min(ORIG_H, str2double(char(box.getAttribute('ybr'))));
                w = xbr-xtl; h = ybr-ytl;
                if w<=0 || h<=0, continue; end
                fIdx = frameNum+1;
                if fIdx > length(frameAnnotations) || isempty(frameAnnotations(fIdx).bboxes)
                    frameAnnotations(fIdx).bboxes   = zeros(0,4);
                    frameAnnotations(fIdx).labels   = {};
                    frameAnnotations(fIdx).filename = '';
                end
                frameAnnotations(fIdx).bboxes(end+1,:) = [xtl, ytl, w, h];
                frameAnnotations(fIdx).labels{end+1}   = trackLabel;
            end
        end
    else
        imageNodes = xmlData.getElementsByTagName('image');
        for i = 0 : imageNodes.getLength - 1
            imgNode  = imageNodes.item(i);
            imgId    = str2double(char(imgNode.getAttribute('id')));
            imgName  = strtrim(char(imgNode.getAttribute('name')));
            boxNodes = imgNode.getElementsByTagName('box');
            fIdx     = imgId + 1;
            if fIdx > length(frameAnnotations) || isempty(frameAnnotations(fIdx).bboxes)
                frameAnnotations(fIdx).bboxes   = zeros(0,4);
                frameAnnotations(fIdx).labels   = {};
                frameAnnotations(fIdx).filename = imgName;
            end
            for b = 0 : boxNodes.getLength - 1
                box = boxNodes.item(b);
                lbl = strtrim(char(box.getAttribute('label')));
                xtl = max(0, str2double(char(box.getAttribute('xtl'))));
                ytl = max(0, str2double(char(box.getAttribute('ytl'))));
                xbr = min(ORIG_W, str2double(char(box.getAttribute('xbr'))));
                ybr = min(ORIG_H, str2double(char(box.getAttribute('ybr'))));
                w = xbr-xtl; h = ybr-ytl;
                if w<=0 || h<=0, continue; end
                frameAnnotations(fIdx).bboxes(end+1,:) = [xtl, ytl, w, h];
                frameAnnotations(fIdx).labels{end+1}   = lbl;
            end
        end
    end
end

% ------------------------------------------------------------------
% resizeWithPadding: resize gambar ke target size dengan letterbox padding
% ------------------------------------------------------------------
function [imgOut, bboxOut] = resizeWithPadding(img, bboxIn, targetSize, origW, origH)
    tH = targetSize(1); tW = targetSize(2);
    scale = min(tW/origW, tH/origH);
    newW  = round(origW * scale);
    newH  = round(origH * scale);

    if ~isempty(img) && max(img(:)) > 0
        imgResized = imresize(img, [newH, newW]);
    else
        imgResized = zeros(newH, newW, 3, 'uint8');
    end

    padTop  = floor((tH-newH)/2);
    padLeft = floor((tW-newW)/2);
    imgOut  = zeros(tH, tW, 3, 'uint8');
    imgOut(padTop+1:padTop+newH, padLeft+1:padLeft+newW, :) = imgResized;

    if isempty(bboxIn)
        bboxOut = zeros(0,4); return;
    end
    bboxOut      = bboxIn;
    bboxOut(:,1) = bboxIn(:,1)*scale + padLeft;
    bboxOut(:,2) = bboxIn(:,2)*scale + padTop;
    bboxOut(:,3) = bboxIn(:,3)*scale;
    bboxOut(:,4) = bboxIn(:,4)*scale;

    bboxOut(:,1) = max(1, bboxOut(:,1));
    bboxOut(:,2) = max(1, bboxOut(:,2));
    bboxOut(:,3) = min(bboxOut(:,3), tW - bboxOut(:,1) + 1);
    bboxOut(:,4) = min(bboxOut(:,4), tH - bboxOut(:,2) + 1);

    valid   = bboxOut(:,3) >= 2 & bboxOut(:,4) >= 2;
    bboxOut = bboxOut(valid,:);
end