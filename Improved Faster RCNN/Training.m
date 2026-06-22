%% ============================================================
%% TRAINING — Custom Faster R-CNN
%% Backbone : ResNet-50vd + DCN (DeformableConv) + CoordConv
%% Neck     : FPN dengan SPP (Spatial Pyramid Pooling) + Top-down
%% Head     : RPN (4 level: P2-P5) + FC Detection Head
%% Input    : 224x224x3  (letterbox dari 1920x1080)
%% Dataset  : CVAT XML annotations
%%
%% File pendukung (letakkan satu folder):
%%   DeformableConvolution2DLayer.m  CoordConv2DLayer.m
%%   applyBBoxDeltas.m   applyNMS.m   clipBoxes.m   customNMS.m
%%   generateAnchorBoxes.m  generateAnchorGrid.m
%%   generateProposalsPerLevel.m  processRPNOutputs.m  roiPooling.m
%% ============================================================

clear; clc; close all;

%% ============================================================
%% KONFIGURASI
%% ============================================================
datasetFolder  = 'dataset\images_aug';
annotationFile = 'dataset\annotations_aug.xml';
resizedFolder  = 'dataset\images_480';

frameNameFormat = 'frame_%06d';
frameExtension  = '.png';

TARGET_SIZE = [480 480];
ORIG_W = 956;   % ukuran asli dataset (CVAT annotations_aug.xml)
ORIG_H = 720;

% --- Training hyperparameter ---
NUM_EPOCHS        = 100;    % tambah epoch karena training dari scratch / partial pretrained
MINI_BATCH        = 2;

% Freeze backbone N epoch pertama agar head bisa stabil dulu
% Kemudian backbone di-unfreeze dengan LR kecil
FREEZE_BACKBONE_EPOCHS = 3;   % set 0 untuk tidak freeze

BASE_LR_BACKBONE  = 1e-4;
BASE_LR_HEAD      = 1e-3;
LR_WARMUP_ITERS   = 200;
LR_DROP_EPOCHS    = [10, 16];
LR_DROP_FACTOR    = 0.1;

% AdamW (lebih stabil dari SGDM untuk training dari random init)
ADAM_BETA1        = 0.9;
ADAM_BETA2        = 0.999;
ADAM_EPS          = 1e-8;
WEIGHT_DECAY      = 1e-4;

% --- RPN / Proposal ---
NMS_THRESHOLD     = 0.7;
SCORE_THRESHOLD   = 0.05;
MAX_PROPOSALS     = 300;   % per gambar saat training
PRE_NMS_TOPN      = 1000;

% --- Anchor matching ---
% Input 224x224, objek dari 1920x1080 dikecilkan ~8.5x
% → objek yang awalnya 100px menjadi ~12px di 224
% Turunkan posIoU agar lebih banyak anchor jadi positive
POS_IOU_THRESH    = 0.4;   % lebih longgar dari 0.5
NEG_IOU_THRESH_HI = 0.2;   % zona ignore: 0.2–0.4
NUM_RPN_SAMPLES   = 256;
POS_FRACTION      = 0.5;

% --- ROI / Detection head ---
NUM_ROI_SAMPLES   = 128;
ROI_POS_FRACTION  = 0.25;
ROI_POOL_SIZE     = 7;
DET_POS_IOU       = 0.4;   % lebih longgar
DET_NEG_IOU_HI    = 0.3;   % zona ignore: 0.3–0.4 (WAJIB < DET_POS_IOU)

% --- Loss weights ---
% RPN_reg mendominasi karena banyak anchor kosong → turunkan bobotnya
RPN_CLS_W  = 1.0;
RPN_REG_W  = 1.0;
DET_CLS_W  = 2.0;   % naikkan agar detection head dipaksa belajar
DET_REG_W  = 2.0;

% --- Output ---
CHECKPOINT_DIR = 'checkpoints';
SAVE_EVERY     = 2;

% --- Resume Training ---
% Set RESUME_FROM ke path checkpoint untuk melanjutkan training
% Contoh: RESUME_FROM = 'checkpoints\checkpoint_epoch10.mat';
% Set ke '' untuk mulai dari awal
RESUME_FROM    = '';          % ← ISI PATH CHECKPOINT DI SINI
EXTRA_EPOCHS   = 20;          % jumlah epoch TAMBAHAN saat resume
                               % (diabaikan jika RESUME_FROM kosong)   % simpan checkpoint setiap N epoch

fprintf('==============================================\n');
fprintf('  Custom Faster R-CNN — ResNet50vd+DCN+FPN\n');
fprintf('==============================================\n\n');

%% ============================================================
%% STEP 1: Baca Anotasi CVAT XML
%% ============================================================
fprintf('[1/6] Membaca anotasi...\n');

if ~isfile(annotationFile)
    error('File anotasi tidak ditemukan: %s', annotationFile);
end

xmlData    = xmlread(annotationFile);

% Image-based CVAT export: <image name="..." width=.. height=..> with child
% <box label="..." xtl ytl xbr ybr/>. (The previous parser expected a video
% <track><box frame=..> export, which produced ZERO annotations for this data.)
imageNodes = xmlData.getElementsByTagName('image');
imgAnnotations = struct('name', {}, 'bboxes', {}, 'labels', {});
labelSet = {};
for ii = 0:imageNodes.getLength-1
    imgNode = imageNodes.item(ii);
    iname   = strtrim(char(imgNode.getAttribute('name')));
    boxNodes = imgNode.getElementsByTagName('box');
    bb = zeros(0,4); lbls = {};
    for b = 0:boxNodes.getLength-1
        box = boxNodes.item(b);
        lbl = strtrim(char(box.getAttribute('label')));
        xtl = str2double(char(box.getAttribute('xtl')));
        ytl = str2double(char(box.getAttribute('ytl')));
        xbr = str2double(char(box.getAttribute('xbr')));
        ybr = str2double(char(box.getAttribute('ybr')));
        xtl = max(0,xtl); ytl = max(0,ytl);
        xbr = min(ORIG_W,xbr); ybr = min(ORIG_H,ybr);
        if (xbr-xtl)<=0 || (ybr-ytl)<=0, continue; end
        bb(end+1,:) = [xtl, ytl, xbr-xtl, ybr-ytl]; %#ok<SAGROW>  (xywh)
        lbls{end+1} = lbl;                            %#ok<SAGROW>
        labelSet{end+1} = lbl;                        %#ok<SAGROW>
    end
    if isempty(bb), continue; end
    k = numel(imgAnnotations) + 1;
    imgAnnotations(k).name   = iname;
    imgAnnotations(k).bboxes = bb;
    imgAnnotations(k).labels = lbls;
end
labelNames = unique(labelSet);
fprintf('  Label: %s\n', strjoin(labelNames,', '));
fprintf('  Gambar teranotasi: %d\n\n', numel(imgAnnotations));

%% ============================================================
%% STEP 2: Resize + Letterbox, Simpan ke Disk
%% ============================================================
fprintf('[2/6] Resize + letterbox padding → %dx%d...\n', TARGET_SIZE(2),TARGET_SIZE(1));

if ~exist(resizedFolder,'dir'), mkdir(resizedFolder); end

imageFilenames = {};
bboxesAll      = {};
labelsAll      = {};
validCount     = 0;

for k = 1:numel(imgAnnotations)
    iname   = imgAnnotations(k).name;            % e.g. frame_000011_aug1.png
    imgPath = fullfile(datasetFolder, iname);
    if ~isfile(imgPath), continue; end
    resizedPath = fullfile(resizedFolder, iname);
    try
        img = imread(imgPath);
        if size(img,3)==1,  img = repmat(img,[1 1 3]); end
        if size(img,3)==4,  img = img(:,:,1:3);         end

        [imgOut, bboxOut, validMask] = helperResizePad( ...
            img, imgAnnotations(k).bboxes, TARGET_SIZE, ORIG_W, ORIG_H);
    catch, continue; end

    if isempty(bboxOut), continue; end
    if ~isfile(resizedPath)
        try; imwrite(imgOut, resizedPath); catch; continue; end
    end

    validCount = validCount+1;
    imageFilenames{validCount} = resizedPath;                        %#ok<SAGROW>
    bboxesAll{validCount}      = bboxOut;                            %#ok<SAGROW>
    rawLbls = imgAnnotations(k).labels;
    filteredLbls = rawLbls(validMask);
    labelsAll{validCount} = filteredLbls(:);                         %#ok<SAGROW>

    if mod(validCount,200)==0
        fprintf('  %d gambar diproses...\n', validCount);
    end
end

fprintf('  Total gambar valid: %d\n', validCount);
if validCount==0
    error('Tidak ada gambar valid! Periksa path dan format nama file.');
end

%% ============================================================
%% STEP 3: Bangun Dataset & Class Names
%% ============================================================
fprintf('\n[3/6] Membangun dataset...\n');

% Gabungkan semua label — paksa tiap elemen jadi column dulu
allLabelsCells = cellfun(@(x) x(:), labelsAll, 'UniformOutput', false);
allLabelsFlat  = vertcat(allLabelsCells{:});
classNames     = unique(allLabelsFlat(:));
numClasses    = length(classNames);
fprintf('  Classes (%d): %s\n', numClasses, strjoin(classNames,', '));

% Label encoding: class index (1-based) + background = 0
% background class index = numClasses+1 (untuk FC head)
BG_CLASS = numClasses + 1;

% Split 80/20
rng(42);
idx      = randperm(validCount);
nTrain   = round(0.8 * validCount);
trainIdx = idx(1:nTrain);
valIdx   = idx(nTrain+1:end);
fprintf('  Train: %d | Val: %d\n\n', length(trainIdx), length(valIdx));

%% ============================================================
%% STEP 4: Bangun Network (DARI Faster_RCNN_Modified.m)
%% ============================================================
fprintf('[4/6] Membangun custom network...\n');

imageHeight   = TARGET_SIZE(1);
imageWidth    = TARGET_SIZE(2);
imageChannels = 3;
numAnchors    = 2;   % per location, sesuai arsitektur (scores=4ch → 2 anchor×2)
roiPoolSize   = ROI_POOL_SIZE;

% Tambahkan path custom layer jika ada subfolder
if exist(fullfile(pwd,'CustomLayer'),'dir'),            addpath(fullfile(pwd,'CustomLayer'));            end
if exist(fullfile(pwd,'RegionProposalNetwork'),'dir'),  addpath(fullfile(pwd,'RegionProposalNetwork'));  end
if exist(fullfile(pwd,'RoiPooling'),'dir'),             addpath(fullfile(pwd,'RoiPooling'));             end

%% ---- Bangun backbone + FPN + RPN (identik dengan Faster_RCNN_Modified.m) ----
net = dlnetwork;

tempNet = [
    imageInputLayer([imageHeight imageWidth 3],"Name","imageinput")
    convolution2dLayer([3 3],32,"Name","conv","Padding","same","Stride",[2 2])
    convolution2dLayer([3 3],32,"Name","conv_2","Padding","same")
    convolution2dLayer([3 3],64,"Name","conv_1","Padding","same")
    maxPooling2dLayer([3 3],"Name","maxpool","Padding","same","Stride",[2 2])];
net = addLayers(net,tempNet);

% Stage 1 (C2) - 3 bottleneck blocks
tempNet = [
    convolution2dLayer([1 1],64,"Name","conv_3","Padding","same")
    batchNormalizationLayer("Name","batchnorm")
    reluLayer("Name","relu")
    convolution2dLayer([3 3],64,"Name","conv_4","Padding","same")
    batchNormalizationLayer("Name","batchnorm_1")
    reluLayer("Name","relu_1")
    convolution2dLayer([1 1],256,"Name","conv_5","Padding","same")
    batchNormalizationLayer("Name","batchnorm_2")];
net = addLayers(net,tempNet);
tempNet = [convolution2dLayer([1 1],256,"Name","conv_6","Padding","same"); batchNormalizationLayer("Name","batchnorm_3")];
net = addLayers(net,tempNet);
tempNet = [additionLayer(2,"Name","addition"); reluLayer("Name","relu_2")];
net = addLayers(net,tempNet);

tempNet = [
    convolution2dLayer([1 1],64,"Name","conv_7","Padding","same")
    batchNormalizationLayer("Name","batchnorm_4")
    reluLayer("Name","relu_3")
    convolution2dLayer([3 3],64,"Name","conv_8","Padding","same")
    batchNormalizationLayer("Name","batchnorm_5")
    reluLayer("Name","relu_4")
    convolution2dLayer([1 1],256,"Name","conv_9","Padding","same")
    batchNormalizationLayer("Name","batchnorm_6")];
net = addLayers(net,tempNet);
tempNet = [additionLayer(2,"Name","addition_1"); reluLayer("Name","relu_5")];
net = addLayers(net,tempNet);

tempNet = [
    convolution2dLayer([1 1],64,"Name","conv_10","Padding","same")
    batchNormalizationLayer("Name","batchnorm_7")
    reluLayer("Name","relu_6")
    convolution2dLayer([3 3],64,"Name","conv_11","Padding","same")
    batchNormalizationLayer("Name","batchnorm_8")
    reluLayer("Name","relu_7")
    convolution2dLayer([1 1],256,"Name","conv_12","Padding","same")
    batchNormalizationLayer("Name","batchnorm_9")];
net = addLayers(net,tempNet);
tempNet = [additionLayer(2,"Name","addition_2"); reluLayer("Name","relu_8")];
net = addLayers(net,tempNet);

% Stage 2 (C3) - 4 bottleneck blocks dengan DeformableConv
tempNet = [
    convolution2dLayer([1 1],128,"Name","conv_13","Padding","same")
    batchNormalizationLayer("Name","batchnorm_10")
    reluLayer("Name","relu_9")
    DeformableConvolution2DLayer([3 3],128,"Name","deformConv","Padding","same","Stride",[2 2])
    batchNormalizationLayer("Name","batchnorm_12")
    reluLayer("Name","relu_10")
    convolution2dLayer([1 1],512,"Name","conv_15","Padding","same")
    batchNormalizationLayer("Name","batchnorm_13")];
net = addLayers(net,tempNet);
tempNet = [
    averagePooling2dLayer([2 2],"Name","avgpool2d","Padding","same","Stride",[2 2])
    convolution2dLayer([1 1],512,"Name","conv_14","Padding","same")
    batchNormalizationLayer("Name","batchnorm_11")];
net = addLayers(net,tempNet);
tempNet = [additionLayer(2,"Name","addition_3"); reluLayer("Name","relu_11")];
net = addLayers(net,tempNet);

tempNet = [
    convolution2dLayer([1 1],128,"Name","conv_17","Padding","same")
    batchNormalizationLayer("Name","batchnorm_14")
    reluLayer("Name","relu_12")
    DeformableConvolution2DLayer([3 3],128,"Name","deformConv_1","Padding","same")
    batchNormalizationLayer("Name","batchnorm_15")
    reluLayer("Name","relu_13")
    convolution2dLayer([1 1],512,"Name","conv_16","Padding","same")
    batchNormalizationLayer("Name","batchnorm_16")];
net = addLayers(net,tempNet);
tempNet = [additionLayer(2,"Name","addition_4"); reluLayer("Name","relu_14")];
net = addLayers(net,tempNet);

tempNet = [
    convolution2dLayer([1 1],128,"Name","conv_19","Padding","same")
    batchNormalizationLayer("Name","batchnorm_17")
    reluLayer("Name","relu_15")
    DeformableConvolution2DLayer([3 3],128,"Name","deformConv_2","Padding","same")
    batchNormalizationLayer("Name","batchnorm_18")
    reluLayer("Name","relu_16")
    convolution2dLayer([1 1],512,"Name","conv_18","Padding","same")
    batchNormalizationLayer("Name","batchnorm_19")];
net = addLayers(net,tempNet);
tempNet = [additionLayer(2,"Name","addition_5"); reluLayer("Name","relu_17")];
net = addLayers(net,tempNet);

tempNet = [
    convolution2dLayer([1 1],128,"Name","conv_21","Padding","same")
    batchNormalizationLayer("Name","batchnorm_20")
    reluLayer("Name","relu_18")
    DeformableConvolution2DLayer([3 3],128,"Name","deformConv_3","Padding","same")
    batchNormalizationLayer("Name","batchnorm_21")
    reluLayer("Name","relu_19")
    convolution2dLayer([1 1],512,"Name","conv_20","Padding","same")
    batchNormalizationLayer("Name","batchnorm_22")];
net = addLayers(net,tempNet);
tempNet = [additionLayer(2,"Name","addition_6"); reluLayer("Name","relu_20")];
net = addLayers(net,tempNet);

% Stage 3 (C4) - 6 bottleneck blocks dengan DeformableConv
tempNet = [
    convolution2dLayer([1 1],256,"Name","conv_22","Padding","same")
    batchNormalizationLayer("Name","batchnorm_23")
    reluLayer("Name","relu_21")
    DeformableConvolution2DLayer([3 3],256,"Name","deformConv_4","Padding","same","Stride",[2 2])
    batchNormalizationLayer("Name","batchnorm_25")
    reluLayer("Name","relu_22")
    convolution2dLayer([1 1],1024,"Name","conv_23","Padding","same")
    batchNormalizationLayer("Name","batchnorm_26")];
net = addLayers(net,tempNet);
tempNet = [
    averagePooling2dLayer([2 2],"Name","avgpool2d_1","Padding","same","Stride",[2 2])
    convolution2dLayer([1 1],1024,"Name","conv_41","Padding","same")
    batchNormalizationLayer("Name","batchnorm_24")];
net = addLayers(net,tempNet);
tempNet = [additionLayer(2,"Name","addition_7"); reluLayer("Name","relu_23")];
net = addLayers(net,tempNet);

tempNet = [
    convolution2dLayer([1 1],256,"Name","conv_25","Padding","same")
    batchNormalizationLayer("Name","batchnorm_27")
    reluLayer("Name","relu_24")
    DeformableConvolution2DLayer([3 3],256,"Name","deformConv_5","Padding","same")
    batchNormalizationLayer("Name","batchnorm_28")
    reluLayer("Name","relu_25")
    convolution2dLayer([1 1],1024,"Name","conv_24","Padding","same")
    batchNormalizationLayer("Name","batchnorm_29")];
net = addLayers(net,tempNet);
tempNet = [additionLayer(2,"Name","addition_8"); reluLayer("Name","relu_26")];
net = addLayers(net,tempNet);

tempNet = [
    convolution2dLayer([1 1],256,"Name","conv_27","Padding","same")
    batchNormalizationLayer("Name","batchnorm_30")
    reluLayer("Name","relu_27")
    DeformableConvolution2DLayer([3 3],256,"Name","deformConv_6","Padding","same")
    batchNormalizationLayer("Name","batchnorm_31")
    reluLayer("Name","relu_28")
    convolution2dLayer([1 1],1024,"Name","conv_26","Padding","same")
    batchNormalizationLayer("Name","batchnorm_32")];
net = addLayers(net,tempNet);
tempNet = [additionLayer(2,"Name","addition_9"); reluLayer("Name","relu_29")];
net = addLayers(net,tempNet);

tempNet = [
    convolution2dLayer([1 1],256,"Name","conv_29","Padding","same")
    batchNormalizationLayer("Name","batchnorm_33")
    reluLayer("Name","relu_30")
    DeformableConvolution2DLayer([3 3],256,"Name","deformConv_7","Padding","same")
    batchNormalizationLayer("Name","batchnorm_34")
    reluLayer("Name","relu_31")
    convolution2dLayer([1 1],1024,"Name","conv_28","Padding","same")
    batchNormalizationLayer("Name","batchnorm_35")];
net = addLayers(net,tempNet);
tempNet = [additionLayer(2,"Name","addition_10"); reluLayer("Name","relu_32")];
net = addLayers(net,tempNet);

tempNet = [
    convolution2dLayer([1 1],256,"Name","conv_31","Padding","same")
    batchNormalizationLayer("Name","batchnorm_36")
    reluLayer("Name","relu_33")
    DeformableConvolution2DLayer([3 3],256,"Name","deformConv_8","Padding","same")
    batchNormalizationLayer("Name","batchnorm_37")
    reluLayer("Name","relu_34")
    convolution2dLayer([1 1],1024,"Name","conv_30","Padding","same")
    batchNormalizationLayer("Name","batchnorm_38")];
net = addLayers(net,tempNet);
tempNet = [additionLayer(2,"Name","addition_11"); reluLayer("Name","relu_35")];
net = addLayers(net,tempNet);

tempNet = [
    convolution2dLayer([1 1],256,"Name","conv_33","Padding","same")
    batchNormalizationLayer("Name","batchnorm_39")
    reluLayer("Name","relu_36")
    DeformableConvolution2DLayer([3 3],256,"Name","deformConv_9","Padding","same")
    batchNormalizationLayer("Name","batchnorm_40")
    reluLayer("Name","relu_37")
    convolution2dLayer([1 1],1024,"Name","conv_32","Padding","same")
    batchNormalizationLayer("Name","batchnorm_41")];
net = addLayers(net,tempNet);
tempNet = [additionLayer(2,"Name","addition_12"); reluLayer("Name","relu_38")];
net = addLayers(net,tempNet);

% Stage 4 (C5) - 3 bottleneck blocks dengan DeformableConv
tempNet = [
    convolution2dLayer([1 1],512,"Name","conv_34","Padding","same")
    batchNormalizationLayer("Name","batchnorm_42")
    reluLayer("Name","relu_39")
    DeformableConvolution2DLayer([3 3],512,"Name","deformConv_10","Padding","same","Stride",[2 2])
    batchNormalizationLayer("Name","batchnorm_44")
    reluLayer("Name","relu_40")
    convolution2dLayer([1 1],2048,"Name","conv_36","Padding","same")
    batchNormalizationLayer("Name","batchnorm_45")];
net = addLayers(net,tempNet);
tempNet = [
    averagePooling2dLayer([2 2],"Name","avgpool2d_2","Padding","same","Stride",[2 2])
    convolution2dLayer([1 1],2048,"Name","conv_35","Padding","same")
    batchNormalizationLayer("Name","batchnorm_43")];
net = addLayers(net,tempNet);
tempNet = [additionLayer(2,"Name","addition_13"); reluLayer("Name","relu_41")];
net = addLayers(net,tempNet);

tempNet = [
    convolution2dLayer([1 1],512,"Name","conv_38","Padding","same")
    batchNormalizationLayer("Name","batchnorm_46")
    reluLayer("Name","relu_42")
    DeformableConvolution2DLayer([3 3],512,"Name","deformConv_11","Padding","same")
    batchNormalizationLayer("Name","batchnorm_47")
    reluLayer("Name","relu_43")
    convolution2dLayer([1 1],2048,"Name","conv_37","Padding","same")
    batchNormalizationLayer("Name","batchnorm_48")];
net = addLayers(net,tempNet);
tempNet = [additionLayer(2,"Name","addition_14"); reluLayer("Name","relu_44")];
net = addLayers(net,tempNet);

tempNet = [
    convolution2dLayer([1 1],512,"Name","conv_40","Padding","same")
    batchNormalizationLayer("Name","batchnorm_49")
    reluLayer("Name","relu_45")
    DeformableConvolution2DLayer([3 3],512,"Name","deformConv_12","Padding","same")
    batchNormalizationLayer("Name","batchnorm_50")
    reluLayer("Name","relu_46")
    convolution2dLayer([1 1],2048,"Name","conv_39","Padding","same")
    batchNormalizationLayer("Name","batchnorm_51")];
net = addLayers(net,tempNet);

% C5 top + SPP per level (identik dengan Faster_RCNN_Modified.m)
tempNet = [
    additionLayer(2,"Name","addition_15")
    reluLayer("Name","relu_47")
    CoordConv2DLayer([1 1],256,"Name","coordConv_9","Padding","same")
    convolution2dLayer([3 3],512,"Name","conv_51","Padding","same")
    CoordConv2DLayer([1 1],256,"Name","coordConv_10","Padding","same")];
net = addLayers(net,tempNet);

% P2 branch — dari C2 relu_8
tempNet = [
    CoordConv2DLayer([1 1],256,"Name","coordConv","Padding","same")
    convolution2dLayer([3 3],512,"Name","conv_42","Padding","same")
    CoordConv2DLayer([1 1],256,"Name","coordConv_1","Padding","same")];
net = addLayers(net,tempNet);
net = addLayers(net, maxPooling2dLayer([5 5], "Name","maxpool_1","Padding","same"));
net = addLayers(net, maxPooling2dLayer([9 9], "Name","maxpool_2","Padding","same"));
net = addLayers(net, maxPooling2dLayer([13 13],"Name","maxpool_3","Padding","same"));
tempNet = [
    concatenationLayer(3,4,"Name","concat")
    convolution2dLayer([1 1],512,"Name","conv_43","Padding","same")
    batchNormalizationLayer("Name","batchnorm_52")
    convolution2dLayer([3 3],512,"Name","conv_44","Padding","same")
    CoordConv2DLayer([1 1],256,"Name","coordConv_2","Padding","same")];
net = addLayers(net,tempNet);

% P3 branch — dari C3 relu_20
tempNet = [
    CoordConv2DLayer([1 1],256,"Name","coordConv_3","Padding","same")
    convolution2dLayer([3 3],512,"Name","conv_45","Padding","same")
    CoordConv2DLayer([1 1],256,"Name","coordConv_4","Padding","same")];
net = addLayers(net,tempNet);
net = addLayers(net, maxPooling2dLayer([5 5], "Name","maxpool_4","Padding","same"));
net = addLayers(net, maxPooling2dLayer([9 9], "Name","maxpool_5","Padding","same"));
net = addLayers(net, maxPooling2dLayer([13 13],"Name","maxpool_6","Padding","same"));
tempNet = [
    concatenationLayer(3,4,"Name","concat_2")
    convolution2dLayer([1 1],512,"Name","conv_46","Padding","same")
    batchNormalizationLayer("Name","batchnorm_53")
    convolution2dLayer([3 3],512,"Name","conv_47","Padding","same")
    CoordConv2DLayer([1 1],256,"Name","coordConv_5","Padding","same")];
net = addLayers(net,tempNet);

% P4 branch — dari C4 relu_38
tempNet = [
    CoordConv2DLayer([1 1],256,"Name","coordConv_6","Padding","same")
    convolution2dLayer([3 3],512,"Name","conv_48","Padding","same")
    CoordConv2DLayer([1 1],256,"Name","coordConv_7","Padding","same")];
net = addLayers(net,tempNet);
net = addLayers(net, maxPooling2dLayer([5 5], "Name","maxpool_7","Padding","same"));
net = addLayers(net, maxPooling2dLayer([9 9], "Name","maxpool_8","Padding","same"));
net = addLayers(net, maxPooling2dLayer([13 13],"Name","maxpool_9","Padding","same"));
tempNet = [
    concatenationLayer(3,4,"Name","concat_4")
    convolution2dLayer([1 1],512,"Name","conv_49","Padding","same")
    batchNormalizationLayer("Name","batchnorm_54")
    convolution2dLayer([3 3],512,"Name","conv_50","Padding","same")
    CoordConv2DLayer([1 1],256,"Name","coordConv_8","Padding","same")];
net = addLayers(net,tempNet);

% P5 branch — dari C5 coordConv_10
net = addLayers(net, maxPooling2dLayer([5 5], "Name","maxpool_10","Padding","same"));
net = addLayers(net, maxPooling2dLayer([9 9], "Name","maxpool_11","Padding","same"));
net = addLayers(net, maxPooling2dLayer([13 13],"Name","maxpool_12","Padding","same"));
tempNet = [
    concatenationLayer(3,4,"Name","concat_6")
    convolution2dLayer([1 1],512,"Name","conv_52","Padding","same")
    batchNormalizationLayer("Name","batchnorm_55")
    convolution2dLayer([3 3],512,"Name","conv_53","Padding","same")
    CoordConv2DLayer([1 1],256,"Name","coordConv_11","Padding","same")];
net = addLayers(net,tempNet);

% Top-down FPN merges (resize + concat)
net = addLayers(net, resize2dLayer("Name","resize-scale","GeometricTransformMode","half-pixel","Method","nearest","NearestRoundingMode","round","Scale",[2 2]));
tempNet = [concatenationLayer(3,2,"Name","concat_1"); convolution2dLayer([3 3],256,"Name","conv_54","Padding","same")];
net = addLayers(net,tempNet);

net = addLayers(net, resize2dLayer("Name","resize-scale_1","GeometricTransformMode","half-pixel","Method","nearest","NearestRoundingMode","round","Scale",[2 2]));
tempNet = [concatenationLayer(3,2,"Name","concat_3"); convolution2dLayer([3 3],256,"Name","conv_55","Padding","same")];
net = addLayers(net,tempNet);

net = addLayers(net, resize2dLayer("Name","resize-scale_2","GeometricTransformMode","half-pixel","Method","nearest","NearestRoundingMode","round","Scale",[2 2]));
tempNet = [concatenationLayer(3,2,"Name","concat_5"); convolution2dLayer([3 3],256,"Name","conv_56","Padding","same")];
net = addLayers(net,tempNet);

% RPN output heads per level
net = addLayers(net, convolution2dLayer([1 1],4,"Name","scoresP2","Padding","same"));
net = addLayers(net, convolution2dLayer([1 1],8,"Name","boxDeltasP2","Padding","same"));
net = addLayers(net, convolution2dLayer([1 1],4,"Name","scoresP3","Padding","same"));
net = addLayers(net, convolution2dLayer([1 1],8,"Name","boxDeltasP3","Padding","same"));
net = addLayers(net, convolution2dLayer([1 1],4,"Name","scoresP4","Padding","same"));
net = addLayers(net, convolution2dLayer([1 1],8,"Name","boxDeltasP4","Padding","same"));
net = addLayers(net, convolution2dLayer([3 3],256,"Name","conv_57","Padding","same"));
net = addLayers(net, convolution2dLayer([1 1],4,"Name","scoresP5","Padding","same"));
net = addLayers(net, convolution2dLayer([1 1],8,"Name","boxDeltasP5","Padding","same"));
clear tempNet;

%% ---- Koneksi antar layer (identik dengan Faster_RCNN_Modified.m) ----
net = connectLayers(net,"maxpool","conv_3");
net = connectLayers(net,"maxpool","conv_6");
net = connectLayers(net,"batchnorm_2","addition/in1");
net = connectLayers(net,"batchnorm_3","addition/in2");
net = connectLayers(net,"relu_2","conv_7");
net = connectLayers(net,"relu_2","addition_1/in2");
net = connectLayers(net,"batchnorm_6","addition_1/in1");
net = connectLayers(net,"relu_5","conv_10");
net = connectLayers(net,"relu_5","addition_2/in2");
net = connectLayers(net,"batchnorm_9","addition_2/in1");
net = connectLayers(net,"relu_8","conv_13");
net = connectLayers(net,"relu_8","avgpool2d");
net = connectLayers(net,"relu_8","coordConv");
net = connectLayers(net,"batchnorm_11","addition_3/in2");
net = connectLayers(net,"batchnorm_13","addition_3/in1");
net = connectLayers(net,"relu_11","conv_17");
net = connectLayers(net,"relu_11","addition_4/in2");
net = connectLayers(net,"batchnorm_16","addition_4/in1");
net = connectLayers(net,"relu_14","conv_19");
net = connectLayers(net,"relu_14","addition_5/in2");
net = connectLayers(net,"batchnorm_19","addition_5/in1");
net = connectLayers(net,"relu_17","conv_21");
net = connectLayers(net,"relu_17","addition_6/in2");
net = connectLayers(net,"batchnorm_22","addition_6/in1");
net = connectLayers(net,"relu_20","conv_22");
net = connectLayers(net,"relu_20","avgpool2d_1");
net = connectLayers(net,"relu_20","coordConv_3");
net = connectLayers(net,"batchnorm_26","addition_7/in1");
net = connectLayers(net,"batchnorm_24","addition_7/in2");
net = connectLayers(net,"relu_23","conv_25");
net = connectLayers(net,"relu_23","addition_8/in2");
net = connectLayers(net,"batchnorm_29","addition_8/in1");
net = connectLayers(net,"relu_26","conv_27");
net = connectLayers(net,"relu_26","addition_9/in2");
net = connectLayers(net,"batchnorm_32","addition_9/in1");
net = connectLayers(net,"relu_29","conv_29");
net = connectLayers(net,"relu_29","addition_10/in2");
net = connectLayers(net,"batchnorm_35","addition_10/in1");
net = connectLayers(net,"relu_32","conv_31");
net = connectLayers(net,"relu_32","addition_11/in2");
net = connectLayers(net,"batchnorm_38","addition_11/in1");
net = connectLayers(net,"relu_35","conv_33");
net = connectLayers(net,"relu_35","addition_12/in2");
net = connectLayers(net,"batchnorm_41","addition_12/in1");
net = connectLayers(net,"relu_38","conv_34");
net = connectLayers(net,"relu_38","avgpool2d_2");
net = connectLayers(net,"relu_38","coordConv_6");
net = connectLayers(net,"batchnorm_43","addition_13/in2");
net = connectLayers(net,"batchnorm_45","addition_13/in1");
net = connectLayers(net,"relu_41","conv_38");
net = connectLayers(net,"relu_41","addition_14/in2");
net = connectLayers(net,"batchnorm_48","addition_14/in1");
net = connectLayers(net,"relu_44","conv_40");
net = connectLayers(net,"relu_44","addition_15/in2");
net = connectLayers(net,"batchnorm_51","addition_15/in1");
net = connectLayers(net,"coordConv_1","maxpool_1");
net = connectLayers(net,"coordConv_1","maxpool_2");
net = connectLayers(net,"coordConv_1","maxpool_3");
net = connectLayers(net,"coordConv_1","concat/in4");
net = connectLayers(net,"maxpool_1","concat/in2");
net = connectLayers(net,"maxpool_2","concat/in1");
net = connectLayers(net,"maxpool_3","concat/in3");
net = connectLayers(net,"coordConv_2","concat_1/in1");
net = connectLayers(net,"coordConv_4","maxpool_4");
net = connectLayers(net,"coordConv_4","maxpool_5");
net = connectLayers(net,"coordConv_4","maxpool_6");
net = connectLayers(net,"coordConv_4","concat_2/in4");
net = connectLayers(net,"maxpool_4","concat_2/in2");
net = connectLayers(net,"maxpool_5","concat_2/in1");
net = connectLayers(net,"maxpool_6","concat_2/in3");
net = connectLayers(net,"coordConv_5","resize-scale");
net = connectLayers(net,"coordConv_5","concat_3/in1");
net = connectLayers(net,"coordConv_7","maxpool_7");
net = connectLayers(net,"coordConv_7","maxpool_8");
net = connectLayers(net,"coordConv_7","maxpool_9");
net = connectLayers(net,"coordConv_7","concat_4/in4");
net = connectLayers(net,"maxpool_7","concat_4/in2");
net = connectLayers(net,"maxpool_8","concat_4/in1");
net = connectLayers(net,"maxpool_9","concat_4/in3");
net = connectLayers(net,"coordConv_8","resize-scale_1");
net = connectLayers(net,"coordConv_8","concat_5/in1");
net = connectLayers(net,"coordConv_10","maxpool_10");
net = connectLayers(net,"coordConv_10","maxpool_11");
net = connectLayers(net,"coordConv_10","maxpool_12");
net = connectLayers(net,"coordConv_10","concat_6/in4");
net = connectLayers(net,"maxpool_10","concat_6/in2");
net = connectLayers(net,"maxpool_11","concat_6/in1");
net = connectLayers(net,"maxpool_12","concat_6/in3");
net = connectLayers(net,"coordConv_11","resize-scale_2");
net = connectLayers(net,"coordConv_11","conv_57");
net = connectLayers(net,"resize-scale","concat_1/in2");
net = connectLayers(net,"resize-scale_1","concat_3/in2");
net = connectLayers(net,"resize-scale_2","concat_5/in2");
net = connectLayers(net,"conv_54","scoresP2");
net = connectLayers(net,"conv_54","boxDeltasP2");
net = connectLayers(net,"conv_55","scoresP3");
net = connectLayers(net,"conv_55","boxDeltasP3");
net = connectLayers(net,"conv_56","scoresP4");
net = connectLayers(net,"conv_56","boxDeltasP4");
net = connectLayers(net,"conv_57","scoresP5");
net = connectLayers(net,"conv_57","boxDeltasP5");
net = initialize(net);

fprintf('  ✓ Backbone+FPN+RPN diinisialisasi\n');

%% ---- FC Detection Head (identik dengan Faster_RCNN_Modified.m) ----
% Hitung channel feature map dari output net (conv_54 → 256 channel)
numFMChannels = 256;

fcNet = dlnetwork;
tempNet = [
    imageInputLayer([roiPoolSize roiPoolSize numFMChannels], ...
        "Name","roiInput","Normalization","none")
    globalAveragePooling2dLayer("Name","avgpool_fc1")
    fullyConnectedLayer(1024,"Name","fc1")
    reluLayer("Name","relu_fc1")
    fullyConnectedLayer(1024,"Name","fc2")
    reluLayer("Name","relu_fc2")];
fcNet = addLayers(fcNet,tempNet);
tempNet = [
    fullyConnectedLayer(numClasses+1,"Name","fc3_cls")   % +1 untuk background
    softmaxLayer("Name","classScores")];
fcNet = addLayers(fcNet,tempNet);
tempNet = fullyConnectedLayer(numClasses*4,"Name","fc3_bbox");  % hanya fg class
fcNet = addLayers(fcNet,tempNet);
clear tempNet;
fcNet = connectLayers(fcNet,"relu_fc2","fc3_cls");
fcNet = connectLayers(fcNet,"relu_fc2","fc3_bbox");
fcNet = initialize(fcNet);

fprintf('  ✓ FC Detection Head diinisialisasi\n');
fprintf('  Classes: %d + 1 background = %d output neurons\n\n', numClasses, numClasses+1);

%% ============================================================
%% STEP 4b: Transfer Pretrained Weights dari ResNet-50
%% ============================================================
fprintf('[4b] Transfer pretrained weights dari ResNet-50...\n');
fprintf('  (backbone diinisialisasi random — pretrained transfer jauh lebih baik)\n');

try
    resnetPretrained = resnet50;

    % Peta nama layer ResNet-50 MATLAB → nama layer custom network kita
    % Backbone custom kita mengikuti arsitektur ResNet-50vd:
    %   conv1 3×3 → 'conv', 'conv_2', 'conv_1'
    %   Stage1 (C2) bottleneck blocks → conv_3..conv_12
    %   Stage2 (C3) dengan DCN → conv_13..conv_21 (skip DCN, salin BN+1x1)
    %   dst.

    % Ambil semua learnable dari ResNet-50 pretrained
    rnLearn = resnetPretrained.Learnables;
    rnNames = rnLearn.Layer;

    % Ambil semua learnable dari custom net kita
    netLearn = net.Learnables;

    % Strategi transfer: salin layer yang UKURAN PERSIS SAMA
    % (conv weight shape dan bias shape cocok)
    numTransferred = 0;
    numSkipped     = 0;

    for i = 1:height(netLearn)
        myLayerName = netLearn.Layer(i);
        myParam     = netLearn.Parameter(i);
        myVal       = netLearn.Value{i};
        mySize      = size(extractdata(myVal));

        % Cari di ResNet-50 layer dengan nama sama dan ukuran sama
        matchIdx = find(strcmp(rnNames, myLayerName) & ...
                        strcmp(rnLearn.Parameter, myParam));

        if ~isempty(matchIdx)
            rnVal  = rnLearn.Value{matchIdx(1)};
            rnSize = size(extractdata(rnVal));
            if isequal(mySize, rnSize)
                net.Learnables.Value{i} = rnVal;
                numTransferred = numTransferred + 1;
            else
                numSkipped = numSkipped + 1;
            end
        end
    end

    fprintf('  ✓ Weights transferred: %d parameter groups\n', numTransferred);
    fprintf('  ✗ Skipped (ukuran beda / layer baru): %d\n\n', numSkipped);

    if numTransferred == 0
        fprintf('  ⚠ Tidak ada weight yang cocok — nama layer custom berbeda dari ResNet-50.\n');
        fprintf('    Mencoba transfer by position untuk conv layers...\n');

        % Transfer by position: ambil semua conv weight dari ResNet-50
        % yang ukurannya cocok dengan conv weight di custom net
        rnConvIdx  = find(strcmp(rnLearn.Parameter,'Weights') & ...
                         contains(rnNames,'conv'));
        myConvIdx  = find(strcmp(netLearn.Parameter,'Weights'));

        posTransfer = 0;
        usedRN = false(length(rnConvIdx),1);
        for mi = 1:length(myConvIdx)
            myI   = myConvIdx(mi);
            mySz  = size(extractdata(net.Learnables.Value{myI}));
            % Cari ResNet conv dengan ukuran sama yang belum dipakai
            for ri = 1:length(rnConvIdx)
                if usedRN(ri), continue; end
                rnI  = rnConvIdx(ri);
                rnSz = size(extractdata(rnLearn.Value{rnI}));
                if isequal(mySz, rnSz)
                    net.Learnables.Value{myI} = rnLearn.Value{rnI};
                    usedRN(ri) = true;
                    posTransfer = posTransfer + 1;
                    break;
                end
            end
        end
        fprintf('    Transfer by position: %d conv layers\n\n', posTransfer);
        numTransferred = posTransfer;
    end

catch ME
    fprintf('  ⚠ Transfer pretrained gagal: %s\n', ME.message);
    fprintf('    Lanjut dengan random initialization.\n\n');
end

%% ============================================================
%% STEP 4c: Resume dari Checkpoint (jika ada)
%% ============================================================
RESUME_EPOCH   = 0;    % epoch terakhir yang sudah selesai
RESUME_ITER    = 0;    % iterasi global terakhir
RESUME_ADAMT   = 0;    % Adam step counter terakhir
resumeLossHist = struct('total',[],'rpnCls',[],'rpnReg',[],'detCls',[],'detReg',[]);

if ~isempty(RESUME_FROM)
    if ~isfile(RESUME_FROM)
        error('Checkpoint tidak ditemukan: %s', RESUME_FROM);
    end

    fprintf('[4c] Memuat checkpoint: %s\n', RESUME_FROM);
    cp = load(RESUME_FROM);

    % Restore network weights
    % Checkpoint menyimpan CPU version, kita load dulu lalu pindah ke GPU nanti
    net   = cp.netCPU;
    fcNet = cp.fcNetCPU;

    % Restore optimizer moments
    backbone_m = cp.bb_m_cpu;
    backbone_v = cp.bb_v_cpu;
    fcNet_m    = cp.fc_m_cpu;
    fcNet_v    = cp.fc_v_cpu;

    % Restore training state
    RESUME_EPOCH  = cp.epoch;
    RESUME_ITER   = cp.iterCount;
    RESUME_ADAMT  = cp.adamT;

    % Restore loss history
    resumeLossHist.total  = cp.lossHistory;
    if isfield(cp,'lossHistRPNcls'), resumeLossHist.rpnCls = cp.lossHistRPNcls; end
    if isfield(cp,'lossHistRPNreg'), resumeLossHist.rpnReg = cp.lossHistRPNreg; end
    if isfield(cp,'lossHistDETcls'), resumeLossHist.detCls = cp.lossHistDETcls; end
    if isfield(cp,'lossHistDETreg'), resumeLossHist.detReg = cp.lossHistDETreg; end

    % Hitung total epoch baru
    NUM_EPOCHS = RESUME_EPOCH + EXTRA_EPOCHS;

    fprintf('  ✓ Resume dari epoch %d | iterasi %d | Adam step %d\n', ...
        RESUME_EPOCH, RESUME_ITER, RESUME_ADAMT);
    fprintf('  Lanjut hingga epoch %d (%d epoch tambahan)\n\n', ...
        NUM_EPOCHS, EXTRA_EPOCHS);
    clear cp;
else
    fprintf('[4c] Training baru dari awal (tidak ada checkpoint)\n\n');
end
fprintf('[5/6] Setup GPU + optimizer...\n');

useGPU = canUseGPU();
if useGPU
    fprintf('  ✓ GPU tersedia: %s\n', gpuDevice().Name);
    net   = dlupdate(@gpuArray, net);
    fcNet = dlupdate(@gpuArray, fcNet);
else
    fprintf('  ⚠ GPU tidak tersedia, menggunakan CPU\n');
end

outputNames = {'scoresP2','boxDeltasP2', ...
               'scoresP3','boxDeltasP3', ...
               'scoresP4','boxDeltasP4', ...
               'scoresP5','boxDeltasP5', ...
               'conv_54','conv_55','conv_56','coordConv_11'};

zeroLike = @(t) zeros(size(t),'like',t);

if isempty(RESUME_FROM)
    % Inisialisasi optimizer baru
    backbone_m = net.Learnables;
    backbone_m.Value = cellfun(zeroLike, backbone_m.Value, 'UniformOutput',false);
    backbone_v = net.Learnables;
    backbone_v.Value = cellfun(zeroLike, backbone_v.Value, 'UniformOutput',false);
    fcNet_m = fcNet.Learnables;
    fcNet_m.Value = cellfun(zeroLike, fcNet_m.Value, 'UniformOutput',false);
    fcNet_v = fcNet.Learnables;
    fcNet_v.Value = cellfun(zeroLike, fcNet_v.Value, 'UniformOutput',false);
    fprintf('  Optimizer state: inisialisasi baru\n');
else
    % Pindahkan optimizer moments ke GPU (sudah di-load dari checkpoint)
    if useGPU
        backbone_m = dlupdate(@gpuArray, backbone_m);
        backbone_v = dlupdate(@gpuArray, backbone_v);
        fcNet_m    = dlupdate(@gpuArray, fcNet_m);
        fcNet_v    = dlupdate(@gpuArray, fcNet_v);
    end
    fprintf('  Optimizer state: dilanjutkan dari checkpoint (Adam step=%d)\n', RESUME_ADAMT);
end

if ~exist(CHECKPOINT_DIR,'dir'), mkdir(CHECKPOINT_DIR); end

anchorBoxes = generateAnchorBoxes();
imageSize   = [imageHeight, imageWidth];

% --- Pre-compute anchor grid sekali di CPU, lalu upload ke GPU ---
% Ini menghindari rekomputasi tiap iterasi di dalam computeRPNLoss
featureMapSizes = {
    [floor(imageSize(1)/4),  floor(imageSize(2)/4)],
    [floor(imageSize(1)/8),  floor(imageSize(2)/8)],
    [floor(imageSize(1)/16), floor(imageSize(2)/16)],
    [floor(imageSize(1)/32), floor(imageSize(2)/32)]
};
% Pre-compute anchor grid sesuai arsitektur network:
% scoresP* = 4ch → 2 anchor/loc × 2 class
% boxDeltasP* = 8ch → 2 anchor/loc × 4 delta
% Rasio anchor sesuai jurnal (Wang et al. 2021): [0.25,0.5,1,2,5] -> 5 anchor/lokasi.
aspectRatios224    = [0.25, 0.5, 1.0, 2.0, 5.0];
NUM_ANCHORS_PER_LOC = numel(aspectRatios224);   % = 5

% Base size per level (P2..P5) disetel untuk cakupan dataset @ input 480x480
% (box 480-space sqrt-area ~18..153 px) -> recall@IoU0.5 = 99.8%.
anchorBaseSizes224 = [20, 40, 72, 120];   % nama dipertahankan; nilai utk 480

anchorBoxes2 = cell(1,4);
for lv = 1:4
    bs = anchorBaseSizes224(lv);
    anchors = zeros(NUM_ANCHORS_PER_LOC,2,'single');
    for r = 1:NUM_ANCHORS_PER_LOC
        anchors(r,:) = [bs * sqrt(aspectRatios224(r)), bs / sqrt(aspectRatios224(r))];
    end
    anchorBoxes2{lv} = anchors;
end

% Diagnostic: cetak ukuran anchor dan bandingkan dengan GT boxes
fprintf('  Anchor sizes untuk 224x224:\n');
for lv = 1:4
    fprintf('    P%d: [%.0fx%.0f] [%.0fx%.0f]\n', lv+1, ...
        anchorBoxes2{lv}(1,1), anchorBoxes2{lv}(1,2), ...
        anchorBoxes2{lv}(2,1), anchorBoxes2{lv}(2,2));
end

% Estimasi ukuran GT box setelah resize 1920→224 (skala ≈ 0.117)
scaleEst = 224/1920;
fprintf('  Estimasi ukuran objek setelah resize (skala=%.3f):\n', scaleEst);
allGTsizes = [];
for i = 1:min(50, validCount)
    bb = bboxesAll{i};
    if ~isempty(bb)
        % bbox masih dalam koordinat 224 karena sudah di-resize di Step 2
        sizes = sqrt(bb(:,3) .* bb(:,4));
        allGTsizes = [allGTsizes; sizes]; %#ok<AGROW>
    end
end
if ~isempty(allGTsizes)
    fprintf('    GT box size (sqrt(w*h)): min=%.1f med=%.1f max=%.1f px\n', ...
        min(allGTsizes), median(allGTsizes), max(allGTsizes));
end

allAnchorsCPU = zeros(0,4,'single');
for lv = 1:4
    stride_lv = [imageSize(1)/featureMapSizes{lv}(1), imageSize(2)/featureMapSizes{lv}(2)];
    [ag, ~]   = generateAnchorGrid(anchorBoxes2{lv}, featureMapSizes{lv}, stride_lv);
    allAnchorsCPU = [allAnchorsCPU; ag]; %#ok<AGROW>
end
allAnchorsCPU = clipBoxes(allAnchorsCPU, imageSize);

if useGPU
    allAnchorsGPU = gpuArray(allAnchorsCPU);
else
    allAnchorsGPU = allAnchorsCPU;
end

% Diagnostic IoU: cek berapa anchor yang match GT di sampel pertama
sampleBB = bboxesAll{trainIdx(1)};
if ~isempty(sampleBB)
    sampleXYXY = single([sampleBB(:,1), sampleBB(:,2), ...
                          sampleBB(:,1)+sampleBB(:,3), sampleBB(:,2)+sampleBB(:,4)]);
    % Hitung max IoU setiap anchor terhadap GT
    iouDiag = zeros(size(allAnchorsCPU,1), size(sampleXYXY,1),'single');
    for gi = 1:size(sampleXYXY,1)
        x1=max(allAnchorsCPU(:,1),sampleXYXY(gi,1)); y1=max(allAnchorsCPU(:,2),sampleXYXY(gi,2));
        x2=min(allAnchorsCPU(:,3),sampleXYXY(gi,3)); y2=min(allAnchorsCPU(:,4),sampleXYXY(gi,4));
        inter=max(0,x2-x1).*max(0,y2-y1);
        aA=max(0,allAnchorsCPU(:,3)-allAnchorsCPU(:,1)).*max(0,allAnchorsCPU(:,4)-allAnchorsCPU(:,2));
        aB=max(0,sampleXYXY(gi,3)-sampleXYXY(gi,1))*max(0,sampleXYXY(gi,4)-sampleXYXY(gi,2));
        iouDiag(:,gi) = inter./(aA+aB-inter+1e-6);
    end
    maxIouPerAnchor = max(iouDiag,[],2);
    numPos04 = sum(maxIouPerAnchor >= 0.4);
    numPos05 = sum(maxIouPerAnchor >= 0.5);
    fprintf('  Anchor-GT IoU diagnostic (sampel #1, %d GT boxes):\n', size(sampleXYXY,1));
    fprintf('    Anchor dengan IoU≥0.4: %d / %d (%.1f%%)\n', numPos04, size(allAnchorsCPU,1), 100*numPos04/size(allAnchorsCPU,1));
    fprintf('    Anchor dengan IoU≥0.5: %d / %d (%.1f%%)\n', numPos05, size(allAnchorsCPU,1), 100*numPos05/size(allAnchorsCPU,1));
    fprintf('    Max IoU anchor terbaik: %.3f\n', max(maxIouPerAnchor));
    if max(maxIouPerAnchor) < 0.3
        fprintf('  ⚠ PERINGATAN: max IoU < 0.3 → anchor terlalu besar/kecil vs objek!\n');
        fprintf('    Sesuaikan anchorBaseSizes224 berdasarkan GT box sizes di atas.\n');
    end
end
fprintf('  Total anchors: %d\n\n', size(allAnchorsCPU,1));

%% ============================================================
%% STEP 6: Custom Training Loop — AdamW + LR Warmup
%% ============================================================
fprintf('[6/6] Memulai training...\n\n');
fprintf('==============================================\n');
fprintf('  Epochs    : %d | Batch: %d\n', NUM_EPOCHS, MINI_BATCH);
fprintf('  LR backbone: %.1e | LR head: %.1e\n', BASE_LR_BACKBONE, BASE_LR_HEAD);
fprintf('  Warmup     : %d iter | Drop@ep: %s\n', LR_WARMUP_ITERS, mat2str(LR_DROP_EPOCHS));
fprintf('  Optimizer  : AdamW (β1=%.2f β2=%.3f ε=%.0e)\n', ADAM_BETA1, ADAM_BETA2, ADAM_EPS);
fprintf('  GPU        : %s\n', string(useGPU));
fprintf('==============================================\n\n');

lossHistory        = resumeLossHist.total;
lossHistRPNcls     = resumeLossHist.rpnCls;
lossHistRPNreg     = resumeLossHist.rpnReg;
lossHistDETcls     = resumeLossHist.detCls;
lossHistDETreg     = resumeLossHist.detReg;
iterCount          = RESUME_ITER;
adamT              = RESUME_ADAMT;
trainingStart      = tic;

START_EPOCH = RESUME_EPOCH + 1;
fprintf('  Mulai dari epoch %d hingga epoch %d\n\n', START_EPOCH, NUM_EPOCHS);

% --- SMOKE TEST: verify one gpuComputeGradients step produces gradients ---
if exist('SMOKE_TEST','var') && SMOKE_TEST
    fprintf('\n=== SMOKE TEST: satu langkah gpuComputeGradients ===\n');
    imgT = dlarray(rand(imageHeight,imageWidth,3,1,'single'),'SSCB');
    if useGPU, imgT = gpuArray(imgT); end
    gtT = single([30 40 90 120; 100 60 180 150; 50 150 110 210]);
    gtC = int32([1;4;7]);
    [gB,gF,lrc,lrr,ldc,ldr] = dlfeval(@gpuComputeGradients, net, fcNet, imgT, outputNames, ...
        allAnchorsGPU, anchorBoxes2, featureMapSizes, imageSize, gtT, gtC, ...
        roiPoolSize, numClasses, NUM_RPN_SAMPLES, POS_FRACTION, POS_IOU_THRESH, NEG_IOU_THRESH_HI, ...
        NUM_ROI_SAMPLES, ROI_POS_FRACTION, DET_POS_IOU, DET_NEG_IOU_HI, ...
        RPN_CLS_W, RPN_REG_W, DET_CLS_W, DET_REG_W, useGPU);
    f = @(x) gather(extractdata(x));
    fprintf('  losses: RPNcls=%.4f RPNreg=%.4f DETcls=%.4f DETreg=%.4f\n', f(lrc),f(lrr),f(ldc),f(ldr));
    nzfun = @(g) sum(cellfun(@(v) gather(sum(abs(v(:))))>0, g.Value));
    fprintf('  backbone: %d/%d params nonzero-grad\n', nzfun(gB), numel(gB.Value));
    fprintf('  fcNet   : %d/%d params nonzero-grad\n', nzfun(gF), numel(gF.Value));
    isDcn = contains(string(gB.Layer),'deformConv');
    isCC  = contains(string(gB.Layer),'coordConv');
    ndz = @(m) sum(cellfun(@(v) gather(sum(abs(v(:))))>0, gB.Value(m)));
    fprintf('  DCN nonzero-grad: %d/%d | CoordConv: %d/%d\n', ndz(isDcn),sum(isDcn), ndz(isCC),sum(isCC));
    disp('=== SMOKE TEST DONE ===');
    return;
end

for epoch = START_EPOCH:NUM_EPOCHS

    % LR base setelah warmup (drop schedule)
    lrBackbone = BASE_LR_BACKBONE;
    lrHead     = BASE_LR_HEAD;
    for d = LR_DROP_EPOCHS
        if epoch >= d
            lrBackbone = lrBackbone * LR_DROP_FACTOR;
            lrHead     = lrHead     * LR_DROP_FACTOR;
        end
    end

    % Freeze/unfreeze backbone
    if epoch <= FREEZE_BACKBONE_EPOCHS
        lrBackbone = 0;   % LR=0 efektif membekukan backbone
        if epoch == 1
            fprintf('  Backbone FROZEN untuk %d epoch pertama\n', FREEZE_BACKBONE_EPOCHS);
        end
    elseif epoch == FREEZE_BACKBONE_EPOCHS + 1
        fprintf('  Backbone UNFROZEN mulai epoch %d\n', epoch);
    end

    shuffledTrain = trainIdx(randperm(length(trainIdx)));
    epochLoss  = 0;
    numBatches = 0;

    for bStart = 1:MINI_BATCH:length(shuffledTrain)
        bEnd     = min(bStart+MINI_BATCH-1, length(shuffledTrain));
        batchIdx = shuffledTrain(bStart:bEnd);
        batchSz  = length(batchIdx);

        accBackboneGrad = [];
        accFcGrad       = [];
        totalLoss    = 0;
        sumRPNcls = 0; sumRPNreg = 0; sumDETcls = 0; sumDETreg = 0;

        for bi = 1:batchSz
            sampleIdx  = batchIdx(bi);
            imgPath    = imageFilenames{sampleIdx};
            gtBboxes   = bboxesAll{sampleIdx};
            gtLabelStr = labelsAll{sampleIdx};

            try; img = imread(imgPath); catch; continue; end
            if size(img,3)==1, img = repmat(img,[1 1 3]); end

            imgSingle = single(img) / 255;
            if useGPU
                imgDL = dlarray(gpuArray(imgSingle), 'SSCB');
            else
                imgDL = dlarray(imgSingle, 'SSCB');
            end

            gtBoxesXYXY = single([gtBboxes(:,1), gtBboxes(:,2), ...
                                   gtBboxes(:,1)+gtBboxes(:,3), ...
                                   gtBboxes(:,2)+gtBboxes(:,4)]);
            [~, gtClassIdx] = ismember(gtLabelStr(:), classNames);
            gtClassIdx = int32(gtClassIdx);

            % Single dlfeval — loss components dikembalikan terpisah
            [gradsBackbone, gradsFC, lRPNcls, lRPNreg, lDETcls, lDETreg] = dlfeval( ...
                @gpuComputeGradients, ...
                net, fcNet, imgDL, outputNames, ...
                allAnchorsGPU, anchorBoxes2, featureMapSizes, imageSize, ...
                gtBoxesXYXY, gtClassIdx, ...
                roiPoolSize, numClasses, ...
                NUM_RPN_SAMPLES,  POS_FRACTION, ...
                POS_IOU_THRESH,   NEG_IOU_THRESH_HI, ...
                NUM_ROI_SAMPLES,  ROI_POS_FRACTION, ...
                DET_POS_IOU,      DET_NEG_IOU_HI, ...
                RPN_CLS_W, RPN_REG_W, DET_CLS_W, DET_REG_W, ...
                useGPU);

            rc = double(gather(extractdata(lRPNcls)));
            rr = double(gather(extractdata(lRPNreg)));
            dc = double(gather(extractdata(lDETcls)));
            dr = double(gather(extractdata(lDETreg)));
            sampleLoss = RPN_CLS_W*rc + RPN_REG_W*rr + DET_CLS_W*dc + DET_REG_W*dr;

            totalLoss = totalLoss + sampleLoss;
            sumRPNcls = sumRPNcls + rc;
            sumRPNreg = sumRPNreg + rr;
            sumDETcls = sumDETcls + dc;
            sumDETreg = sumDETreg + dr;

            if isempty(accBackboneGrad)
                accBackboneGrad = gradsBackbone;
                accFcGrad       = gradsFC;
            else
                for gi = 1:height(accBackboneGrad)
                    accBackboneGrad.Value{gi} = accBackboneGrad.Value{gi} + gradsBackbone.Value{gi};
                end
                for gi = 1:height(accFcGrad)
                    accFcGrad.Value{gi} = accFcGrad.Value{gi} + gradsFC.Value{gi};
                end
            end
        end

        if isempty(accBackboneGrad), continue; end

        % Rata-rata gradien
        invB = single(1/batchSz);
        for gi = 1:height(accBackboneGrad)
            accBackboneGrad.Value{gi} = accBackboneGrad.Value{gi} * invB;
        end
        for gi = 1:height(accFcGrad)
            accFcGrad.Value{gi} = accFcGrad.Value{gi} * invB;
        end

        % Gradient clipping
        accBackboneGrad = gpuClipGradients(accBackboneGrad, 10);
        accFcGrad       = gpuClipGradients(accFcGrad, 10);

        % LR warmup linear: 0..LR_WARMUP_ITERS → LR/10..LR
        iterCount = iterCount + 1;
        adamT     = adamT + 1;
        warmupScale = min(1.0, iterCount / max(LR_WARMUP_ITERS,1));
        lrBB = lrBackbone * warmupScale;
        lrHD = lrHead     * warmupScale;

        % AdamW update — backbone dan head dengan LR berbeda
        [net,   backbone_m, backbone_v] = gpuAdamWUpdate( ...
            net,   accBackboneGrad, backbone_m, backbone_v, ...
            lrBB, ADAM_BETA1, ADAM_BETA2, ADAM_EPS, WEIGHT_DECAY, adamT);
        [fcNet, fcNet_m, fcNet_v] = gpuAdamWUpdate( ...
            fcNet, accFcGrad, fcNet_m, fcNet_v, ...
            lrHD, ADAM_BETA1, ADAM_BETA2, ADAM_EPS, WEIGHT_DECAY, adamT);

        avgLoss   = totalLoss / batchSz;
        epochLoss = epochLoss + avgLoss;
        numBatches = numBatches + 1;

        lossHistory(end+1)    = avgLoss;                  %#ok<AGROW>
        lossHistRPNcls(end+1) = sumRPNcls/batchSz;        %#ok<AGROW>
        lossHistRPNreg(end+1) = sumRPNreg/batchSz;        %#ok<AGROW>
        lossHistDETcls(end+1) = sumDETcls/batchSz;        %#ok<AGROW>
        lossHistDETreg(end+1) = sumDETreg/batchSz;        %#ok<AGROW>

        if mod(iterCount, 20) == 0
            fprintf('  Ep%2d Iter%4d | LR_bb=%.1e LR_hd=%.1e | Total=%.3f RPN_cls=%.3f RPN_reg=%.3f DET_cls=%.3f DET_reg=%.3f\n', ...
                epoch, iterCount, lrBB, lrHD, avgLoss, ...
                sumRPNcls/batchSz, sumRPNreg/batchSz, sumDETcls/batchSz, sumDETreg/batchSz);
        end
    end

    avgEpochLoss = epochLoss / max(numBatches,1);
    fprintf('► Epoch %2d | AvgLoss=%.4f | LR_bb=%.1e LR_hd=%.1e\n', ...
        epoch, avgEpochLoss, lrBB, lrHD);

    % Checkpoint — gather ke CPU untuk save
    % Simpan SEMUA state yang diperlukan untuk resume
    if mod(epoch, SAVE_EVERY)==0 || epoch==NUM_EPOCHS
        cpFile = fullfile(CHECKPOINT_DIR, sprintf('checkpoint_epoch%02d.mat', epoch));
        netCPU   = dlupdate(@gather, net);
        fcNetCPU = dlupdate(@gather, fcNet);
        % Gather optimizer moments ke CPU
        bb_m_cpu = dlupdate(@gather, backbone_m);
        bb_v_cpu = dlupdate(@gather, backbone_v);
        fc_m_cpu = dlupdate(@gather, fcNet_m);
        fc_v_cpu = dlupdate(@gather, fcNet_v);
        save(cpFile, ...
            'netCPU', 'fcNetCPU', ...
            'bb_m_cpu', 'bb_v_cpu', 'fc_m_cpu', 'fc_v_cpu', ...
            'classNames', 'epoch', 'iterCount', 'adamT', ...
            'lossHistory', 'lossHistRPNcls', 'lossHistRPNreg', ...
            'lossHistDETcls', 'lossHistDETreg', ...
            '-v7.3');
        fprintf('  ✓ Checkpoint: %s\n', cpFile);
        clear netCPU fcNetCPU bb_m_cpu bb_v_cpu fc_m_cpu fc_v_cpu;
    end
end

trainingTime = toc(trainingStart);
fprintf('\n==============================================\n');
fprintf('  ✓ TRAINING SELESAI: %.1f menit\n', trainingTime/60);
fprintf('==============================================\n\n');

%% ============================================================
%% SIMPAN MODEL FINAL
%% ============================================================
timestamp   = datestr(now,'yyyymmdd_HHMMSS');
modelFile   = sprintf('CustomFasterRCNN_%s.mat', timestamp);
save(modelFile, 'net','fcNet','classNames','anchorBoxes', ...
     'lossHistory','TARGET_SIZE','ORIG_W','ORIG_H', ...
     'roiPoolSize','numClasses');
fprintf('✓ Model final: %s\n\n', modelFile);

%% ============================================================
%% PLOT TRAINING LOSS
%% ============================================================
if ~isempty(lossHistory)
    figure('Name','Training Loss — Per Component');
    subplot(2,1,1);
    lossSmooth = movmean(lossHistory, min(20,length(lossHistory)));
    plot(lossHistory,'Color',[0.75 0.75 0.75],'LineWidth',0.5); hold on;
    plot(lossSmooth,'b-','LineWidth',2); hold off;
    xlabel('Iteration'); ylabel('Total Loss');
    title('Total Loss'); legend('Per-iter','MA-20'); grid on;

    subplot(2,1,2);
    if length(lossHistRPNcls) > 1
        hold on;
        plot(movmean(lossHistRPNcls,min(20,numel(lossHistRPNcls))),'r-','LineWidth',1.5,'DisplayName','RPN cls');
        plot(movmean(lossHistRPNreg,min(20,numel(lossHistRPNreg))),'m-','LineWidth',1.5,'DisplayName','RPN reg');
        plot(movmean(lossHistDETcls,min(20,numel(lossHistDETcls))),'b-','LineWidth',1.5,'DisplayName','DET cls');
        plot(movmean(lossHistDETreg,min(20,numel(lossHistDETreg))),'c-','LineWidth',1.5,'DisplayName','DET reg');
        hold off;
        xlabel('Iteration'); ylabel('Loss');
        title('Loss per komponen'); legend('Location','northeast'); grid on;
    end
    saveas(gcf, sprintf('training_loss_%s.png', timestamp));
end

%% ============================================================
%% VISUALISASI DETEKSI (Validation Set)
%% ============================================================
fprintf('Visualisasi deteksi pada %d sampel validasi...\n', min(6,length(valIdx)));
numViz = min(6, length(valIdx));
vizIdx = valIdx(randperm(length(valIdx), numViz));

figure('Name','Detection Results','Position',[50 50 1400 750]);
for vi = 1:numViz
    si      = vizIdx(vi);
    imgPath = imageFilenames{si};
    try; img = imread(imgPath); catch; continue; end
    if size(img,3)==1, img = repmat(img,[1 1 3]); end

    imgSingle = single(img)/255;
    imgDL     = dlarray(imgSingle,'SSCB');

    % Forward pass
    try
        netOuts = cell(1,numel(outputNames));
        [netOuts{:}] = predict(net, imgDL, 'Outputs', outputNames);
    catch, subplot(2,3,vi); imshow(img); title('Error'); continue; end

    rpnOuts = { extractdata(netOuts{1}), extractdata(netOuts{2}), ...
                extractdata(netOuts{3}), extractdata(netOuts{4}), ...
                extractdata(netOuts{5}), extractdata(netOuts{6}), ...
                extractdata(netOuts{7}), extractdata(netOuts{8}) };

    [proposals, ~] = processRPNOutputs(rpnOuts, anchorBoxes, imageSize, ...
        'NMSThreshold',0.5,'ScoreThreshold',0.3,'MaxProposals',100);

    detBBoxes = []; detScores = []; detLabels = {};
    if ~isempty(proposals)
        fmCell = { squeeze(extractdata(netOuts{9})), ...
                   squeeze(extractdata(netOuts{10})), ...
                   squeeze(extractdata(netOuts{11})), ...
                   squeeze(extractdata(netOuts{12})) };
        roiFeats = roiPooling(fmCell, proposals, imageSize, roiPoolSize);
        [clsScores, bboxPreds] = predict(fcNet, roiFeats, ...
            'Outputs',{'classScores','fc3_bbox'});
        clsData  = extractdata(clsScores);   % [numClasses+1 x N]
        bboxData = extractdata(bboxPreds);   % [numClasses*4 x N]

        [scores, clsIds] = max(clsData(1:numClasses,:), [], 1);
        fgMask = scores > 0.5;

        if any(fgMask)
            fgProps   = proposals(fgMask,:);
            fgScores  = scores(fgMask)';
            fgClsIds  = clsIds(fgMask)';
            fgDeltas  = bboxData(:, fgMask);

            % Decode bbox per class
            for k = 1:sum(fgMask)
                cid     = fgClsIds(k);
                dStart  = (cid-1)*4+1;
                delta_k = double(fgDeltas(dStart:dStart+3, k)');
                prop_k  = double(fgProps(k,:));
                propXYXY= [prop_k(1), prop_k(2), prop_k(1)+prop_k(3), prop_k(2)+prop_k(4)];
                decBox  = applyBBoxDeltas(propXYXY, delta_k);
                decBox  = clipBoxes(decBox, imageSize);
                w = decBox(3)-decBox(1); h = decBox(4)-decBox(2);
                detBBoxes  = [detBBoxes;  decBox(1), decBox(2), w, h]; %#ok<AGROW>
                detScores  = [detScores;  fgScores(k)];                %#ok<AGROW>
                detLabels{end+1} = classNames{cid};                    %#ok<AGROW>
            end

            % NMS final
            if ~isempty(detBBoxes)
                [detBBoxes, detScores] = applyNMS(detBBoxes, detScores, ...
                    'OverlapThreshold',0.5,'MaxProposals',50);
                detLabels = detLabels(1:size(detBBoxes,1));
            end
        end
    end

    subplot(2,3,vi);
    if ~isempty(detBBoxes)
        annots = arrayfun(@(k) sprintf('%s %.0f%%', detLabels{k}, detScores(k)*100), ...
            1:length(detLabels), 'UniformOutput',false);
        imgAnnot = insertObjectAnnotation(img,'rectangle',detBBoxes,annots, ...
            'LineWidth',2,'FontSize',9,'Color','yellow');
        imshow(imgAnnot);
    else
        imshow(img);
    end

    % Ground truth (hijau)
    hold on;
    gtB = bboxesAll{si};
    for b = 1:size(gtB,1)
        rectangle('Position',gtB(b,:),'EdgeColor','g','LineWidth',2,'LineStyle','--');
    end
    hold off;
    title(sprintf('%d det | %d GT', size(detBBoxes,1), size(gtB,1)),'FontSize',9);
end
sgtitle('Deteksi (kuning) vs Ground Truth (hijau putus-putus)','FontSize',11);
saveas(gcf, sprintf('detections_%s.png', timestamp));

fprintf('\n==============================================\n');
fprintf('  SELESAI!\n');
fprintf('  Model  : %s\n', modelFile);
fprintf('  Waktu  : %.1f menit\n', trainingTime/60);
fprintf('==============================================\n');

%% ============================================================
%% LOCAL HELPER FUNCTIONS
%% ============================================================

function [imgOut, bboxOut, validMask] = helperResizePad(img, bboxIn, tSz, oW, oH)
    tH = tSz(1); tW = tSz(2);
    scale = min(tW/oW, tH/oH);
    newW  = round(oW*scale); newH = round(oH*scale);
    padT  = floor((tH-newH)/2); padL = floor((tW-newW)/2);
    imgR  = imresize(img,[newH,newW]);
    imgOut = uint8(zeros(tH,tW,3));
    imgOut(padT+1:padT+newH, padL+1:padL+newW, :) = imgR;
    if isempty(bboxIn)
        bboxOut = zeros(0,4); validMask = false(0,1); return;
    end
    bboxOut = bboxIn;
    bboxOut(:,1) = bboxIn(:,1)*scale + padL;
    bboxOut(:,2) = bboxIn(:,2)*scale + padT;
    bboxOut(:,3) = bboxIn(:,3)*scale;
    bboxOut(:,4) = bboxIn(:,4)*scale;
    bboxOut(:,1) = max(1, bboxOut(:,1));
    bboxOut(:,2) = max(1, bboxOut(:,2));
    bboxOut(:,3) = min(bboxOut(:,3), tW-bboxOut(:,1)+1);
    bboxOut(:,4) = min(bboxOut(:,4), tH-bboxOut(:,2)+1);
    validMask = bboxOut(:,3)>=2 & bboxOut(:,4)>=2;
    bboxOut   = bboxOut(validMask,:);
end

% ================================================================
% gpuComputeGradients — satu forward + satu backward, GPU-first
% Prinsip:
%   - extractdata HANYA untuk NMS (non-diff) dan randperm (no GPU API)
%   - gt boxes TETAP CPU (tidak perlu GPU, hanya untuk scalar ops)
%   - anchor grid sudah di GPU sejak setup, tidak pernah turun
%   - IoU vectorized broadcast di GPU tanpa loop
%   - target delta dihitung vectorized di GPU
%   - label one-hot dibangun vectorized di GPU
% ================================================================
function [gradsBackbone, gradsFC, lRPNcls, lRPNreg, lDETcls, lDETreg] = gpuComputeGradients( ...
        net, fcNet, imgDL, outputNames, ...
        allAnchors, anchorBoxes2, featureMapSizes, imageSize, ...
        gtBoxesCPU, gtClassIdx, ...
        roiPoolSize, numClasses, ...
        numRPNSamples, posFrac, posIoU, negIoU, ...
        numROISamples, roiPosFrac, detPosIoU, detNegIoU, ...
        rpnClsW, rpnRegW, detClsW, detRegW, useGPU)

    %% --- Forward pass ---
    netOuts = cell(1, numel(outputNames));
    [netOuts{:}] = forward(net, imgDL, 'Outputs', outputNames);

    scoresP2=netOuts{1}; boxDeltasP2=netOuts{2};
    scoresP3=netOuts{3}; boxDeltasP3=netOuts{4};
    scoresP4=netOuts{5}; boxDeltasP4=netOuts{6};
    scoresP5=netOuts{7}; boxDeltasP5=netOuts{8};
    fmP2=netOuts{9}; fmP3=netOuts{10}; fmP4=netOuts{11}; fmP5=netOuts{12};

    %% --- Reshape RPN outputs ---
    % scoresP* : [H x W x 4 x 1]  →  flat [H*W*2 x 2]
    % boxDeltas*: [H x W x 8 x 1]  →  flat [H*W*2 x 4]
    rpnScoresDL = cell(1,4);
    rpnDeltasDL = cell(1,4);
    rpnScoresCPU = cell(1,4);   % untuk NMS (CPU, non-diff)
    rpnDeltasCPU = cell(1,4);

    rpnRaw = {scoresP2,boxDeltasP2,scoresP3,boxDeltasP3, ...
              scoresP4,boxDeltasP4,scoresP5,boxDeltasP5};
    for lv = 1:4
        sc = rpnRaw{(lv-1)*2+1};   % [H x W x 4 x 1] dlarray (2 anchors x 2 cls)
        bd = rpnRaw{(lv-1)*2+2};   % [H x W x 8 x 1] dlarray (2 anchors x 4 deltas)

        % --- DIFFERENTIABLE reshape (kept TRACED), ANCHOR-MAJOR to match the
        %     anchor-major rows of generateAnchorGrid. Channel layout per level:
        %     scores [a1c0 a1c1 a2c0 a2c1], deltas [a1(1:4) a2(5:8)]. Stacking each
        %     anchor's HxW locations (column-major, row/h fastest) reproduces the
        %     anchor grid order. (The old code extractdata'd here -> RPN loss got
        %     no gradient; and used a location-major interleaved order that did NOT
        %     match the anchors.)
        scU = stripdims(sc); if ndims(scU)==4, scU = scU(:,:,:,1); end   % [H,W,4]
        bdU = stripdims(bd); if ndims(bdU)==4, bdU = bdU(:,:,:,1); end   % [H,W,8]
        nA  = size(scU,3)/2;
        scParts = cell(1,nA); bdParts = cell(1,nA);
        for a = 1:nA
            scParts{a} = reshape(scU(:,:,(a-1)*2 + (1:2)), [], 2);   % [H*W,2]
            bdParts{a} = reshape(bdU(:,:,(a-1)*4 + (1:4)), [], 4);   % [H*W,4]
        end
        rpnScoresDL{lv} = vertcat(scParts{:});   % [nA*H*W,2] traced, anchor-major
        rpnDeltasDL{lv} = vertcat(bdParts{:});   % [nA*H*W,4] traced

        % Detached CPU copies (HxWxC maps) ONLY for the NMS proposal path.
        rpnScoresCPU{lv} = gather(extractdata(scU));
        rpnDeltasCPU{lv} = gather(extractdata(bdU));
    end

    allScoresDL = vertcat(rpnScoresDL{:});   % [TotalAnc x 2] dlarray GPU
    allDeltasDL = vertcat(rpnDeltasDL{:});   % [TotalAnc x 4] dlarray GPU

    %% --- Proposal generation (NMS, CPU) ---
    rpnForNMS = {rpnScoresCPU{1}, rpnDeltasCPU{1}, ...
                 rpnScoresCPU{2}, rpnDeltasCPU{2}, ...
                 rpnScoresCPU{3}, rpnDeltasCPU{3}, ...
                 rpnScoresCPU{4}, rpnDeltasCPU{4}};
    [proposals, ~] = processRPNOutputs(rpnForNMS, anchorBoxes2, imageSize, ...
        'NMSThreshold',0.7,'ScoreThreshold',0.05,'MaxProposals',300);

    %% --- RPN Loss (GPU) ---
    rpnLoss = gpuRPNLoss(allScoresDL, allDeltasDL, allAnchors, ...
        gtBoxesCPU, numRPNSamples, posFrac, posIoU, negIoU, useGPU);

    %% --- Detection Loss (GPU) ---
    detLoss.cls = dlarray(single(0));
    detLoss.reg = dlarray(single(0));

    if ~isempty(proposals)
        % gpuSampleROIs: IoU di GPU, sampling+targets di CPU, upload GPU
        [roiLabels, bboxTargetsGPU, sampledRoisCPU] = gpuSampleROIs( ...
            single(proposals), gtBoxesCPU, gtClassIdx, ...
            numROISamples, roiPosFrac, detPosIoU, detNegIoU, numClasses, useGPU);

        if ~isempty(sampledRoisCPU)
            % Pass LIVE (traced) feature maps so the detection loss backprops
            % through RoI pooling into the FPN/backbone. roiPooling is now
            % differentiable and strips the labels / picks the single image.
            roiFeats = roiPooling({fmP2, fmP3, fmP4, fmP5}, sampledRoisCPU, imageSize, roiPoolSize);

            [clsScores, bboxPreds] = forward(fcNet, roiFeats, ...
                'Outputs', {'classScores','fc3_bbox'});

            detLoss = gpuDetLoss(clsScores, bboxPreds, ...
                roiLabels, bboxTargetsGPU, numClasses, useGPU);
        end
    end

    totalLoss = rpnClsW*rpnLoss.cls + rpnRegW*rpnLoss.reg + ...
                detClsW*detLoss.cls + detRegW*detLoss.reg;

    % Single multi-output dlgradient call: the autodiff tape is consumed after
    % the first dlgradient, so two separate calls would error. Get both at once.
    [gradsBackbone, gradsFC] = dlgradient(totalLoss, net.Learnables, fcNet.Learnables);

    % Return komponen terpisah untuk monitoring
    lRPNcls = rpnLoss.cls;
    lRPNreg = rpnLoss.reg;
    lDETcls = detLoss.cls;
    lDETreg = detLoss.reg;
end

% ================================================================
% gpuBBoxIoU — fully vectorized broadcast, no loop, runs on GPU
% ================================================================
function iou = gpuBBoxIoU(boxesA, boxesB)
    % boxesA [N x 4], boxesB [M x 4] → iou [N x M]
    x1A=boxesA(:,1); y1A=boxesA(:,2); x2A=boxesA(:,3); y2A=boxesA(:,4);
    x1B=boxesB(:,1)';y1B=boxesB(:,2)';x2B=boxesB(:,3)';y2B=boxesB(:,4)';
    inter = max(0, min(x2A,x2B)-max(x1A,x1B)) .* ...
            max(0, min(y2A,y2B)-max(y1A,y1B));
    areaA = max(0,x2A-x1A).*max(0,y2A-y1A);
    areaB = max(0,x2B-x1B).*max(0,y2B-y1B);
    iou   = inter ./ (areaA + areaB - inter + 1e-6);
end

% ================================================================
% gpuRPNLoss — target delta vectorized di GPU, gather minimal
% ================================================================
function rpnLoss = gpuRPNLoss(allScoresDL, allDeltasDL, allAnchors, ...
        gtBoxesCPU, numSamples, posFrac, posIoU, negIoU, useGPU)

    rpnLoss.cls = dlarray(single(0));
    rpnLoss.reg = dlarray(single(0));
    if isempty(gtBoxesCPU), return; end

    N = min(size(allAnchors,1), size(allScoresDL,1));
    anchors  = allAnchors(1:N,:);       % gpuArray [N x 4]
    scoresDL = allScoresDL(1:N,:);      % dlarray GPU [N x 2]
    deltasDL = allDeltasDL(1:N,:);      % dlarray GPU [N x 4]

    % GT ke GPU sekali
    if useGPU
        gtGPU = gpuArray(single(gtBoxesCPU));
    else
        gtGPU = single(gtBoxesCPU);
    end

    % IoU fully vectorized di GPU [N x M]
    iou = gpuBBoxIoU(anchors, gtGPU);
    [maxIoU, bestGT] = max(iou, [], 2);   % [N x 1] GPU

    % Label assignment di GPU
    labels = gpuArray(zeros(N,1,'single')) - 1;   % -1 = ignore
    labels(maxIoU >= single(posIoU)) = 1;
    labels(maxIoU <  single(negIoU)) = 0;         % strict < negIoU
    % zona negIoU..posIoU tetap -1 (ignore)
    [~, bestAnchor] = max(iou, [], 1);             % tiap GT punya 1 anchor positif
    labels(bestAnchor) = 1;

    % Gather sekali untuk randperm
    labelsCPU = gather(labels);
    posIdx = find(labelsCPU == 1);
    negIdx = find(labelsCPU == 0);

    % Fallback: jika posIdx kosong (tidak ada GT match), ambil top-IoU anchor
    if isempty(posIdx)
        [~, topAnc] = max(gather(maxIoU));
        posIdx   = topAnc;
        labelsCPU(topAnc) = 1;
    end
    numPos = min(round(numSamples*posFrac), length(posIdx));
    numNeg = min(numSamples-numPos, length(negIdx));
    if length(posIdx)>numPos, posIdx=posIdx(randperm(length(posIdx),numPos)); end
    if length(negIdx)>numNeg, negIdx=negIdx(randperm(length(negIdx),numNeg)); end
    sampIdx = [posIdx; negIdx];
    if isempty(sampIdx), return; end

    % Classification loss — sampTarget dibangun di GPU
    isFg = single(labelsCPU(sampIdx)==1);
    if useGPU, isFg = gpuArray(isFg); end
    predFg = sigmoid(scoresDL(sampIdx,2));
    rpnLoss.cls = mean(-isFg.*log(predFg+1e-8) - (1-isFg).*log(1-predFg+1e-8));

    % Regression loss — target delta vectorized di GPU
    if ~isempty(posIdx)
        bestGTcpu = gather(bestGT(posIdx));   % gather sekali, kecil [numPos x 1]

        % Anchors dan GT untuk pos — ambil dari GPU
        posAnch  = anchors(posIdx,:);                     % [numPos x 4] GPU
        gtForPos = gtGPU(bestGTcpu,:);                    % [numPos x 4] GPU

        % Vectorized target delta di GPU
        aW  = posAnch(:,3)-posAnch(:,1)+1e-6;
        aH  = posAnch(:,4)-posAnch(:,2)+1e-6;
        aCX = posAnch(:,1)+aW/2;   aCY = posAnch(:,2)+aH/2;
        gW  = gtForPos(:,3)-gtForPos(:,1)+1e-6;
        gH  = gtForPos(:,4)-gtForPos(:,2)+1e-6;
        gCX = gtForPos(:,1)+gW/2;  gCY = gtForPos(:,2)+gH/2;

        % targDelta tetap di GPU (posAnch sudah GPU)
        targDelta = [(gCX-aCX)./aW, (gCY-aCY)./aH, log(gW./aW), log(gH./aH)];
        diff = deltasDL(posIdx,:) - dlarray(targDelta);
        rpnLoss.reg = mean(gpuSmoothL1(diff(:)));
    end
end

% ================================================================
% gpuSampleROIs — IoU GPU, sampling CPU (randperm), targets vectorized
% ================================================================
function [roiLabels, bboxTargetsGPU, sampledRoisCPU] = gpuSampleROIs( ...
        proposals, gtBoxesCPU, gtClassIdx, ...
        numSamples, posFrac, posIoU, negIoU, nCls, useGPU)

    roiLabels      = [];
    bboxTargetsGPU = [];
    sampledRoisCPU = [];
    if isempty(proposals) || isempty(gtBoxesCPU), return; end

    % IoU di GPU
    if useGPU
        propGPU = gpuArray(proposals);
        gtGPU   = gpuArray(single(gtBoxesCPU));
    else
        propGPU = proposals;
        gtGPU   = single(gtBoxesCPU);
    end

    iou = gpuBBoxIoU(propGPU, gtGPU);          % [N x M] GPU
    [maxIoU, matchIdx] = max(iou, [], 2);       % [N x 1] GPU

    % Gather sekali untuk sampling
    maxIoUcpu   = gather(maxIoU);
    matchIdxCPU = gather(matchIdx);

    % posIdx: IoU >= posIoU
    % negIdx: IoU < negIoU  (zona ignore negIoU..posIoU DIBUANG)
    posIdx = find(maxIoUcpu >= posIoU);
    negIdx = find(maxIoUcpu <  negIoU);   % strict < negIoU, bukan < posIoU

    % Jika tidak ada positive sama sekali, ambil anchor dengan IoU tertinggi
    % sebagai fallback agar ada signal untuk regression
    if isempty(posIdx)
        [~, bestMatch] = max(maxIoUcpu);
        posIdx = bestMatch;
    end

    numPos = min(max(round(numSamples*posFrac),1), length(posIdx));
    numNeg = min(numSamples-numPos, length(negIdx));
    if length(posIdx)>numPos, posIdx=posIdx(randperm(length(posIdx),numPos)); end
    if length(negIdx)>numNeg, negIdx=negIdx(randperm(length(negIdx),numNeg)); end
    sampledIdx = [posIdx; negIdx];
    if isempty(sampledIdx), return; end

    sampledRoisCPU = proposals(sampledIdx,:);   % CPU, untuk roiPooling

    % Labels: fg=class idx, bg=nCls+1
    roiLabels = int32(ones(length(sampledIdx),1) * (nCls+1));
    % Vectorized assignment untuk fg
    fgGTClass = int32(gtClassIdx(matchIdxCPU(posIdx)));
    roiLabels(1:numPos) = fgGTClass;

    % BBox targets — vectorized di CPU, upload GPU sekali
    bboxTargetsCPU = zeros(length(sampledIdx), nCls*4, 'single');
    if numPos > 0
        gt  = gtBoxesCPU(matchIdxCPU(posIdx),:);    % [numPos x 4]
        roi = sampledRoisCPU(1:numPos,:);            % [numPos x 4]
        roiW = roi(:,3)-roi(:,1)+1e-6;  roiH = roi(:,4)-roi(:,2)+1e-6;
        roiCX= roi(:,1)+roiW/2;         roiCY= roi(:,2)+roiH/2;
        gtW  = gt(:,3)-gt(:,1)+1e-6;    gtH  = gt(:,4)-gt(:,2)+1e-6;
        gtCX = gt(:,1)+gtW/2;           gtCY = gt(:,2)+gtH/2;
        dx = (gtCX-roiCX)./roiW;
        dy = (gtCY-roiCY)./roiH;
        dw = log(gtW./roiW);
        dh = log(gtH./roiH);
        % Scatter vectorized: tiap pos ROI → kolom sesuai class
        cids   = double(fgGTClass);
        dStarts= (cids-1)*4 + 1;
        for i = 1:numPos
            bboxTargetsCPU(i, dStarts(i):dStarts(i)+3) = ...
                [dx(i), dy(i), dw(i), dh(i)];
        end
    end

    if useGPU
        bboxTargetsGPU = gpuArray(bboxTargetsCPU);
    else
        bboxTargetsGPU = bboxTargetsCPU;
    end
end

% ================================================================
% gpuDetLoss — one-hot matrix vectorized di GPU
% ================================================================
function detLoss = gpuDetLoss(clsScores, bboxPreds, roiLabels, bboxTargets, nCls, useGPU)
    detLoss.cls = dlarray(single(0));
    detLoss.reg = dlarray(single(0));
    if isempty(roiLabels), return; end

    N = length(roiLabels);

    % One-hot label matrix — vectorized, GPU
    if useGPU
        labelMat = gpuArray(zeros(nCls+1, N, 'single'));
    else
        labelMat = zeros(nCls+1, N, 'single');
    end
    linIdx = sub2ind([nCls+1, N], double(roiLabels(:)'), 1:N);
    labelMat(linIdx) = 1;

    detLoss.cls = mean(-sum(dlarray(labelMat) .* log(clsScores+1e-8), 1));

    fgIdx = find(roiLabels <= int32(nCls));
    if ~isempty(fgIdx)
        predReg   = bboxPreds(:, fgIdx);
        targetReg = dlarray(bboxTargets(fgIdx,:)');   % bboxTargets sudah GPU
        diff = predReg - targetReg;
        detLoss.reg = mean(gpuSmoothL1(diff(:)));
    end
end

function loss = gpuSmoothL1(x)
    absx = abs(x);
    loss = (absx < 1).*(0.5*x.^2) + (absx >= 1).*(absx - 0.5);
end

% ================================================================
% gpuClipGradients — norm sepenuhnya di GPU, tidak ada gather/extractdata
% ================================================================
function grads = gpuClipGradients(grads, maxNorm)
    % Akumulasi squared norm di GPU via dlarray
    totalNormSq = dlarray(single(0));
    for i = 1:height(grads)
        g = grads.Value{i};
        totalNormSq = totalNormSq + sum(g(:).^2);
    end
    totalNorm = sqrt(totalNormSq);
    clipScale = single(maxNorm) ./ (extractdata(totalNorm) + 1e-8);
    if clipScale < 1
        for i = 1:height(grads)
            grads.Value{i} = grads.Value{i} * single(clipScale);
        end
    end
end

% ================================================================
% gpuAdamWUpdate — AdamW optimizer, semua operasi di GPU
% AdamW = Adam + weight decay langsung pada bobot (bukan gradient)
% ================================================================
function [net, m, v] = gpuAdamWUpdate(net, grads, m, v, lr, b1, b2, eps, wd, t)
    b1s = single(b1);  b2s = single(b2);
    lrs = single(lr);  eps = single(eps);  wds = single(wd);
    % Bias correction
    bc1 = single(1 - b1^t);
    bc2 = single(1 - b2^t);
    lrHat = lrs * sqrt(bc2) / bc1;
    for i = 1:height(grads)
        g  = grads.Value{i};
        mi = b1s .* m.Value{i} + (1-b1s) .* g;
        vi = b2s .* v.Value{i} + (1-b2s) .* (g .* g);
        m.Value{i} = mi;
        v.Value{i} = vi;
        % AdamW: weight decay langsung, tidak lewat gradient
        net.Learnables.Value{i} = net.Learnables.Value{i} .* (1 - lrs*wds) ...
                                  - lrHat .* mi ./ (sqrt(vi) + eps);
    end
end