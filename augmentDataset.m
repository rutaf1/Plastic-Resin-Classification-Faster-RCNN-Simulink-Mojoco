%% =========================================================================
%  AUGMENTASI DATASET - CVAT XML Format  [v2 - Target-Based Balancing]
%  =========================================================================
%  Strategi augmentasi:
%    - Hitung TARGET sampel per kelas = max(classCount) * BALANCE_RATIO
%    - Setiap frame dihitung NUM_AUG berdasarkan seberapa jauh kelas
%      yang ada di frame tersebut dari TARGET-nya
%    - Frame kelas mayoritas: NUM_AUG = 0 (tidak di-aug sama sekali)
%    - Frame kelas minoritas: NUM_AUG besar sesuai kebutuhan (dibatasi cap)
%
%  Augmentasi (COIN FLIP INDEPENDEN - bisa kombinasi):
%    1. Flip Horizontal  (P=0.5)
%    2. Flip Vertikal    (P=0.3)
%    3. Rotasi           (P=0.7)
%    4. Brightness/Contrast (P=0.7)
%    5. Zoom / Scale     (P=0.6)
% =========================================================================

clear; clc; close all;

%% ============================================================
%% KONFIGURASI
%% ============================================================

% --- Path ---
inputFolder    = 'C:\Users\fatur\Documents\Tugas_Akhir\Kode MATLAB\dataset\images';
annotationFile = 'C:\Users\fatur\Documents\Tugas_Akhir\Kode MATLAB\dataset\annotations.xml';
outputFolder   = 'C:\Users\fatur\Documents\Tugas_Akhir\Kode MATLAB\dataset\images_aug';

% --- Target keseimbangan ---
% BALANCE_RATIO: rasio target terhadap kelas terbanyak
%   1.0 = usahakan semua kelas setara dengan kelas terbanyak (penuh)
%   0.7 = target 70% dari kelas terbanyak (lebih realistis)
%   0.5 = target 50% dari kelas terbanyak
BALANCE_RATIO = 0.8;

% --- Batas aug per frame (mencegah satu frame di-aug terlalu ekstrem) ---
MAX_AUG_PER_FRAME = 20;

% --- Batas total gambar aug (0 = tidak ada batas) ---
MAX_AUG_IMAGES = 0;

% --- Resolusi asli ---
ORIG_W = 956;
ORIG_H = 720;

% --- Format nama file ---
frameNameFormat = 'frame_%06d';
frameExtension  = '.png';

% ============================================================
%  Probabilitas augmentasi - COIN FLIP INDEPENDEN
%  (bisa kena beberapa sekaligus)
% ============================================================
P_FLIP_H = 0.5;
P_FLIP_V = 0.3;
P_ROTATE = 0.7;
P_BRIGHT = 0.7;
P_ZOOM   = 0.6;

% Parameter detail
ROT_MIN      = -15;   ROT_MAX      =  15;
BRIGHT_RANGE =  30;   CONTRAST_MIN = 0.8;   CONTRAST_MAX = 1.2;
ZOOM_MIN     = 1.1;   ZOOM_MAX     = 1.4;

fprintf('=== AUGMENTASI DATASET (Target-Based Balancing) ===\n\n');
fprintf('Probabilitas coin flip independen:\n');
fprintf('  Flip H              : %.2f\n', P_FLIP_H);
fprintf('  Flip V              : %.2f\n', P_FLIP_V);
fprintf('  Rotasi              : %.2f\n', P_ROTATE);
fprintf('  Brightness/Contrast : %.2f\n', P_BRIGHT);
fprintf('  Zoom                : %.2f\n', P_ZOOM);
fprintf('  → Satu gambar bisa kena 0–5 augmentasi sekaligus\n\n');

%% ============================================================
%% Persiapan Output
%% ============================================================
if ~exist(outputFolder, 'dir'), mkdir(outputFolder); end

%% ============================================================
%% 1. Baca Anotasi XML CVAT
%% ============================================================
fprintf('Membaca anotasi: %s\n', annotationFile);
if ~isfile(annotationFile)
    error('File anotasi tidak ditemukan: %s', annotationFile);
end

xmlData    = xmlread(annotationFile);
trackNodes = xmlData.getElementsByTagName('track');
numTracks  = trackNodes.getLength;
fprintf('Jumlah track ditemukan: %d\n\n', numTracks);

frameAnnotations = struct('bboxes', {}, 'labels', {});

for t = 0 : numTracks - 1
    track      = trackNodes.item(t);
    trackLabel = strtrim(char(track.getAttribute('label')));
    boxNodes   = track.getElementsByTagName('box');

    for b = 0 : boxNodes.getLength - 1
        box     = boxNodes.item(b);
        if str2double(char(box.getAttribute('outside'))) == 1, continue; end

        frameNum = str2double(char(box.getAttribute('frame')));
        xtl = max(0,      str2double(char(box.getAttribute('xtl'))));
        ytl = max(0,      str2double(char(box.getAttribute('ytl'))));
        xbr = min(ORIG_W, str2double(char(box.getAttribute('xbr'))));
        ybr = min(ORIG_H, str2double(char(box.getAttribute('ybr'))));
        w = xbr - xtl;  h = ybr - ytl;
        if w <= 0 || h <= 0, continue; end

        fIdx = frameNum + 1;
        if fIdx > length(frameAnnotations) || isempty(frameAnnotations(fIdx).bboxes)
            frameAnnotations(fIdx).bboxes = zeros(0,4);
            frameAnnotations(fIdx).labels = {};
        end
        frameAnnotations(fIdx).bboxes(end+1,:) = [xtl, ytl, w, h];
        frameAnnotations(fIdx).labels{end+1}    = trackLabel;
    end
end

annotatedFrames = find(arrayfun(@(s) ~isempty(s.labels), frameAnnotations));
fprintf('Frame teranotasi: %d\n\n', length(annotatedFrames));

%% ============================================================
%% 2. Hitung Distribusi Kelas & TARGET per Kelas
%% ============================================================

% Kumpulkan semua label dari frame asli
allLabelsRaw = {};
for fi0 = 1 : length(annotatedFrames)
    allLabelsRaw = [allLabelsRaw, frameAnnotations(annotatedFrames(fi0)).labels]; %#ok<AGROW>
end
classNames = unique(allLabelsRaw);
numClasses = length(classNames);
classCount = zeros(1, numClasses);
for c = 1:numClasses
    classCount(c) = sum(strcmp(allLabelsRaw, classNames{c}));
end

% TARGET per kelas = kelas terbanyak × BALANCE_RATIO
maxCount   = max(classCount);
targetCount = round(maxCount * BALANCE_RATIO);

% Defisit per kelas = seberapa banyak anotasi tambahan yang dibutuhkan
classDeficit = max(0, targetCount - classCount);

fprintf('Distribusi kelas asli & target (BALANCE_RATIO=%.2f):\n', BALANCE_RATIO);
fprintf('  Target per kelas: %d anotasi\n\n', targetCount);
fprintf('  %-22s  %6s  %6s  %8s\n', 'Kelas', 'Asli', 'Target', 'Defisit');
fprintf('  %s\n', repmat('-', 1, 50));
for c = 1:numClasses
    flag = '';
    if classDeficit(c) == 0, flag = ' ✓ mayoritas'; end
    fprintf('  %-22s  %6d  %6d  %8d%s\n', ...
        classNames{c}, classCount(c), targetCount, classDeficit(c), flag);
end
fprintf('\n');

%% ============================================================
%% 3. Hitung NUM_AUG per Frame (target-based)
%% ============================================================
%
%  Untuk setiap frame, cari kelas yang paling butuh aug (defisit terbesar).
%  Estimasi kontribusi 1 aug dari frame ini = 1 anotasi per kelas yang ada.
%  NUM_AUG = ceil(defisit_kelas_terbesar / kontribusi_per_aug)
%  dibatasi MAX_AUG_PER_FRAME.
%
%  Frame yang semua kelasnya sudah memenuhi target → NUM_AUG = 0.

% Hitung berapa frame yang mengandung tiap kelas (untuk estimasi kontribusi)
framesPerClass = zeros(1, numClasses);
for fi0 = 1:length(annotatedFrames)
    lblsUniq = unique(frameAnnotations(annotatedFrames(fi0)).labels);
    for k = 1:length(lblsUniq)
        cIdx = find(strcmp(classNames, lblsUniq{k}), 1);
        if ~isempty(cIdx)
            framesPerClass(cIdx) = framesPerClass(cIdx) + 1;
        end
    end
end

frameNumAug = zeros(1, length(annotatedFrames));

for fi0 = 1:length(annotatedFrames)
    lbls = frameAnnotations(annotatedFrames(fi0)).labels;
    maxNeeded = 0;

    for k = 1:length(lbls)
        cIdx = find(strcmp(classNames, lbls{k}), 1);
        if isempty(cIdx), continue; end

        if classDeficit(cIdx) > 0 && framesPerClass(cIdx) > 0
            % Estimasi: seberapa banyak aug yang dibutuhkan frame ini
            % agar kelas ini mencapai target
            needed = ceil(classDeficit(cIdx) / framesPerClass(cIdx));
            maxNeeded = max(maxNeeded, needed);
        end
    end

    % Batasi dengan MAX_AUG_PER_FRAME
    frameNumAug(fi0) = min(maxNeeded, MAX_AUG_PER_FRAME);
end

% Ringkasan NUM_AUG
numAugZero    = sum(frameNumAug == 0);
numAugNonZero = sum(frameNumAug > 0);
fprintf('Rencana augmentasi per frame (MAX_AUG_PER_FRAME=%d):\n', MAX_AUG_PER_FRAME);
fprintf('  Frame tidak di-aug (kelas sudah cukup) : %d frame\n', numAugZero);
fprintf('  Frame yang akan di-aug                 : %d frame\n', numAugNonZero);
fprintf('  Rata-rata aug per frame aktif          : %.1f\n', ...
        mean(frameNumAug(frameNumAug > 0)));
fprintf('  Distribusi NUM_AUG:\n');
augValues = unique(frameNumAug);
for v = 1:length(augValues)
    fprintf('    %2dx aug : %d frame\n', augValues(v), sum(frameNumAug == augValues(v)));
end

% Proyeksi anotasi hasil aug per kelas
projAug = zeros(1, numClasses);
for fi0 = 1:length(annotatedFrames)
    if frameNumAug(fi0) == 0, continue; end
    lbls = frameAnnotations(annotatedFrames(fi0)).labels;
    for k = 1:length(lbls)
        cIdx = find(strcmp(classNames, lbls{k}), 1);
        if ~isempty(cIdx)
            projAug(cIdx) = projAug(cIdx) + frameNumAug(fi0);
        end
    end
end

fprintf('\nProyeksi hasil (sebelum aug dijalankan):\n');
fprintf('  %-22s  %6s  %8s  %8s  %6s\n', 'Kelas','Asli','Proj.Aug','Total','Target');
fprintf('  %s\n', repmat('-', 1, 60));
for c = 1:numClasses
    projTotal = classCount(c) + projAug(c);
    gap = projTotal - targetCount;
    gapStr = '';
    if gap >= 0
        gapStr = sprintf(' (+%d)', gap);
    else
        gapStr = sprintf(' (%d)', gap);
    end
    fprintf('  %-22s  %6d  %8d  %8d  %6d%s\n', ...
        classNames{c}, classCount(c), projAug(c), projTotal, targetCount, gapStr);
end
fprintf('\n');

%% ============================================================
%% 4. Proses Augmentasi
%% ============================================================
fprintf('Memulai augmentasi...\n\n');

totalSaved   = 0;
totalFailed  = 0;
countFlipH   = 0;
countFlipV   = 0;
countRotate  = 0;
countBright  = 0;
countZoom    = 0;
countSkipped = 0;

augRecords = struct('filename',{}, 'bboxes',{}, 'labels',{}, 'augDesc',{});

for fi = 1 : length(annotatedFrames)

    if MAX_AUG_IMAGES > 0 && totalSaved >= MAX_AUG_IMAGES
        fprintf('  Batas MAX_AUG_IMAGES (%d) tercapai.\n', MAX_AUG_IMAGES);
        break;
    end

    % Skip frame yang tidak butuh aug
    if frameNumAug(fi) == 0, continue; end

    fIdx     = annotatedFrames(fi);
    frameNum = fIdx - 1;

    imgName = sprintf('%s%s', sprintf(frameNameFormat, frameNum), frameExtension);
    imgPath = fullfile(inputFolder, imgName);

    if ~isfile(imgPath)
        alt = fullfile(inputFolder, sprintf('frame%06d%s', frameNum, frameExtension));
        if isfile(alt)
            imgPath = alt;
            imgName = sprintf('frame%06d%s', frameNum, frameExtension);
        else
            totalFailed = totalFailed + 1;
            continue;
        end
    end

    try
        img = imread(imgPath);
        if size(img,3)==1, img = repmat(img,[1 1 3]); end
        if size(img,3)==4, img = img(:,:,1:3); end
    catch
        totalFailed = totalFailed + 1;
        continue;
    end

    rawBboxes = frameAnnotations(fIdx).bboxes;
    rawLabels = frameAnnotations(fIdx).labels;

    % Salin gambar asli ke output
    dstOrig = fullfile(outputFolder, imgName);
    if ~isfile(dstOrig), imwrite(img, dstOrig); end

    [~, baseName, ~] = fileparts(imgName);

    for a = 1 : frameNumAug(fi)

        if MAX_AUG_IMAGES > 0 && totalSaved >= MAX_AUG_IMAGES, break; end

        augImg        = img;
        augBboxes     = rawBboxes;
        augLabels_cur = rawLabels;
        imgW = ORIG_W;  imgH = ORIG_H;
        augDesc = {};

        %% (A) FLIP HORIZONTAL
        if rand() < P_FLIP_H
            augImg = fliplr(augImg);
            if ~isempty(augBboxes)
                augBboxes(:,1) = imgW - (augBboxes(:,1) + augBboxes(:,3));
            end
            countFlipH = countFlipH + 1;
            augDesc{end+1} = 'FlipH';
        end

        %% (B) FLIP VERTIKAL
        if rand() < P_FLIP_V
            augImg = flipud(augImg);
            if ~isempty(augBboxes)
                augBboxes(:,2) = imgH - (augBboxes(:,2) + augBboxes(:,4));
            end
            countFlipV = countFlipV + 1;
            augDesc{end+1} = 'FlipV';
        end

        %% (C) ROTASI
        if rand() < P_ROTATE
            angle  = ROT_MIN + rand()*(ROT_MAX - ROT_MIN);
            augImg = imrotate(augImg, angle, 'bilinear', 'crop');
            countRotate = countRotate + 1;
            augDesc{end+1} = sprintf('Rot(%.1f°)', angle);

            if ~isempty(augBboxes)
                cx = imgW/2;  cy = imgH/2;
                theta = deg2rad(-angle);
                cosT = cos(theta);  sinT = sin(theta);

                newBboxes = zeros(size(augBboxes));
                keepBox   = true(size(augBboxes,1), 1);

                for k = 1:size(augBboxes,1)
                    corners = [augBboxes(k,1),                   augBboxes(k,2);
                               augBboxes(k,1)+augBboxes(k,3),    augBboxes(k,2);
                               augBboxes(k,1)+augBboxes(k,3),    augBboxes(k,2)+augBboxes(k,4);
                               augBboxes(k,1),                   augBboxes(k,2)+augBboxes(k,4)];
                    rotC = zeros(4,2);
                    for p = 1:4
                        dx = corners(p,1)-cx;  dy = corners(p,2)-cy;
                        rotC(p,1) = cx + cosT*dx - sinT*dy;
                        rotC(p,2) = cy + sinT*dx + cosT*dy;
                    end
                    x1 = max(0,    min(rotC(:,1)));  y1 = max(0,    min(rotC(:,2)));
                    x2 = min(imgW, max(rotC(:,1)));  y2 = min(imgH, max(rotC(:,2)));
                    nw = x2-x1;  nh = y2-y1;
                    if nw<4 || nh<4
                        keepBox(k) = false;
                    else
                        newBboxes(k,:) = [x1, y1, nw, nh];
                    end
                end
                augBboxes     = newBboxes(keepBox,:);
                augLabels_cur = augLabels_cur(keepBox);
            end
        end

        %% (D) BRIGHTNESS & CONTRAST
        if rand() < P_BRIGHT
            alpha  = CONTRAST_MIN + rand()*(CONTRAST_MAX - CONTRAST_MIN);
            beta   = (rand()*2-1)*BRIGHT_RANGE;
            augImg = uint8(min(255, max(0, double(augImg)*alpha + beta)));
            countBright = countBright + 1;
            augDesc{end+1} = sprintf('Bright(C:%.2f B:%+.0f)', alpha, beta);
        end

        %% (E) ZOOM / SCALE
        if rand() < P_ZOOM
            zoomFactor = ZOOM_MIN + rand()*(ZOOM_MAX - ZOOM_MIN);
            cropW = round(imgW/zoomFactor);
            cropH = round(imgH/zoomFactor);
            x0 = max(1, round((imgW-cropW)/2));
            y0 = max(1, round((imgH-cropH)/2));

            % Pastikan crop tidak melebihi batas gambar
            x0end = min(size(augImg,2), x0+cropW-1);
            y0end = min(size(augImg,1), y0+cropH-1);
            cropImg = augImg(y0:y0end, x0:x0end, :);
            augImg  = imresize(cropImg, [imgH, imgW]);
            countZoom = countZoom + 1;
            augDesc{end+1} = sprintf('Zoom(%.2fx)', zoomFactor);

            if ~isempty(augBboxes)
                newBboxes2 = zeros(size(augBboxes));
                keepBox2   = true(size(augBboxes,1), 1);
                for k = 1:size(augBboxes,1)
                    xtl_z = (augBboxes(k,1) - x0) * zoomFactor;
                    ytl_z = (augBboxes(k,2) - y0) * zoomFactor;
                    w_z   =  augBboxes(k,3) * zoomFactor;
                    h_z   =  augBboxes(k,4) * zoomFactor;
                    xtl_z = max(0, xtl_z);
                    ytl_z = max(0, ytl_z);
                    w_z   = min(w_z, imgW - xtl_z);
                    h_z   = min(h_z, imgH - ytl_z);
                    if w_z<4 || h_z<4
                        keepBox2(k) = false;
                    else
                        newBboxes2(k,:) = [xtl_z, ytl_z, w_z, h_z];
                    end
                end
                augBboxes     = newBboxes2(keepBox2,:);
                augLabels_cur = augLabels_cur(keepBox2);
            end
        end

        % Skip jika tidak ada aug yang terjadi
        if isempty(augDesc)
            countSkipped = countSkipped + 1;
            continue;
        end

        % Skip jika semua bbox hilang
        if isempty(augBboxes)
            countSkipped = countSkipped + 1;
            continue;
        end

        % Simpan
        augName    = sprintf('%s_aug%d%s', baseName, a, frameExtension);
        augOutPath = fullfile(outputFolder, augName);
        try
            imwrite(augImg, augOutPath);
            totalSaved = totalSaved + 1;
            recIdx = length(augRecords)+1;
            augRecords(recIdx).filename = augOutPath;
            augRecords(recIdx).bboxes   = augBboxes;
            augRecords(recIdx).labels   = augLabels_cur;
            augRecords(recIdx).augDesc  = augDesc;
        catch
            totalFailed = totalFailed + 1;
        end
    end

    if mod(fi, 50) == 0
        fprintf('  Progress: %d / %d frame diproses...\n', fi, length(annotatedFrames));
    end
end

%% ============================================================
%% 5. Summary
%% ============================================================

% Hitung anotasi aktual dari hasil aug
allLabelsAug = {};
for r = 1:length(augRecords)
    if ~isempty(augRecords(r).labels)
        allLabelsAug = [allLabelsAug, augRecords(r).labels{:}]; %#ok<AGROW>
    end
end
countAug   = zeros(1, numClasses);
for c = 1:numClasses
    countAug(c) = sum(strcmp(allLabelsAug, classNames{c}));
end
countTotal = classCount + countAug;

fprintf('\n');
fprintf('╔══════════════════════════════════════════════════════╗\n');
fprintf('║       SUMMARY AUGMENTASI (Target-Based)              ║\n');
fprintf('╠══════════════════════════════════════════════════════╣\n');
fprintf('║  STATISTIK GAMBAR                                    ║\n');
fprintf('╠══════════════════════════════════════════════════════╣\n');
fprintf('║  Frame teranotasi asli           : %5d             ║\n', length(annotatedFrames));
fprintf('║  Frame tidak di-aug (mayoritas)  : %5d             ║\n', numAugZero);
fprintf('║  Frame yang di-aug               : %5d             ║\n', numAugNonZero);
fprintf('║  MAX_AUG_PER_FRAME               : %5d             ║\n', MAX_AUG_PER_FRAME);
if MAX_AUG_IMAGES == 0
    fprintf('║  Batas total aug                 : tidak ada batas  ║\n');
else
    fprintf('║  Batas total aug                 : %5d gambar      ║\n', MAX_AUG_IMAGES);
end
fprintf('║  Gambar aug berhasil disimpan    : %5d             ║\n', totalSaved);
fprintf('║  Dilewati (no aug / bbox hilang) : %5d             ║\n', countSkipped);
fprintf('║  Gagal simpan                    : %5d             ║\n', totalFailed);
fprintf('║  Total gambar di folder output   : %5d             ║\n', length(annotatedFrames)+totalSaved);
fprintf('╠══════════════════════════════════════════════════════╣\n');
fprintf('║  COIN FLIP INDEPENDEN                                ║\n');
fprintf('╠══════════════════════════════════════════════════════╣\n');
fprintf('║  %-24s : p=%.2f  %5d kali    ║\n', 'Flip Horizontal',       P_FLIP_H, countFlipH);
fprintf('║  %-24s : p=%.2f  %5d kali    ║\n', 'Flip Vertikal',         P_FLIP_V, countFlipV);
fprintf('║  %-24s : p=%.2f  %5d kali    ║\n', 'Rotasi',                P_ROTATE, countRotate);
fprintf('║  %-24s : p=%.2f  %5d kali    ║\n', 'Brightness & Contrast', P_BRIGHT, countBright);
fprintf('║  %-24s : p=%.2f  %5d kali    ║\n', 'Zoom / Scale',          P_ZOOM,   countZoom);
fprintf('╠══════════════════════════════════════════════════════╣\n');
fprintf('║  ANOTASI PER KELAS (vs TARGET)                       ║\n');
fprintf('╠════════════════════╦═══════╦═══════╦═══════╦════════╣\n');
fprintf('║  %-18s  ║  Asli ║  Aug  ║ Total ║ Target ║\n', 'Kelas');
fprintf('╠════════════════════╬═══════╬═══════╬═══════╬════════╣\n');
for c = 1:numClasses
    pct = round(countTotal(c)/targetCount*100);
    fprintf('║  %-18s  ║%6d ║%6d ║%6d ║%5d (%3d%%)║\n', ...
        classNames{c}, classCount(c), countAug(c), countTotal(c), targetCount, pct);
end
fprintf('╠════════════════════╬═══════╬═══════╬═══════╬════════╣\n');
fprintf('║  %-18s  ║%6d ║%6d ║%6d ║        ║\n', ...
    'TOTAL', sum(classCount), sum(countAug), sum(countTotal));
fprintf('╠════════════════════╩═══════╩═══════╩═══════╩════════╣\n');
fprintf('║  Output: %-43s║\n', outputFolder(max(1,end-42):end));
fprintf('╚══════════════════════════════════════════════════════╝\n\n');

%% ============================================================
%% 6. Preview - Grid (tiap gambar dengan judul augmentasi lengkap)
%% ============================================================
if totalSaved > 0 && ~isempty(augRecords)
    fprintf('Menampilkan preview...\n');

    MAX_PREVIEW = 12;
    numPreview  = min(MAX_PREVIEW, length(augRecords));

    % Pilih sample yang representatif: ambil tersebar, bukan hanya awal
    step     = max(1, floor(length(augRecords) / numPreview));
    idxPrev  = 1 : step : length(augRecords);
    idxPrev  = idxPrev(1:min(numPreview, length(idxPrev)));

    nCols = min(4, length(idxPrev));
    nRows = ceil(length(idxPrev) / nCols);
    figW  = 430 * nCols;
    figH  = 360 * nRows;

    figure('Name','Preview Hasil Augmentasi', ...
           'Position',[20 20 figW figH], ...
           'Color',[0.12 0.12 0.12]);

    for i = 1:length(idxPrev)
        rec = augRecords(idxPrev(i));
        ax  = subplot(nRows, nCols, i);

        try
            imgPrev = imread(rec.filename);
        catch
            set(ax,'Color',[0.2 0.2 0.2]);
            text(0.5,0.5,'Gagal baca gambar','HorizontalAlignment','center', ...
                'Color','white','Units','normalized','FontSize',9);
            axis off;
            continue;
        end

        imshow(imgPrev, 'Parent', ax);
        hold(ax,'on');

        % Warna unik per label kelas (bukan per bbox)
        uniqueLabels = unique(rec.labels);
        cmap = lines(length(uniqueLabels));
        colorMap = containers.Map(uniqueLabels, num2cell(cmap,2));

        for k = 1:size(rec.bboxes,1)
            bb  = rec.bboxes(k,:);
            clr = colorMap(rec.labels{k});
            rectangle('Parent',ax,'Position',bb,'EdgeColor',clr,'LineWidth',2);
            text(bb(1)+3, bb(2)+14, rec.labels{k}, ...
                'Parent',ax,'Color','yellow','FontSize',7,'FontWeight','bold', ...
                'BackgroundColor',[0 0 0 0.65],'Interpreter','none');
        end
        hold(ax,'off');

        % Judul: baris 1 = nama file, baris 2 = kombinasi aug
        [~, fn, ext] = fileparts(rec.filename);
        if isempty(rec.augDesc)
            augStr = '(tidak ada aug)';
        else
            augStr = strjoin(rec.augDesc, ' + ');
        end
        titleStr = sprintf('%s\n%s', [fn ext], augStr);
        title(ax, titleStr, ...
              'Interpreter','none','FontSize',7,'FontWeight','bold','Color','white');
        set(ax,'Color',[0.1 0.1 0.1],'XColor','none','YColor','none');
    end

    sgtitle(sprintf('Preview Augmentasi  —  %d sample ditampilkan  (total %d gambar tersimpan)', ...
            length(idxPrev), totalSaved), ...
            'FontSize',10,'FontWeight','bold','Color','white');

    % ---- Figure 2: Bar chart anotasi per kelas ----
    figure('Name','Distribusi Anotasi per Kelas', ...
           'Position',[20+figW+10, 20, 560, 380], ...
           'Color',[0.12 0.12 0.12]);

    x   = 1:numClasses;
    bAug = bar(x, [classCount; countAug; repmat(targetCount,1,numClasses)]', 'grouped');
    bAug(1).FaceColor = [0.27 0.51 0.71];   % biru - asli
    bAug(2).FaceColor = [0.20 0.63 0.17];   % hijau - aug
    bAug(3).FaceColor = [0.84 0.15 0.16];   % merah - target
    bAug(3).FaceAlpha = 0.4;
    bAug(3).EdgeColor = [0.84 0.15 0.16];
    bAug(3).LineWidth = 1.5;

    set(gca,'XTick',x,'XTickLabel',classNames,'XTickLabelRotation',15, ...
            'FontSize',8,'Color',[0.12 0.12 0.12],'XColor','white','YColor','white');
    ylabel('Jumlah anotasi','Color','white');
    title('Anotasi per Kelas: Asli vs Aug vs Target','Color','white','FontWeight','bold');
    legend({'Asli','Augmentasi','Target'},'TextColor','white','Color',[0.2 0.2 0.2], ...
           'Location','northeast');
    grid on; set(gca,'GridColor',[0.35 0.35 0.35]);

    % Nilai di atas bar total (asli+aug)
    for c = 1:numClasses
        text(c, countTotal(c) + max(countTotal)*0.02, ...
             num2str(countTotal(c)), ...
             'HorizontalAlignment','center','Color','white','FontSize',7.5,'FontWeight','bold');
    end

    % ---- Figure 3: Frekuensi tipe aug ----
    figure('Name','Frekuensi Tipe Augmentasi', ...
           'Position',[20+figW+10, 420, 560, 320], ...
           'Color',[0.12 0.12 0.12]);

    augNames  = {'Flip H','Flip V','Rotasi','Bright/C','Zoom'};
    augCounts = [countFlipH, countFlipV, countRotate, countBright, countZoom];
    bh = bar(augCounts,'FaceColor','flat');
    bh.CData = lines(5);
    set(gca,'XTickLabel',augNames,'FontSize',9, ...
            'Color',[0.12 0.12 0.12],'XColor','white','YColor','white');
    ylabel('Jumlah diterapkan','Color','white');
    title('Frekuensi Tiap Tipe Augmentasi','Color','white','FontWeight','bold');
    grid on; set(gca,'GridColor',[0.35 0.35 0.35]);
    for b = 1:length(augCounts)
        text(b, augCounts(b)+max(augCounts)*0.02, num2str(augCounts(b)), ...
             'HorizontalAlignment','center','Color','white','FontWeight','bold','FontSize',9);
    end

    fprintf('  Preview: %d gambar | Bar chart: distribusi & frekuensi aug\n', length(idxPrev));
end

fprintf('\nProgram selesai.\n');