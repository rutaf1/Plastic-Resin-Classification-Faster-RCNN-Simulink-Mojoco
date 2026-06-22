function roiFeatures = roiPooling(featureMaps, proposals, imageSize, poolSize)
% roiPooling - ROI Align GPU-batched untuk Faster R-CNN
%
% PERUBAHAN vs versi lama:
%   - Loop per-proposal DIHAPUS, diganti fully-vectorized batch ops
%   - Semua sampling coords dihitung sekaligus di GPU [N*P*P x 1]
%   - Gather/scatter menggunakan indexing matrix, bukan loop
%   - FPN level assignment di-batch via arrayfun → logical mask
%   - Output tetap dlarray 'SSCB' untuk kompatibilitas downstream
%
% Input:
%   featureMaps - Cell {fmP2,fmP3,fmP4,fmP5}, tiap [H x W x C] atau [H x W x C x 1]
%   proposals   - [N x 4] [xmin, ymin, xmax, ymax]
%   imageSize   - [height, width]
%   poolSize    - ukuran output (default: 7)
%
% Output:
%   roiFeatures - dlarray [poolSize x poolSize x C x N] format 'SSCB'

    if nargin < 4, poolSize = 7; end

    numProposals = size(proposals, 1);

    %% Persiapkan feature maps → plain gpuArray/single [H x W x C]
    fms   = cell(1, 4);
    isGPU = false;
    anyDL = false;
    for k = 1:4
        fm = featureMaps{k};
        % DIFFERENTIABLE: keep dlarray feature maps as traced dlarrays (strip the
        % 'SSCB' labels so manual reshape/index works). Only the bilinear weights
        % are constant; gradients flow through the gathered feature values back to
        % the backbone/FPN. (The previous version extractdata'd here, detaching it.)
        if isa(fm, 'dlarray')
            anyDL = true;
            if isa(extractdata(fm), 'gpuArray'), isGPU = true; end
            fm = stripdims(fm);
        else
            if isa(fm, 'gpuArray'), isGPU = true; end
            if ~isa(fm, 'single'), fm = single(fm); end
        end
        if ndims(fm) == 4, fm = fm(:,:,:,1); end   % single image
        fms{k} = fm;
    end

    C      = size(fms{1}, 3);
    P2     = poolSize * poolSize;   % jumlah grid points per ROI

    roiOut = zeros(poolSize, poolSize, C, numProposals, 'single');
    if isGPU, roiOut = gpuArray(roiOut); end
    if anyDL, roiOut = dlarray(roiOut); end   % enable differentiable scatter below

    if numProposals == 0
        roiFeatures = dlarray(roiOut, 'SSCB');
        return;
    end

    %% FPN level assignment — sekaligus untuk semua proposals [N x 1]
    canonicalSize = sqrt(double(imageSize(1)) * double(imageSize(2))) / 4;
    propW = max(proposals(:,3) - proposals(:,1), 1);
    propH = max(proposals(:,4) - proposals(:,2), 1);
    roiArea = sqrt(double(propW) .* double(propH));

    levels = floor(4 + log2(roiArea / canonicalSize + 1e-6));
    levels = max(1, min(4, levels));   % [N x 1], nilai 1..4

    fmH = cellfun(@(f) size(f,1), fms);
    fmW = cellfun(@(f) size(f,2), fms);

    %% Pre-build meshgrid untuk poolSize grid [P2 x 1]
    [PW_grid, PH_grid] = meshgrid(1:poolSize, 1:poolSize);
    PH_idx = PH_grid(:);   % [P2 x 1]
    PW_idx = PW_grid(:);   % [P2 x 1]

    %% Proses per level — bukan per proposal
    %  Satu GPU kernel launch per level (max 4), bukan per proposal (max 2000)
    for lv = 1:4
        roiIdx = find(levels == lv);
        if isempty(roiIdx), continue; end

        nRoi = numel(roiIdx);
        fm   = fms{lv};
        curH = fmH(lv);
        curW = fmW(lv);

        %% Hitung sampling coordinates: [P2 x nRoi] secara vectorized
        xmin = double(proposals(roiIdx, 1))';   % [1 x nRoi]
        ymin = double(proposals(roiIdx, 2))';
        xmax = double(proposals(roiIdx, 3))';
        ymax = double(proposals(roiIdx, 4))';

        scaleX = curW / imageSize(2);
        scaleY = curH / imageSize(1);

        xs1 = xmin * scaleX;  xs2 = xmax * scaleX;   % [1 x nRoi]
        ys1 = ymin * scaleY;  ys2 = ymax * scaleY;

        % Titik tengah bin: [poolSize x nRoi]
        t_x = linspace(0, 1, poolSize+1);
        t_x = (t_x(1:end-1) + t_x(2:end)) / 2;   % [1 x poolSize] midpoints

        % xc_mat: [poolSize x nRoi] — kolom = ROI, baris = bin x
        xc_mat = xs1 + t_x' .* (xs2 - xs1);     % broadcast [poolSize x nRoi]
        yc_mat = ys1 + t_x' .* (ys2 - ys1);

        xc_mat = single(max(1, min(xc_mat, curW)));
        yc_mat = single(max(1, min(yc_mat, curH)));

        if isGPU
            xc_mat = gpuArray(xc_mat);
            yc_mat = gpuArray(yc_mat);
        end

        %% Bilinear weights per-bin: [poolSize x nRoi]
        x0 = floor(xc_mat);  x1i = min(x0 + 1, curW);  x0 = max(x0, 1);
        y0 = floor(yc_mat);  y1i = min(y0 + 1, curH);  y0 = max(y0, 1);

        wx1 = xc_mat - floor(xc_mat);  wx0 = 1 - wx1;
        wy1 = yc_mat - floor(yc_mat);  wy0 = 1 - wy1;

        %% Expand ke grid P2: [P2 x nRoi]
        % PW_idx, PH_idx: [P2 x 1] → index ke axis poolSize
        ix0 = x0(PW_idx, :);   ix1 = x1i(PW_idx, :);   % [P2 x nRoi]
        iy0 = y0(PH_idx, :);   iy1 = y1i(PH_idx, :);

        w00 = wy0(PH_idx,:) .* wx0(PW_idx,:);   % [P2 x nRoi]
        w01 = wy0(PH_idx,:) .* wx1(PW_idx,:);
        w10 = wy1(PH_idx,:) .* wx0(PW_idx,:);
        w11 = wy1(PH_idx,:) .* wx1(PW_idx,:);

        %% Linear indices ke fmFlat [curH*curW x C]
        %  idx: [P2 x nRoi]
        idx00 = (ix0-1)*curH + iy0;
        idx01 = (ix1-1)*curH + iy0;
        idx10 = (ix0-1)*curH + iy1;
        idx11 = (ix1-1)*curH + iy1;

        fmFlat = reshape(fm, curH*curW, C);   % [HW x C]

        %% Vectorized gather + weighted sum untuk semua ROI sekaligus
        %  Expand idx ke [P2*nRoi x 1] untuk single indexing
        idx00v = idx00(:);  idx01v = idx01(:);
        idx10v = idx10(:);  idx11v = idx11(:);
        w00v   = w00(:);    w01v   = w01(:);
        w10v   = w10(:);    w11v   = w11(:);

        % [P2*nRoi x C]
        pooledFlat = w00v .* fmFlat(idx00v,:) + w01v .* fmFlat(idx01v,:) + ...
                     w10v .* fmFlat(idx10v,:) + w11v .* fmFlat(idx11v,:);

        % Reshape → [poolSize x poolSize x C x nRoi]
        pooledAll = reshape(pooledFlat, P2, nRoi, C);   % [P2 x nRoi x C]
        pooledAll = permute(reshape(pooledAll, poolSize, poolSize, nRoi, C), [1 2 4 3]);
        % → [poolSize x poolSize x C x nRoi]

        roiOut(:,:,:,roiIdx) = pooledAll;
    end

    roiFeatures = dlarray(roiOut, 'SSCB');
end
