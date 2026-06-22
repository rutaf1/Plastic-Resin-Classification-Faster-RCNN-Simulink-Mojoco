classdef DeformableConvolution2DLayer < nnet.layer.Layer
    % DeformableConvolution2DLayer   2-D deformable convolution layer (DCNv2)
    %   OPTIMIZED Version
    %
    %   A deformable convolution layer that learns spatial offsets and 
    %   modulation masks to adaptively adjust sampling locations, enabling
    %   geometric transformations and improved feature extraction.
    %
    %   Example 1: Basic usage
    %      layer = DeformableConvolution2DLayer([3 3], 64, 'Name', 'dcn1');
    %
    %   Example 2: With stride and dilation
    %      layer = DeformableConvolution2DLayer([3 3], 128, ...
    %          'Stride', 2, 'DilationFactor', 2, 'Name', 'dcn_dilated');
    %
    %   Example 3: In a network
    %      layers = [
    %          imageInputLayer([224 224 3])
    %          convolution2dLayer(3, 64, 'Padding', 'same')
    %          reluLayer
    %          DeformableConvolution2DLayer([3 3], 128, 'Stride', 2)
    %          reluLayer
    %          globalAveragePooling2dLayer
    %          fullyConnectedLayer(10)
    %          softmaxLayer
    %          classificationLayer];
    %
    %   Reference:
    %      Zhu, X., et al. (2019). "Deformable ConvNets v2: More Deformable,
    %      Better Results." CVPR 2019.
    
    properties
        NumFilters          % Number of output filters
        FilterSize          % Filter dimensions [height width]
        Stride              % Convolution stride [vertical horizontal]
        DilationFactor      % Dilation factor [vertical horizontal]
        PaddingSize         % Padding size [vertical horizontal]
        Im2ColStep          % Batch size for sliced processing
    end
    
    properties (Learnable)
        Weights             % Main convolution weights
        Bias                % Main convolution bias
        OffsetWeights       % Offset prediction weights
        OffsetBias          % Offset prediction bias
        MaskWeights         % Modulation mask weights
        MaskBias            % Modulation mask bias
    end
    
    properties (Access = private)
        BaseGrid            % Cached sampling grid
        KernelOffsets       % Kernel position offsets
        LastOutputSize      % Last output dimensions for caching
        NumChannels         % Number of input channels
        GPUBuffers          % Pre-allocated GPU buffers (reserved)
    end
    
    methods
        function layer = DeformableConvolution2DLayer(filterSize, numFilters, varargin)
            % DeformableConvolution2DLayer   Construct a deformable convolution layer
            %
            %   layer = DeformableConvolution2DLayer(filterSize, numFilters)
            %   creates a DCNv2 layer with specified filter size and number of filters.
            %
            %   layer = DeformableConvolution2DLayer(__, Name, Value) specifies options:
            %      'Stride'          - Convolution stride. Default: 1
            %      'Padding'         - 'same' or 'valid'. Default: 'same'
            %      'DilationFactor'  - Dilation factor. Default: 1
            %      'Im2ColStep'      - Batch slice size. Default: 64
            %      'Name'            - Layer name. Default: ''
            
            p = inputParser;
            addRequired(p, 'filterSize');
            addRequired(p, 'numFilters');
            addParameter(p, 'Stride', 1);
            addParameter(p, 'Padding', 'same');
            addParameter(p, 'DilationFactor', 1);
            addParameter(p, 'Im2ColStep', 64);
            addParameter(p, 'Name', '');
            parse(p, filterSize, numFilters, varargin{:});
            
            layer.NumFilters = numFilters;
            layer.Im2ColStep = p.Results.Im2ColStep;
            
            % Convert scalar to 2D
            if isscalar(filterSize)
                layer.FilterSize = [filterSize, filterSize];
            else
                layer.FilterSize = filterSize;
            end
            
            if isscalar(p.Results.Stride)
                layer.Stride = [p.Results.Stride, p.Results.Stride];
            else
                layer.Stride = p.Results.Stride;
            end
            
            if isscalar(p.Results.DilationFactor)
                layer.DilationFactor = [p.Results.DilationFactor, p.Results.DilationFactor];
            else
                layer.DilationFactor = p.Results.DilationFactor;
            end
            
            % Calculate padding for 'same' mode
            if strcmp(p.Results.Padding, 'same')
                kH = layer.FilterSize(1);
                kW = layer.FilterSize(2);
                dH = layer.DilationFactor(1);
                dW = layer.DilationFactor(2);
                layer.PaddingSize = [floor((kH + (kH-1)*(dH-1))/2), ...
                                    floor((kW + (kW-1)*(dW-1))/2)];
            else
                layer.PaddingSize = [0, 0];
            end
            
            layer.Name = p.Results.Name;
            layer.Type = 'Deformable Convolution 2D';
            layer.Description = sprintf('DCNv2: %dx%d, %d filters', ...
                layer.FilterSize(1), layer.FilterSize(2), numFilters);
            
            layer.LastOutputSize = [0 0 0];
            layer.NumChannels = [];
            layer.GPUBuffers = struct();
        end
        
        function layer = initialize(layer, layout)
            % Initialize learnable parameters
            inputSize = layout.Size;
            
            if numel(inputSize) == 3
                C_in = inputSize(3);
            elseif numel(inputSize) == 4
                C_in = inputSize(3);
            else
                error('Invalid input size format');
            end
            
            layer.NumChannels = C_in;
            kH = layer.FilterSize(1);
            kW = layer.FilterSize(2);
            
            % Initialize main convolution weights
            layer.Weights = initializeGlorot([kH, kW, C_in, layer.NumFilters]);
            layer.Bias = zeros(1, 1, layer.NumFilters, 'single');
            
            % Initialize offset prediction network (2 offsets per kernel position)
            numOffsetChannels = 2 * kH * kW;
            layer.OffsetWeights = initializeGlorot([kH, kW, C_in, numOffsetChannels]) * 0.01;
            layer.OffsetBias = zeros(1, 1, numOffsetChannels, 'single');
            
            % Initialize modulation mask network (1 mask per kernel position)
            numMaskChannels = kH * kW;
            layer.MaskWeights = initializeGlorot([kH, kW, C_in, numMaskChannels]) * 0.01;
            layer.MaskBias = zeros(1, 1, numMaskChannels, 'single');
            
            % Pre-compute kernel base offsets
            [k_j, k_i] = meshgrid(0:kW-1, 0:kH-1);
            layer.KernelOffsets = single([k_i(:) * layer.DilationFactor(1), ...
                                          k_j(:) * layer.DilationFactor(2)]);
        end
        
        function Z = predict(layer, X)
            % Forward propagation with deformable sampling (DIFFERENTIABLE).
            %
            % X, offsets, masks and all learnables are kept as traced dlarrays so
            % dlgradient reaches Weights, OffsetWeights, MaskWeights (and biases)
            % AND propagates to the input. The previous version used extractdata +
            % manual sub2ind indexing and re-wrapped a bare dlarray, which severed
            % autodiff (offset/mask sub-nets never learned; upstream got no signal).

            % dlnetwork may call predict with an unformatted dlarray during size
            % inference; dlconv needs a labeled input. Match output format-ness to
            % the input (custom-layer contract): unformatted in -> unformatted out.
            wasFormatted = ~isempty(dims(X));
            if ~wasFormatted, X = dlarray(X, 'SSCB'); end

            % Prepare convolution arguments for the offset/mask predictors.
            if layer.DilationFactor(1) ~= 1 || layer.DilationFactor(2) ~= 1
                convArgs = {'Stride', layer.Stride, 'Padding', layer.PaddingSize, ...
                            'DilationFactor', layer.DilationFactor};
            else
                convArgs = {'Stride', layer.Stride, 'Padding', layer.PaddingSize};
            end

            % Offsets (2*K channels) and modulation masks (K channels) — learnables
            % passed DIRECTLY so their gradients are tracked.
            % cast(...,'like',X) is differentiable (keeps gradients) and matches
            % the input precision/device (needed when dtype differs, e.g. checkLayer).
            offsets = dlconv(X, cast(layer.OffsetWeights,'like',X), cast(layer.OffsetBias,'like',X), convArgs{:});
            masks   = sigmoid(dlconv(X, cast(layer.MaskWeights,'like',X), cast(layer.MaskBias,'like',X), convArgs{:}));

            Z = deformableSample(layer, X, offsets, masks);   % unformatted
            if wasFormatted, Z = dlarray(Z, 'SSCB'); end
        end

        function Z = deformableSample(layer, X, offsets, masks)
            % Modulated deformable bilinear sampling + convolution, differentiable.
            kH = layer.FilterSize(1);  kW = layer.FilterSize(2);  K = kH * kW;
            H = size(X,1);  W = size(X,2);  C = size(X,3);  N = size(X,4);
            H_out = size(offsets,1);  W_out = size(offsets,2);
            sH = layer.Stride(1);  sW = layer.Stride(2);
            pH = layer.PaddingSize(1);  pW = layer.PaddingSize(2);

            % Manual reshape/indexing below needs unformatted (but still traced)
            % dlarrays; strip the 'SSCB' labels and relabel the result at the end.
            X = stripdims(X);  offsets = stripdims(offsets);  masks = stripdims(masks);

            % Base sampling grid (1-based, unpadded coords) + kernel tap offsets.
            % Constants cast to the input's type/device (so a double input gives a
            % double output -> type-consistent; gradients unaffected).
            [gj, gi] = meshgrid(0:W_out-1, 0:H_out-1);          % H_out x W_out
            baseY = cast(reshape(gi*sH + 1, H_out, W_out, 1, 1), 'like', X);
            baseX = cast(reshape(gj*sW + 1, H_out, W_out, 1, 1), 'like', X);
            kOff  = cast(layer.KernelOffsets, 'like', X);        % K x 2 (dy,dx)
            kY = reshape(kOff(:,1), 1, 1, K, 1);
            kX = reshape(kOff(:,2), 1, 1, K, 1);

            % Split offsets into dy/dx and reshape masks (all traced dlarrays).
            off = reshape(offsets, H_out, W_out, K, 2, N);
            dy  = reshape(off(:,:,:,1,:), H_out, W_out, K, N);
            dx  = reshape(off(:,:,:,2,:), H_out, W_out, K, N);
            m   = reshape(masks, H_out, W_out, K, N);

            % Sampling positions in input coords (subtract padding). Traced via dy/dx.
            sampleY = (baseY + kY - pH) + dy;                    % [H_out,W_out,K,N]
            sampleX = (baseX + kX - pW) + dx;

            % Integer corners (detached); fractional weights stay traced -> grad to offsets.
            sYd = extractdata(sampleY);   sXd = extractdata(sampleX);
            y0 = floor(sYd);   x0 = floor(sXd);
            wy = sampleY - y0; wx = sampleX - x0;                % traced
            valid = cast((sYd>=1) & (sYd<=H) & (sXd>=1) & (sXd<=W), 'like', X);
            y0c = max(1,min(H,y0));   x0c = max(1,min(W,x0));
            y1c = max(1,min(H,y0+1)); x1c = max(1,min(W,x0+1));

            P = H_out * W_out * K;
            Wr = reshape(cast(layer.Weights,'like',X), K*C, layer.NumFilters);  % [K*C, F] (tap fastest)
            Zc = cell(1, N);
            for n = 1:N
                Xn = reshape(X(:,:,:,n), H*W, C);                % [H*W, C] traced
                yv0 = reshape(y0c(:,:,:,n),P,1);  xv0 = reshape(x0c(:,:,:,n),P,1);
                yv1 = reshape(y1c(:,:,:,n),P,1);  xv1 = reshape(x1c(:,:,:,n),P,1);
                i00 = yv0 + (xv0-1)*H;  i01 = yv0 + (xv1-1)*H;
                i10 = yv1 + (xv0-1)*H;  i11 = yv1 + (xv1-1)*H;

                g00 = Xn(i00,:);  g01 = Xn(i01,:);  g10 = Xn(i10,:);  g11 = Xn(i11,:);  % [P,C]

                wyn = reshape(wy(:,:,:,n),P,1);  wxn = reshape(wx(:,:,:,n),P,1);
                vn  = reshape(valid(:,:,:,n),P,1);
                w00 = ((1-wyn).*(1-wxn)).*vn;  w01 = ((1-wyn).*wxn).*vn;
                w10 = (wyn.*(1-wxn)).*vn;        w11 = (wyn.*wxn).*vn;

                sampled = w00.*g00 + w01.*g01 + w10.*g10 + w11.*g11;   % [P,C] traced
                sampled = sampled .* reshape(m(:,:,:,n),P,1);          % DCNv2 modulation

                % [H_out*W_out*K, C] -> [H_out*W_out, K*C] (tap fastest), then GEMM.
                sampled = reshape(sampled, H_out*W_out, K, C);
                sampled = reshape(sampled, H_out*W_out, K*C);
                Zn = sampled * Wr;                                     % [H_out*W_out, F]
                Zc{n} = reshape(Zn, H_out, W_out, layer.NumFilters);
            end

            Z = cat(4, Zc{:});                                         % [H_out,W_out,F,N]
            Z = Z + reshape(cast(layer.Bias,'like',X), 1, 1, layer.NumFilters, 1);
            % Returned UNFORMATTED; predict() re-applies 'SSCB' when appropriate.
        end
        
        function Z = batchedDeformableConv(layer, X, offsets, masks, inputPrecision)
            % Process large batches in slices to reduce memory usage
            X_data = extractdata(X);
            N = size(X_data, 4);
            S = layer.Im2ColStep;
            
            % Pre-allocate output
            offsets_data = extractdata(offsets);
            [H_out, W_out, ~, ~] = size(offsets_data);
            Z_full = zeros(H_out, W_out, layer.NumFilters, N, inputPrecision, 'like', X_data);
            
            % Process in batches
            for i = 1:S:N
                batch_end = min(i+S-1, N);
                batch_idx = i:batch_end;
                
                X_batch = X(:,:,:,batch_idx);
                offsets_batch = offsets(:,:,:,batch_idx);
                masks_batch = masks(:,:,:,batch_idx);
                
                Z_batch = optimizedDeformableConv(layer, X_batch, offsets_batch, ...
                                                  masks_batch, inputPrecision);
                Z_full(:,:,:,batch_idx) = extractdata(Z_batch);
            end
            
            Z = dlarray(Z_full);
        end
        
        function Z = optimizedDeformableConv(layer, X, offsets, masks, inputPrecision)
            % Core deformable convolution with bilinear sampling
            X_data = extractdata(X);
            offsets_data = extractdata(offsets);
            masks_data = extractdata(masks);
            
            [H, W, C, N] = size(X_data);
            [H_out, W_out, ~, ~] = size(offsets_data);
            kH = layer.FilterSize(1);
            kW = layer.FilterSize(2);
            K = kH * kW;
            
            isGPU = isa(X_data, 'gpuArray');
            
            % Pad input
            X_pad = padarray(X_data, [layer.PaddingSize(1), layer.PaddingSize(2), 0, 0], 0, 'both');
            [H_pad, W_pad, ~, ~] = size(X_pad);
            
            % Transfer kernel offsets to correct device and precision
            kernelOff = cast(layer.KernelOffsets, inputPrecision);
            kernelOff = transferToDevice(kernelOff, isGPU);
            
            % Generate or reuse base sampling grid
            currentSize = [H_out, W_out, N];
            needsRegeneration = ~isequal(layer.LastOutputSize, currentSize) || ...
                                isempty(layer.BaseGrid) || ...
                                ~strcmp(class(layer.BaseGrid), inputPrecision);
            
            if isa(layer.BaseGrid, 'gpuArray')
                needsRegeneration = needsRegeneration || ...
                                   ~strcmp(classUnderlying(layer.BaseGrid), inputPrecision);
            end
            
            if needsRegeneration
                [grid_j, grid_i] = meshgrid(0:W_out-1, 0:H_out-1);
                base_grid = cat(3, ...
                    cast(grid_i * layer.Stride(1) + 1, inputPrecision), ...
                    cast(grid_j * layer.Stride(2) + 1, inputPrecision));
                
                base_grid = transferToDevice(base_grid, isGPU);
                layer.BaseGrid = base_grid;
                layer.LastOutputSize = currentSize;
            end
            
            % Reshape offsets and masks
            offsets_data = reshape(offsets_data, [H_out, W_out, K, 2, N]);
            masks_data = reshape(masks_data, [H_out, W_out, K, 1, N]);
            
            % Compute sampling positions: base_grid + kernel_offsets + learned_offsets
            base_5d = reshape(layer.BaseGrid, [H_out, W_out, 1, 2, 1]);
            kernel_5d = reshape(kernelOff, [1, 1, K, 2, 1]);
            sample_pos = base_5d + kernel_5d + offsets_data;
            
            sample_y = sample_pos(:,:,:,1,:);
            sample_x = sample_pos(:,:,:,2,:);
            
            % Flatten for vectorized processing
            total = H_out * W_out * K * N;
            sample_y = reshape(sample_y, [total, 1]);
            sample_x = reshape(sample_x, [total, 1]);
            
            % Bilinear interpolation coordinates
            y0 = floor(sample_y);
            x0 = floor(sample_x);
            wy = sample_y - y0;
            wx = sample_x - x0;
            
            % Clamp to valid range
            y0 = max(1, min(H_pad, y0));
            x0 = max(1, min(W_pad, x0));
            y1 = max(1, min(H_pad, y0 + 1));
            x1 = max(1, min(W_pad, x0 + 1));
            
            % Boundary mask
            valid = (sample_y >= 1) & (sample_y <= H_pad) & ...
                    (sample_x >= 1) & (sample_x <= W_pad);
            
            % Compute interpolation weights
            wy0 = (1 - wy) .* valid;
            wy1 = wy .* valid;
            wx0 = 1 - wx;
            wx1 = wx;
            
            w00 = repmat(wy0 .* wx0, [1, C]);
            w01 = repmat(wy0 .* wx1, [1, C]);
            w10 = repmat(wy1 .* wx0, [1, C]);
            w11 = repmat(wy1 .* wx1, [1, C]);
            
            % Batch indexing
            if N == 1
                batch_idx = ones(total, 1);
            else
                batch_idx = repelem((1:N)', H_out * W_out * K);
            end
            
            % Vectorized gather operation
            X_3d = reshape(X_pad, [H_pad * W_pad, C, N]);
            H_W = H_pad * W_pad;
            
            batch_exp = repmat(batch_idx, [1, C]);
            chan_idx = repmat(1:C, [total, 1]);
            base_off = (batch_exp - 1) * H_W * C + (chan_idx - 1) * H_W;
            
            idx00 = repmat(sub2ind([H_pad, W_pad], y0, x0), [1, C]) + base_off;
            idx01 = repmat(sub2ind([H_pad, W_pad], y0, x1), [1, C]) + base_off;
            idx10 = repmat(sub2ind([H_pad, W_pad], y1, x0), [1, C]) + base_off;
            idx11 = repmat(sub2ind([H_pad, W_pad], y1, x1), [1, C]) + base_off;
            
            % Bilinear interpolation
            sampled = w00 .* X_3d(idx00) + w01 .* X_3d(idx01) + ...
                      w10 .* X_3d(idx10) + w11 .* X_3d(idx11);
            
            % Apply modulation and reshape
            sampled = reshape(sampled, [H_out, W_out, K, C, N]);
            sampled = sampled .* masks_data;  % DCNv2 modulation
            sampled = reshape(sampled, [H_out, W_out, kH, kW, C, N]);
            
            % Cast weights and perform final convolution
            W = cast(layer.Weights, inputPrecision);
            bias = cast(layer.Bias, inputPrecision);
            W = transferToDevice(W, isGPU);
            bias = transferToDevice(bias, isGPU);
            
            % Efficient matrix multiplication
            sampled_2d = reshape(permute(sampled, [1,2,6,3,4,5]), ...
                                [H_out*W_out*N, kH*kW*C]);
            W_2d = reshape(W, [kH*kW*C, layer.NumFilters]);
            
            Z_2d = sampled_2d * W_2d;
            
            % Reshape and add bias
            Z = reshape(Z_2d, [H_out, W_out, N, layer.NumFilters]);
            Z = permute(Z, [1, 2, 4, 3]);
            Z = Z + reshape(bias, [1, 1, layer.NumFilters, 1]);
            
            Z = dlarray(Z);
        end
    end
end

function weights = initializeGlorot(sz)
    % Glorot (Xavier) uniform initialization
    numIn = prod(sz(1:end-1));
    numOut = sz(end);
    limit = sqrt(6 / (numIn + numOut));
    weights = single((rand(sz) * 2 - 1) * limit);
end

function tf = hasFormat(X)
    % Check if dlarray has dimension labels
    try
        fmt = dims(X);
        tf = ~isempty(fmt);
    catch
        tf = false;
    end
end

function data = transferToDevice(data, toGPU)
    % Transfer data between CPU and GPU
    if toGPU && ~isa(data, 'gpuArray')
        data = gpuArray(data);
    elseif ~toGPU && isa(data, 'gpuArray')
        data = gather(data);
    end
end