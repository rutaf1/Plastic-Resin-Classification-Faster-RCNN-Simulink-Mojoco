classdef CoordConv2DLayer < nnet.layer.Layer
    % CoordConv2DLayer   2-D convolution layer with coordinate information
    %   OPTIMIZED Version
    %
    %   A CoordConv2D layer augments input with coordinate channels (i, j, 
    %   and optionally radius) before performing convolution. This helps 
    %   networks learn position-dependent transformations more effectively.
    %
    %   Example 1: Basic usage
    %      layer = CoordConv2DLayer([3 3], 64, 'Name', 'coordconv1');
    %
    %   Example 2: With radius channel for rotation-aware features
    %      layer = CoordConv2DLayer([5 5], 256, 'WithRadius', true, ...
    %          'Stride', [2 2], 'Name', 'coordconv_r');
    %
    %   Example 3: In a network
    %      layers = [
    %          imageInputLayer([224 224 3])
    %          CoordConv2DLayer([3 3], 64)
    %          batchNormalizationLayer
    %          reluLayer
    %          CoordConv2DLayer([3 3], 128, 'Stride', [2 2])
    %          reluLayer
    %          globalAveragePooling2dLayer
    %          fullyConnectedLayer(10)
    %          softmaxLayer
    %          classificationLayer];
    %
    %   Reference:
    %      Liu, R., et al. (2018). "An intriguing failing of convolutional 
    %      neural networks and the CoordConv solution." NeurIPS 2018.
    
    properties
        NumFilters      % Number of filters
        FilterSize      % Filter dimensions [height width]
        Stride          % Convolution stride [vertical horizontal]
        Padding         % Input padding ('same' or numeric)
        WithRadius      % Include radius channel (logical)
    end
    
    properties (Learnable)
        Weights         % Layer weights
        Bias            % Layer biases
    end
    
    properties (Access = private)
        CachedCoords    % Cached coordinate channels
        CachedSize      % Cached dimensions [H W N]
    end
    
    methods
        function layer = CoordConv2DLayer(filterSize, numFilters, varargin)
            % CoordConv2DLayer   Construct a CoordConv2D layer
            %
            %   layer = CoordConv2DLayer(filterSize, numFilters) creates a
            %   layer with specified filter size and number of filters.
            %
            %   layer = CoordConv2DLayer(__, Name, Value) specifies options:
            %      'Stride'      - Step size [vertical horizontal]. Default: [1 1]
            %      'Padding'     - 'same' or numeric. Default: 'same'
            %      'WithRadius'  - Include radius channel. Default: false
            %      'Name'        - Layer name. Default: ''
            
            p = inputParser;
            addRequired(p, 'filterSize');
            addRequired(p, 'numFilters');
            addParameter(p, 'Stride', [1 1]);
            addParameter(p, 'Padding', 'same');
            addParameter(p, 'WithRadius', false);
            addParameter(p, 'Name', '');
            parse(p, filterSize, numFilters, varargin{:});
            
            layer.NumFilters = numFilters;
            layer.FilterSize = filterSize;
            layer.Stride = p.Results.Stride;
            layer.Padding = p.Results.Padding;
            layer.WithRadius = p.Results.WithRadius;
            layer.Name = p.Results.Name;
            layer.Type = 'CoordConv 2D';
            
            numCoordChannels = 2 + layer.WithRadius;
            layer.Description = sprintf('CoordConv2D %dx%d, %d filters (+%d coord channels)', ...
                filterSize(1), filterSize(2), numFilters, numCoordChannels);
        end
        
        function layer = initialize(layer, layout)
            % Initialize learnable parameters using Glorot initialization
            if isempty(layer.Weights)
                numExtraChannels = 2 + layer.WithRadius;
                inputChannels = layout.Size(3) + numExtraChannels;
                
                sz = [layer.FilterSize, inputChannels, layer.NumFilters];
                layer.Weights = initializeGlorot(sz);
                layer.Bias = zeros(1, 1, layer.NumFilters, 'single');
            end
        end
        
        function Z = predict(layer, X)
            % Forward propagation with coordinate channel augmentation.
            %
            % DIFFERENTIABLE: X is kept as a traced dlarray end-to-end and the
            % learnables are passed to dlconv directly, so gradients flow to
            % layer.Weights/Bias and back to the input. (The previous version
            % called extractdata and re-wrapped a fresh dlarray, which severed
            % the autodiff graph and made the layer untrainable.)
            % Match output format-ness to the input (custom-layer contract).
            wasFormatted = ~isempty(dims(X));
            if ~wasFormatted, X = dlarray(X, 'SSCB'); end
            H = size(X, 1);
            W = size(X, 2);
            N = size(X, 4);

            % Constant coordinate channels in [-1, 1] (no gradient needed).
            i_vals = linspace(-1, 1, H)';      % H x 1 (row coordinate)
            j_vals = linspace(-1, 1, W);       % 1 x W (column coordinate)
            iCh = repmat(i_vals, 1, W);        % H x W
            jCh = repmat(j_vals, H, 1);        % H x W
            if layer.WithRadius
                r = sqrt(iCh.^2 + jCh.^2);
                r = r ./ max(r(:));
                coords = cat(3, iCh, jCh, r);
            else
                coords = cat(3, iCh, jCh);
            end
            coords = repmat(coords, 1, 1, 1, N);                 % H x W x nCoord x N

            % Match the precision/device of X (constant -> use extractdata only
            % as a prototype; X itself stays traced in the cat below).
            coords = cast(coords, 'like', extractdata(X));
            coords = dlarray(coords, 'SSCB');

            % Concatenate along the channel dim (differentiable w.r.t. X) and
            % convolve with the learnables passed DIRECTLY (kept tracked).
            X_with_coords = cat(3, X, coords);
            % cast(...,'like',X) is DIFFERENTIABLE (unlike extractdata) so it
            % matches input precision/device while keeping gradients to the
            % learnables. Needed when the input dtype differs (e.g. checkLayer).
            Wt = cast(layer.Weights, 'like', X);
            Bt = cast(layer.Bias,    'like', X);
            Z = dlconv(X_with_coords, Wt, Bt, ...
                'Stride', layer.Stride, 'Padding', layer.Padding);
            if ~wasFormatted, Z = stripdims(Z); end
        end
        
        function coords = createCoordChannels(layer, H, W, N, precision)
            % Create coordinate channels (normalized to [-1, 1])
            % Optimized version with vectorization
            
            if nargin < 5
                precision = 'single';
            end
            
            % Create coordinate grids efficiently
            i_vals = cast(linspace(-1, 1, H)', precision);
            j_vals = cast(linspace(-1, 1, W), precision);
            
            % Preallocate output
            if layer.WithRadius
                coords = zeros(H, W, 3, N, precision);
            else
                coords = zeros(H, W, 2, N, precision);
            end
            
            % Create coordinate channels using broadcasting
            % i-coordinate (row) channel
            coords(:, :, 1, :) = repmat(i_vals, [1, W, 1, N]);
            
            % j-coordinate (column) channel
            coords(:, :, 2, :) = repmat(j_vals, [H, 1, 1, N]);
            
            % Add radius channel if requested
            if layer.WithRadius
                [J, I] = meshgrid(j_vals, i_vals);
                radius = sqrt(I.^2 + J.^2);
                radius = radius / max(radius(:)); % Normalize to [0, 1]
                coords(:, :, 3, :) = repmat(radius, [1, 1, 1, N]);
            end
        end
    end
end

function weights = initializeGlorot(sz)
    % Glorot (Xavier) initialization for better convergence
    numIn = prod(sz(1:end-1));
    numOut = sz(end);
    limit = sqrt(6 / (numIn + numOut));
    weights = single((rand(sz) * 2 - 1) * limit);
end
