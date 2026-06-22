% saveAnnotatedVideo.m
% Jalankan setelah simulasi selesai untuk menyimpan frame hasil anotasi ke video.
%
% Prasyarat — To Workspace block di Simulink harus dikonfigurasi:
%   Variable name : annotatedFrames
%   Save format   : Array
%   (Limit data points to last: nonaktifkan)
%
% Format data dari To Workspace dengan signal [480x480x3]:
%   annotatedFrames -> [480 x 480 x 3 x N], N = jumlah time step
%
% =========================================================================

%% Konfigurasi
RECORD_FOLDER = 'record';
OUTPUT_FILE   = 'output_annotated.avi';   % ganti ke .mp4 jika diinginkan
FRAME_RATE    = 25;                        % sesuaikan dengan 1/SampleTime blok
QUALITY       = 95;                        % 0-100, hanya berlaku untuk MPEG-4

%% Buat folder record jika belum ada
if ~exist(RECORD_FOLDER, 'dir')
    mkdir(RECORD_FOLDER);
    fprintf('Folder dibuat: %s\n', RECORD_FOLDER);
end
OUTPUT_PATH = fullfile(RECORD_FOLDER, OUTPUT_FILE);

%% Validasi data workspace
if ~exist('annotatedFrames', 'var')
    error('Variabel ''annotatedFrames'' tidak ditemukan di workspace.\nPastikan simulasi sudah dijalankan dan To Workspace block sudah terkonfigurasi.');
end

frameData = annotatedFrames;

% Tangani format timeseries (jika Save format diset ke Timeseries)
if isa(frameData, 'timeseries')
    frameData = frameData.Data;
end

% Pastikan dimensi: [H x W x 3 x N]
if ndims(frameData) == 3
    % Hanya satu frame
    frameData = reshape(frameData, size(frameData, 1), size(frameData, 2), 3, 1);
elseif ndims(frameData) ~= 4
    error('Format annotatedFrames tidak dikenali. Ekspektasi [H x W x 3 x N].');
end

N = size(frameData, 4);
fprintf('Jumlah frame: %d\n', N);

%% Tulis video
[~, ~, ext] = fileparts(OUTPUT_PATH);
switch lower(ext)
    case '.mp4'
        profile = 'MPEG-4';
    case '.avi'
        profile = 'Motion JPEG AVI';
    otherwise
        profile = 'Motion JPEG AVI';
        warning('Ekstensi tidak dikenal, menggunakan Motion JPEG AVI.');
end

v = VideoWriter(OUTPUT_PATH, profile);
v.FrameRate = FRAME_RATE;
if strcmp(profile, 'MPEG-4')
    v.Quality = QUALITY;
end
open(v);

for i = 1:N
    frame = uint8(frameData(:, :, :, i));
    writeVideo(v, frame);
    if mod(i, 100) == 0
        fprintf('  Menulis frame %d / %d...\n', i, N);
    end
end

close(v);
fprintf('Video disimpan: %s\n', fullfile(pwd, OUTPUT_PATH));
