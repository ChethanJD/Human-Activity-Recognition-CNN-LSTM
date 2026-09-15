clc; clear; close all;
%% ==========================================
% LOAD PRETRAINED CNN
% ==========================================
netCNN = resnet50;
inputSize = netCNN.Layers(1).InputSize;
featureLayer = 'avg_pool'; % 2048-d features

%% ==========================================
% DATASET
% ==========================================
datasetPath = 'dataset';
videoFiles = dir(fullfile(datasetPath, '**', '*.mp4'));
features = {};                 % cell array of numeric sequences
labels   = categorical();      % categorical labels

%% ==========================================
% PARAMETERS
% ==========================================
maxFrames  = 20;   % sequence length (time steps)
skipFrames = 2;    % speed

%% ==========================================
% FEATURE EXTRACTION (features × time)
% ==========================================
disp("Extracting features...")
for i = 1:numel(videoFiles)
    filePath = fullfile(videoFiles(i).folder, videoFiles(i).name);
    [~, labelName] = fileparts(videoFiles(i).folder);

    try
        v = VideoReader(filePath);
    catch
        warning("Skipping bad video: " + filePath);
        continue;
    end

    % Preallocate as (features × time) → (2048 × maxFrames)
    seq = zeros(2048, maxFrames, 'single');
    frameIdx = 0;
    t = 0;

    while hasFrame(v)
        frame = readFrame(v);
        frameIdx = frameIdx + 1;

        if mod(frameIdx, skipFrames) ~= 0
            continue;
        end

        if t >= maxFrames
            break;
        end

        t = t + 1;
        frame = imresize(frame, inputSize(1:2));

        % f is (2048×1)
        f = activations(netCNN, frame, featureLayer, 'OutputAs','columns');

        % Store as column in (2048 × time) matrix
        seq(:, t) = single(f);
    end

    if t == 0
        warning("No valid frames: " + filePath);
        continue;
    end

    % (Optional) normalize
    seq = seq ./ (max(abs(seq(:))) + 1e-6);

    features{end+1, 1} = seq;                    % (2048 × maxFrames)
    labels(end+1, 1)   = categorical({labelName});
end

%% ==========================================
% VALIDATION
% ==========================================
disp("Final validation...")
disp("Total samples:"); disp(numel(features))
disp("Classes:"); disp(categories(labels))

numClasses = numel(categories(labels));
if numClasses < 2
    error("Need at least 2 classes.");
end

% Check all sequences are (2048 × maxFrames)
for i = 1:numel(features)
    s = size(features{i});
    if ~isequal(s, [2048 maxFrames])
        error("Bad feature size at sample %d: got [%d %d]", i, s(1), s(2));
    end
end
disp("All sequences are [2048 × 20] ✓")

%% ==========================================
% ENSURE CORRECT DATATYPE
% ==========================================
features = cellfun(@(x) single(x), features, 'UniformOutput', false);

%% ==========================================
% LSTM NETWORK
% ==========================================
layers = [
    sequenceInputLayer(2048, 'Name','input')
    lstmLayer(128, 'OutputMode','last')
    dropoutLayer(0.3)
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer
];

%% ==========================================
% TRAIN OPTIONS
% ==========================================
options = trainingOptions('adam', ...
    'MaxEpochs', 15, ...
    'MiniBatchSize', 2, ...
    'Shuffle','every-epoch', ...
    'Verbose', true, ...
    'Plots','training-progress');

%% ==========================================
% TRAIN
% ==========================================
netLSTM = trainNetwork(features, labels, layers, options);
disp("✅ TRAINING COMPLETED SUCCESSFULLY!");