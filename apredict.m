%% ==========================================
% STEP 2: EVALUATE THE MODEL
% ==========================================

% Predict on training data (or test split if you have one)
predictedLabels = classify(netLSTM, features);

% Accuracy
accuracy = sum(predictedLabels == labels) / numel(labels) * 100;
fprintf("Training Accuracy: %.2f%%\n", accuracy);

% Confusion Matrix
figure;
confusionchart(labels, predictedLabels);
title('Confusion Matrix');

%% ==========================================
% STEP 3: SAVE THE TRAINED MODEL
% ==========================================

save('trainedLSTM.mat', 'netLSTM');
disp("✅ Model saved as trainedLSTM.mat");

%% ==========================================
% STEP 4: EXPORT AS ONNX (for Python/TF/deployment)
% ==========================================

% Requires Deep Learning Toolbox Converter
exportONNXNetwork(netLSTM, 'trainedLSTM.onnx');
disp("✅ Model exported as trainedLSTM.onnx");

%% ==========================================
% STEP 5: TEST ON A NEW VIDEO
% ==========================================

function seq = extractFeatures(videoPath, netCNN, inputSize, featureLayer, maxFrames, skipFrames)
    try
        v = VideoReader(videoPath);
    catch
        error("Cannot read video: " + videoPath);
    end

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
        f = activations(netCNN, frame, featureLayer, 'OutputAs','columns');
        seq(:, t) = single(f);
    end

    if t == 0
        error("No valid frames found in: " + videoPath);
    end

    seq = seq ./ (max(abs(seq(:))) + 1e-6);
end

% --- Run prediction on a new video ---
testVideoPath = 'test_video.mp4';   % <-- change to your test video path

testSeq = extractFeatures(testVideoPath, netCNN, inputSize, featureLayer, maxFrames, skipFrames);

[predLabel, scores] = classify(netLSTM, {testSeq});

fprintf("\n🎯 Predicted Class : %s\n", string(predLabel));
fprintf("📊 Confidence Scores:\n");
classNames = categories(labels);
for i = 1:numel(classNames)
    fprintf("   %-15s : %.2f%%\n", classNames{i}, scores(i)*100);
end

%% ==========================================
% STEP 6: DEPLOY AS STANDALONE APP (optional)
% ==========================================

% To compile as a standalone executable (requires MATLAB Compiler):
% mcc -m nn1.m -o ActionRecognitionApp

% To deploy to MATLAB Production Server:
% Use deploytool → Production Server Compiler

%% ==========================================
% STEP 7: UPLOAD CHECKLIST
% ==========================================
disp("=== UPLOAD CHECKLIST ===")
disp("[ ] trainedLSTM.mat   → MATLAB model file")
disp("[ ] trainedLSTM.onnx  → Cross-platform model (Python/TF/PyTorch)")
disp("[ ] nn1.m             → Training script")
disp("[ ] dataset/          → Folder structure (running/, walking/)")
disp("[ ] test_video.mp4    → Sample test video")
disp("================================")