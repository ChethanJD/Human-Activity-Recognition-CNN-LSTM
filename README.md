# Human Activity Recognition using CNN-LSTM

A hybrid deep-learning system for recognizing human activities from video by combining **ResNet-50** spatial feature extraction with an **LSTM** temporal sequence classifier.

## Overview

Video understanding requires both:

- **Spatial information** — what appears in each frame.
- **Temporal information** — how the visual content changes across frames.

This project combines the two:

```text
Input Video
    │
    ▼
Frame Sampling
(skipFrames = 2)
    │
    ▼
ResNet-50
(avg_pool)
    │
    ▼
2048-D Spatial Features
    │
    ▼
20-Time-Step Sequence
(2048 × 20)
    │
    ▼
LSTM
128 hidden units
    │
    ▼
Dropout (0.3)
    │
    ▼
Fully Connected + Softmax
    │
    ▼
Predicted Activity
```

The project is implemented in **MATLAB** using the Deep Learning Toolbox ecosystem.

## Activities

The supplied dataset is organized into four activity classes:

- `jumping`
- `pushups`
- `running`
- `walking`

The training script obtains the class label from the folder containing each video.

## Key Features

- Pretrained **ResNet-50** used as a spatial feature extractor.
- Extracts 2048-dimensional features from the `avg_pool` layer.
- Samples video frames using `skipFrames = 2`.
- Uses a fixed sequence length of 20 time steps.
- Explicitly formats each sequence as `2048 × 20`.
- LSTM with 128 units and `OutputMode = last`.
- Dropout regularization with rate 0.3.
- Adam optimizer.
- 15 training epochs.
- Mini-batch size of 2.
- Confusion matrix generation.
- Training accuracy calculation.
- Saves the trained MATLAB model as `trainedLSTM.mat`.
- Exports the trained LSTM model to `trainedLSTM.onnx`.

## Project Structure

```text
Human-Activity-Recognition-CNN-LSTM/
│
├── README.md
├── .gitignore
│
├── atrain.m
├── apredict.m
│
├── trainedLSTM.mat
├── trainedLSTM.onnx
│
├── dataset/
│   └── README.md
│
└── test_video.mp4        # local sample; ignored if large
```

The full video dataset is intentionally not included in this repository because video files can make a Git repository unnecessarily large. See [`dataset/README.md`](dataset/README.md). The local `test_video.mp4` can also be kept outside Git if it is large.

## Requirements

The MATLAB scripts require a MATLAB installation with the relevant deep-learning and image/video functionality.

The project uses:

- MATLAB
- Deep Learning Toolbox
- ResNet-50
- VideoReader
- Image Processing functionality
- Deep Learning Toolbox Converter for ONNX export

The exact availability of functions can depend on the MATLAB release and installed products.

## How the Training Works

Run:

```matlab
atrain
```

The training script:

1. Loads pretrained ResNet-50.
2. Searches the `dataset` directory recursively for `.mp4` files.
3. Uses each video's parent folder as its activity label.
4. Reads video frames.
5. Resizes frames to the ResNet-50 input dimensions.
6. Extracts features from the `avg_pool` layer.
7. Stores the extracted features as a `2048 × 20` sequence.
8. Normalizes each sequence.
9. Trains the LSTM classifier.

The feature extraction and sequence preparation are implemented directly in `atrain.m`.

## LSTM Architecture

```text
sequenceInputLayer(2048)
        │
        ▼
lstmLayer(128, OutputMode="last")
        │
        ▼
dropoutLayer(0.3)
        │
        ▼
fullyConnectedLayer(numClasses)
        │
        ▼
softmaxLayer
        │
        ▼
classificationLayer
```

## Training Configuration

| Parameter | Value |
|---|---|
| CNN backbone | ResNet-50 |
| Feature layer | `avg_pool` |
| Feature size | 2048 |
| Sequence length | 20 frames |
| Frame skip | 2 |
| LSTM units | 128 |
| LSTM output mode | `last` |
| Dropout | 0.3 |
| Optimizer | Adam |
| Epochs | 15 |
| Mini-batch size | 2 |
| Shuffle | Every epoch |

## Running Prediction

Place a test video at:

```text
samples/test_video.mp4
```

or update the path inside `apredict.m`.

Then run:

```matlab
apredict
```

The prediction pipeline extracts features from the test video, creates the required sequence, and classifies it using the trained LSTM.

The script also prints confidence scores for the available classes.

## Model Files

Two trained-model formats are provided:

### `trainedLSTM.mat`

MATLAB-native saved model containing the trained LSTM network.

### `trainedLSTM.onnx`

ONNX representation intended to support cross-platform model use and deployment workflows.

The trained models are stored at the project root because the supplied prediction script loads/saves them using root-level filenames.

## Results

The project report documents:

- Successful validation of the extracted sequences as `2048 × 20` arrays.
- A sample prediction of **Walking**.
- Sample confidence scores of **98.20% Walking** and **1.80% Running**.
- A confusion matrix illustrating differentiation between target classes.

The reported result is a sample project output and should not be interpreted as a complete four-class benchmark unless the corresponding experiment is rerun on the full dataset.

## Problem Solved

A direct combination of raw image tensors and an LSTM can create feature-dimension mismatches and unnecessary computational load.

This project addresses that by first compressing each frame into a dense 2048-dimensional representation using ResNet-50, then arranging those representations into a fixed `2048 × 20` time-series matrix before LSTM processing.

## Advantages

- Combines spatial and temporal information.
- Uses transfer learning through pretrained ResNet-50 features.
- Reduces the amount of raw visual data passed to the LSTM.
- Fixed sequence formatting makes the LSTM input consistent.
- ONNX export provides a deployment-oriented model format.

## Limitations

Based on the project design:

- The model uses a fixed sequence length of 20 time steps.
- Frame skipping can remove some motion information.
- The quality of recognition depends on the supplied video dataset.
- The supplied project report does not provide a complete four-class numerical benchmark for all dataset videos.
- Deployment to other environments may require additional ONNX compatibility testing.

## Future Scope

The project report identifies possible extensions including:

- Edge deployment on low-power devices.
- More advanced CNN backbones such as Vision Transformers.
- Multi-stream networks incorporating additional motion information.
- Broader real-world video datasets.

## Authors

**Chethan JD**  
Roll No: `20231ECE0174`

**Dileep J**  
Roll No: `20231ECE0182`

**Vinay S**  
Roll No: `20231ECE0180`

**Nishanth E**  
Roll No: `20231ECE0216`

## Academic Project

This repository contains an academic project on hybrid CNN-RNN video processing and human activity recognition.
