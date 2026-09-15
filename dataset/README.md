# Dataset

The training code expects video files under this directory and searches recursively for `.mp4` files.

The supplied dataset is organized into four activity classes:

```text
dataset/
├── jumping/
├── pushups/
├── running/
└── walking/
```

Each class folder contains example activity videos.

## Why the videos are not stored in Git

The video dataset is intentionally excluded from the Git repository because committing many large video files makes cloning and version control unnecessarily heavy.

The `.gitignore` file therefore ignores video files inside this directory.

## Using the dataset locally

Place the dataset in this structure:

```text
dataset/
├── jumping/
│   ├── jump1.mp4
│   └── ...
├── pushups/
│   ├── pushup1.mp4
│   └── ...
├── running/
│   ├── run1.mp4
│   └── ...
└── walking/
    ├── walk1.mp4
    └── ...
```

Then run `atrain.m`.

The training script automatically uses the parent folder name as the activity label.
