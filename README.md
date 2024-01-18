[English](README.md) | [한국어](README-ko.md)

# EZVitsDataset

## Description

EZVitsDataset processes video datasets.
This library creates a dataset by downloading a YouTube video or loading the video from a specified path.

## Installation

This project can be run only if ffmpeg is installed on your computer.

```bash
git clone https://github.com/hopoduck/EZVitsDataset.git
cd ez-vits-dataset 

conda create --name dataset python=3.10

conda activate dataset

pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118
pip install "audio-separator[gpu]>=0.13.0" # or pip install "audio-separator[cpu]>=0.13.0"
pip install -r requirements.txt
```

## How to use

Import the EZVitsDataset and EZVitsDatasetParams modules and initialize EZVitsDataset. For initialization use: For
options that can be set, see the `EZVitsDatasetParams` object.  
Then call the `main` function to run download or local mode. The `mode` parameter selects `download` or `local`, and
the `path_or_url` parameter selects a YouTube URL or specifies the path where the video file is located.

```python
from ezvitsdataset.EZVitsDataset import EZVitsDataset, EZVitsDatasetParams

dataset = EZVitsDataset(
    EZVitsDatasetParams(
        device="cuda",
        language="en",
    )
)

dataset.main(
    mode="download",
    path_or_url="youtube video or playlist url",
)
```

After that, run `main.py`.

```shell
python main.py
```

## License

EZVitsDataset is released under the MIT license. For more information, see the LICENSE file in the repository.