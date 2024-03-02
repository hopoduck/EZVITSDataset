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
pip install "audio-separator[gpu]>=0.15.2" # or pip install "audio-separator[cpu]>=0.15.2"
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

### `EZVitsDatasetParams` parameters

- language: Video file language (automatically detects whisper if not set, but speed may be slow)
- skip_download: Skip YouTube download (default: `False`)
- youtube_dl_option: [`yt-dlp`](https://github.com/yt-dlp/yt-dlp?tab=readme-ov-file#usage-and-options) options
- sampling_rate: Output wav file sampling rate (default: `44100`)
- skip_min_time: Do not process the dataset if it is shorter than the specified number of seconds (default: `10`)
- skip_max_time: Do not process the dataset if it is longer than the specified number of seconds (default: `10`)
- remove_original_file: Delete the original file after processing the dataset (default: `False`)
- whisper_model: whisper model name (default: `'large-v2'`)
- device: device to run cpu/cuda (default: `'cuda'`)
- compute_type: Change depending on the execution device (default: `'float16'`)
- overwrite_data: Whether to overwrite in case of duplication during data processing (default: `False`)
- audio_separator_model: uvr model name (default: `'UVR_MDXNET_KARA_2.onnx'`)
    - You can check available models
      at [here](https://raw.githubusercontent.com/TRvlvr/application_data/main/filelists/download_checks.json).
- batch_size: Amount of data to be processed at once when processing whisper model, can be lowered when VRAM is
  insufficient (default value: `8`)
- chunk_size: Length of one sentence when processing whisper model (default: `6`)
- remove_tmp_file: Whether to delete temporary files after task completion (default: `False`)
- download_path: Path to download video
- uvr_path: Path to save audio file after uvr processing
- output_path: Path to save the audio file after all processing.
- filelist_path: `filelist.txt`, `train.txt`, `val.txt` creation path

## License

EZVitsDataset is released under the MIT license. For more information, see the LICENSE file in the repository.