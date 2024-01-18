[English](README.md) | [한국어](README-ko.md)

# EZVitsDataset

## 설명

EZVitsDataset은 비디오 데이터셋을 처리합니다.
이 라이브러리는 YouTube 비디오를 다운로드하거나 지정된 경로에서 비디오를 로드하여 데이터 세트를 생성합니다.

## 설치

이 프로젝트는 컴퓨터에 ffmpeg가 설치된 경우에만 실행할 수 있습니다.

```bash
git clone https://github.com/hopoduck/EZVitsDataset.git
cd ez-vits-dataset 

conda create --name dataset python=3.10

conda activate dataset

pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118
pip install "audio-separator[gpu]>=0.13.0" # or pip install "audio-separator[cpu]>=0.13.0"
pip install -r requirements.txt
```

## 사용법

EZVitsDataset 및 EZVitsDatasetParams 모듈을 가져오고 EZVitsDataset를 초기화합니다. 설정할 수 있는 옵션은 `EZVitsDatasetParams` 객체를 참조하세요.
그런 다음 `main` 함수를 호출하여 다운로드 또는 로컬 모드를 실행합니다. `mode` 매개변수는 `download` 또는 `local`을 선택하고, `path_or_url` 매개변수는 YouTube URL을
선택하거나 동영상 파일이 있는 경로를 지정합니다.

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

그 후 `main.py`를 실행하세요.

```shell
python main.py
```

## 라이선스

EZVitsDataset은 MIT 라이센스로 배포됩니다. 자세한 내용은 저장소의 LICENSE 파일을 참조하세요.