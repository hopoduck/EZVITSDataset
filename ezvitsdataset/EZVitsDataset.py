import gc
import glob
import os
import datetime

import ffmpeg
import torch
import whisperx
import whisperx.asr
from audio_separator.separator.separator import Separator
from whisperx.types import TranscriptionResult, AlignedTranscriptionResult
from yt_dlp import YoutubeDL
import logging
from typing import Any


class EZVitsDatasetParams:
    def __init__(
        self,
        download_path: str = None,
        uvr_path: str = None,
        output_path: str = None,
        filelist_path: str = None,
        log_level: int = None,
        skip_download: bool = None,
        youtube_dl_option: Any = None,
        sampling_rate: int = None,
        skip_min_time: int = None,
        skip_max_time: int = None,
        remove_original_file: bool = None,
        whisper_model: str = None,
        device: str = None,
        compute_type: str = None,
        language: str = None,
        overwrite_data: bool = None,
        audio_separator_model: str = None,
        batch_size: int = None,
        chunk_size: int = None,
        remove_tmp_file: bool = None,
    ):
        # global_config
        self.download_path = (
            download_path
            if download_path is not None
            else os.path.join(os.getcwd(), "download")
        )
        self.uvr_path = (
            uvr_path if uvr_path is not None else os.path.join(os.getcwd(), "uvr")
        )
        self.output_path = (
            output_path
            if output_path is not None
            else os.path.join(os.getcwd(), "output")
        )
        self.filelist_path = (
            filelist_path
            if filelist_path is not None
            else os.path.join(os.getcwd(), "filelist")
        )
        self.log_level = log_level if log_level is not None else logging.DEBUG

        # download_youtube
        self.skip_download = (
            skip_download if skip_download is not None else skip_download
        )
        self.youtube_dl_option = (
            youtube_dl_option
            if youtube_dl_option is not None
            else {
                "quiet": True,  # 로그 비활성화
                "format": "bestaudio/best",  # 최고 품질의 오디오만 다운로드
                "outtmpl": os.path.join(
                    self.download_path, "%(playlist_index&{}.|)s%(id)s.%(ext)s"
                ),  # 다운드한 파일의 이름과 경로 설정
                "no_overwrites": True,
            }
        )

        # video_to_audio
        self.sampling_rate = sampling_rate if sampling_rate is not None else 44100
        self.skip_min_time = skip_min_time if skip_min_time is not None else 10
        self.skip_max_time = skip_max_time if skip_max_time is not None else 600
        self.remove_original_file = (
            remove_original_file if remove_original_file is not None else False
        )

        # load_whisper_x_model
        self.whisper_model = whisper_model if whisper_model is not None else "large-v2"
        self.device = device if device is not None else "cuda"
        self.compute_type = compute_type if compute_type is not None else "float16"
        self.language = language

        # vocal_separator
        self.overwrite_data = overwrite_data if overwrite_data is not None else False
        self.audio_separator_model = (
            audio_separator_model
            if audio_separator_model is not None
            else "UVR_MDXNET_KARA_2"
        )

        # transcribe_audio
        self.batch_size = batch_size if batch_size is not None else 8
        self.chunk_size = chunk_size if chunk_size is not None else 6

        # split_audio
        self.remove_tmp_file = remove_tmp_file


class EZVitsDataset:
    _videos: list[str] = []
    _model: whisperx.asr.FasterWhisperPipeline | None = None
    _count: int = 0
    _total_time: int = 0
    _start_time: datetime = datetime.datetime.now()
    _log_handler: logging.StreamHandler

    _params: EZVitsDatasetParams

    def __init__(
        self,
        params: EZVitsDatasetParams = None,
    ):
        self._params = params

        self._logger = logging.getLogger("EZVitsDataset")
        self._logger.setLevel(self._params.log_level)
        self._logger.handlers = []
        self._log_handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        self._log_handler.setFormatter(formatter)
        self._logger.addHandler(self._log_handler)

        self._logger.debug("start making dataset...")

        os.makedirs(self._params.download_path, exist_ok=True)
        os.makedirs(self._params.output_path, exist_ok=True)
        os.makedirs(self._params.uvr_path, exist_ok=True)
        os.makedirs(self._params.filelist_path, exist_ok=True)

    def find_videos(self, video_path: str, glob_expr: str = "*"):
        return glob.glob(glob_expr, root_dir=video_path)

    def download_youtube(self, url: str, glob_expr: str = "*"):
        if self._params.skip_download:
            self._logger.debug("skip download...")
            self._videos = self.find_videos(self._params.download_path)
            return self._videos

        self._logger.debug("start download youtube...")
        self._logger.debug(
            f"download path: {os.path.abspath(self._params.download_path)}"
        )

        dl = YoutubeDL(self._params.youtube_dl_option)
        dl.download([url])
        dl.close()

        self._videos = self.find_videos(self._params.download_path)
        self._logger.debug(f"downloaded files: {self._videos}")
        return self._videos

    def make_transform_list(self, videos: list[str]):
        # self._logger.debug("make transform list")
        transform_list = list(
            map(
                lambda file_name: [file_name, os.path.splitext(file_name)[0] + ".wav"],
                videos,
            )
        )
        # self._logger.debug(transform_list)
        return transform_list

    def video_to_audio(
        self,
        input_file: str,
        output_file: str,
    ):
        self._logger.debug(f"{input_file} to audio using ffmpeg")
        if not self._params.overwrite_data and os.path.exists(output_file):
            return True

        probe = ffmpeg.probe(input_file, format="duration")
        duration = int(float(probe["format"]["duration"]))
        # 영상의 길이가 지정 시간(기본 10분) 이상이면 스킵
        if duration <= self._params.skip_min_time:
            self._logger.debug(
                f"too short video.. skip transform this video: {input_file}"
            )
            return False
        elif duration >= self._params.skip_max_time:
            self._logger.debug(
                f"too long video.. skip transform this video: {input_file}"
            )
            return False

        ffmpeg.input(
            input_file,
        ).output(
            output_file,
            ar=self._params.sampling_rate,
            ac=1,
        ).run(
            quiet=True,
            overwrite_output=True,
        )
        if self._params.remove_original_file:
            os.remove(input_file)
        return True

    def load_whisper_x_model(self):
        if self._model is None:
            self.cleanup()
            self._model = whisperx.load_model(
                self._params.whisper_model,
                self._params.device,
                compute_type=self._params.compute_type,
                language=self._params.language,
            )

    def vocal_separator(
        self,
        audio_file: str,
    ):
        uvr_result_file = os.path.join(
            self._params.uvr_path, os.path.split(audio_file)[-1]
        )
        if not self._params.overwrite_data and os.path.exists(uvr_result_file):
            self._logger.debug(f"[skip uvr] uvr result file exist: {uvr_result_file}")
            # 원본 파일 삭제
            if self._params.remove_original_file and os.path.exists(audio_file):
                os.remove(audio_file)
            return os.path.join(self._params.uvr_path, os.path.split(audio_file)[-1])

        self._logger.debug(f"vocal separator {audio_file} file")
        separator = Separator(
            model_file_dir=os.path.join(os.getcwd(), "audio_model"),
            output_single_stem="vocals",
            output_dir=self._params.uvr_path,
        )
        separator.load_model(self._params.audio_separator_model)

        # 보컬 분리
        result_file = separator.separate(audio_file)
        # 경로 없을 시 오류나서 임시 방어코드
        os.makedirs(separator.model_file_dir, exist_ok=True)
        print(result_file)
        result_file = result_file[0]
        # 원본 파일 이름으로 변경
        os.rename(
            os.path.join(self._params.uvr_path, result_file),
            uvr_result_file,
        )
        result_file = uvr_result_file

        # 원본 파일 삭제
        if self._params.remove_original_file and os.path.exists(audio_file):
            os.remove(audio_file)

        return result_file

    def transcribe_audio(self, audio_file: str):
        self._logger.debug(f"transcribe audio {audio_file} file")
        self.load_whisper_x_model()

        audio = whisperx.load_audio(audio_file)
        result = self._model.transcribe(
            audio,
            batch_size=self._params.batch_size,
            chunk_size=self._params.chunk_size,
            language=self._params.language,
        )

        # 2. Align whisper output
        self._logger.debug(f"align transcribe data {audio_file} file")
        align_model, metadata = whisperx.load_align_model(
            language_code=result["language"], device=self._params.device
        )
        result = whisperx.align(
            result["segments"],
            align_model,
            metadata,
            audio,
            self._params.device,
            return_char_alignments=False,
        )
        del align_model

        return result

    def split_audio(
        self,
        audio: str,
        data: (TranscriptionResult | AlignedTranscriptionResult),
    ):
        self._logger.debug(f"split {audio} file")
        # self._logger.debug(f"script data: {data}")

        output_file = os.path.join(self._params.output_path, os.path.split(audio)[-1])
        script_file = os.path.join(self._params.filelist_path, "filelist.txt")
        with open(script_file, "a+", encoding="UTF-8") as file:
            i = 0
            input_stream = ffmpeg.input(audio)
            for script in data["segments"]:
                # 10초 이상 길이일 경우 스킵
                if (script["end"] - script["start"]) > 10:
                    self._logger.debug(
                        f"script is 10s over... skip this script: {script}"
                    )
                    continue
                # 1초 이하 길이일 경우 스킵
                elif (script["end"] - script["start"]) < 1:
                    self._logger.debug(
                        f"script is 1s shorter... skip this script: {script}"
                    )
                    continue
                i += 1
                separated_audio_file = output_file.replace(".wav", f".{i:04}.wav")
                input_stream.output(
                    separated_audio_file,
                    ss=script["start"],
                    t=script["end"] - script["start"] + 0.2,
                ).run(quiet=True, overwrite_output=True)
                self._count += 1
                self._total_time += script["end"] - script["start"] + 0.2

                file.write(f"{separated_audio_file}|{script['text']}\n")

        if self._params.remove_tmp_file:
            os.remove(audio)

    def split_filelist(self):
        with open(
            os.path.join(self._params.filelist_path, "filelist.txt"), encoding="utf-8"
        ) as filelist:
            lines = filelist.readlines()

        train_lines = []
        val_lines = []

        for i, line in enumerate(lines):
            if i % 5 == 4:
                val_lines.append(line)
            else:
                train_lines.append(line)

        with open(
            os.path.join(self._params.filelist_path, "train.txt"),
            "w",
            encoding="utf-8",
        ) as f:
            f.writelines(train_lines)

        with open(
            os.path.join(self._params.filelist_path, "val.txt"), "w", encoding="utf-8"
        ) as f:
            f.writelines(val_lines)

    def set_log_index(self, index: int, count: int):
        formatter = logging.Formatter(
            f"%(asctime)s - %(name)s - %(levelname)s - [{index+1}/{count}] %(message)s"
        )
        self._log_handler.setFormatter(formatter)

    def finish_log(self):
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        self._log_handler.setFormatter(formatter)

        self._logger.debug("all job finished!")
        self._logger.debug(f"dataset count: {self._count}")
        self._logger.debug(f"dataset total time: {self._total_time} sec")

        self._logger.debug(f"job start time: {self._start_time}")
        self._logger.debug(f"job finish time: {datetime.datetime.now()}")
        self._logger.debug(
            f"job elapsed time: {datetime.datetime.now() - self._start_time}"
        )

    def cleanup(self):
        if self._model is not None:
            self._logger.debug("cleanup memory")
        gc.collect()
        torch.cuda.empty_cache()
        self._model = None

    def main(self, mode: str, path_or_url: str):
        """It handles all tasks.
        Args:
            mode: str - Choose 'download' or 'local' mode. 'Download' will download the video from the path parameter and convert it, while 'local' will attempt to convert all files in the specified path.
            path_or_url: str - Please enter the YouTube URL or the path where the video is located.
        """
        # Delete existing script file
        script_file = os.path.join(self._params.output_path, "output.txt")
        if os.path.exists(script_file):
            os.remove(script_file)

        # 1. Video preparation
        videos = None
        if mode == "download":
            videos = self.download_youtube(url=path_or_url)
        elif mode == "local":
            self._videos = self.find_videos(path_or_url)
            videos = self._videos
        transform_list = self.make_transform_list(videos)

        for i in range(len(transform_list)):
            video, audio = transform_list[i]
            print(f"progress: [{i+1}/{len(transform_list)}] {os.path.split(video)[-1]}")
            self.set_log_index(i, len(transform_list))

            # 2. Convert video to audio file
            # result = self.video_to_audio(
            #     os.path.join(self._params.download_path, video),
            #     os.path.join(self._params.output_path, audio),
            # )
            # if not result:
            #     continue
            #
            # # 3. Separate vocals from audio files
            # self.vocal_separator(
            #     os.path.join(self._params.output_path, audio),
            # )

            # 4. Transcribe audio
            script = self.transcribe_audio(
                os.path.join(self._params.uvr_path, audio),
            )

            # 5. Write file text + dialogue according to the extracted dialogue
            self.split_audio(
                os.path.join(self._params.uvr_path, audio),
                script,
            )

            # 6. Split the train.txt and val.txt files
            self.split_filelist()

            # 6. Program Cleanup
            self.cleanup()
            self.finish_log()
