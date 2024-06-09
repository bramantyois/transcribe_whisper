import os
import argparse
import json

from typing import List
import whisper

from tqdm import tqdm

from s3utils import save_file_to_s3

# load .env file
from dotenv import load_dotenv


load_dotenv()


def get_id(youtube_id: str):
    """
    Get the video id from a youtube url.
    """
    id = youtube_id.split("=")[-1]
    id = id.replace("_", "")
    id = id.split(".")[0]
    return id


def transcribe_file(
    speech_files: List[str],
    whisper_model: str = "large-v3",
    transcript_save_dir: str = "transcripts",
    meta_save_dir: str = "meta",
    upload_to_s3: bool = False,
):
    """
    Transcribe a list of speech files. This method should
    return a list of transcribed text.
    """
    if not os.path.exists(transcript_save_dir):
        os.makedirs(transcript_save_dir)
    if not os.path.exists(meta_save_dir):
        os.makedirs(meta_save_dir)

    model = whisper.load_model(whisper_model)

    # transcribe each speech file with error handling

    for speech_file in tqdm(speech_files):
        fn = get_id(os.path.basename(speech_file))  # get youtube id

        save_path = os.path.join(transcript_save_dir, f"{fn}.txt")

        meta = {
            "transcription_file": save_path,
            "speech_file": speech_file,
            "status": "failed",
        }

        try:
            result = model.transcribe(speech_file)["text"]

            # save the transcribed text
            with open(save_path, "w") as f:
                f.write(result)

            # save meta
            meta["status"] = "succeded"

        except Exception as e:
            print(f"Failed to transcribe {speech_file}: {e}")

        meta_save_path = os.path.join(meta_save_dir, f"{fn}.json")
        with open(meta_save_path, "w") as f:
            json.dump(meta, f)

        if upload_to_s3:
            save_file_to_s3(save_path)
            save_file_to_s3(meta_save_path)


def get_parser():
    parser = argparse.ArgumentParser(description="Transcribe a list of speech files")
    parser.add_argument(
        "speech_files", type=str, nargs="+", help="List of speech files to transcribe"
    )
    parser.add_argument(
        "--whisper_model",
        type=str,
        default="large-v3",
        help="Whisper model to use for transcription",
    )
    parser.add_argument(
        "--transcript_save_dir",
        type=str,
        default="transcripts",
        help="Directory to save transcriptions",
    )
    parser.add_argument(
        "--meta_save_dir", type=str, default="meta", help="Directory to save metadata"
    )
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    transcribe_file(
        args.speech_files,
        args.whisper_model,
        args.transcript_save_dir,
        args.meta_save_dir,
    )
