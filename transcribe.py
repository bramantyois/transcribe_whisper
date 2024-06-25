import os
import argparse
import json

from typing import List, Optional
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
    id = id.split(".")[0]
    return id


def transcribe_batch(audio_files: List[str]):
    """
    Given a list of audio files, transcribe them using the whisper model. Batch processing
    """
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        processor = AutoProcessor.from_pretrained("openai/whisper-large-v3")
        model = WhisperForConditionalGeneration.from_pretrained(
            "openai/whisper-large-v3", torch_dtype=torch_dtype
        )
        model.to(device)

        raw_audios = [whisper.load_audio(a) for a in audio_files]

        inputs = processor(
            raw_audios,
            return_tensors="pt",
            truncation=False,
            padding="longest",
            return_attention_mask=True,
            sampling_rate=16_000,
        )

        inputs = inputs.to(device, torch_dtype)

        result = model.generate(
            **inputs,
            condition_on_prev_tokens=False,
            temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
            logprob_threshold=-1.0,
            compression_ratio_threshold=1.35,
            return_timestamps=True,
        )

        decoded = processor.batch_decode(result, skip_special_tokens=True)
        status = ["succeded"] * len(audio_files)

    except Exception as e:
        decoded = [None] * len(audio_files)
        status = ["failed"] * len(audio_files)
        print(f"Failed to transcribe {audio_files}: {e}")
    return decoded, status


def transcribe_per_files(audio_files: List[str]):
    """
    Given a list of audio files, transcribe them using the whisper model. Simple method.
    """
    model = whisper.load_model("large-v3")

    trancripts = []
    status = []
    for file in audio_files:
        try:
            result = model.transcribe(file)["text"]
            trancripts.append(result)
            status.append("succeded")
        except Exception as e:
            # failing
            trancripts.append(None)
            print(f"Failed to transcribe {file}: {e}")
            status.append("failed")

    return trancripts, status


def write_meta(
    file_list: List[str],
    status: Optional[List[str]] = None,
    upload_to_s3: bool = True,
    meta_save_dir: str = "meta",
    transcript_save_dir: str = "transcripts",
):
    if status is None:
        status = ["transcribing"] * len(file_list)

    metas = []

    for file, stat in zip(file_list, status):
        transcript_fn = get_id(file)
        fn = os.path.join(transcript_save_dir, f"{transcript_fn}.txt")
        meta = {
            "transcription_file": f"{fn}.txt",
            "speech_file": file,
            "status": stat,
        }
        metas.append(meta)

        meta_save_path = os.path.join(meta_save_dir, f"{fn}.json")
        with open(meta_save_path, "w") as f:
            json.dump(meta, f)

        if upload_to_s3:
            save_file_to_s3(meta_save_path)


def write_transcript(
    transcripts: List[str],
    transcript_save_dir: str = "transcripts",
    upload_to_s3: bool = True,
):
    for t in transcripts:
        transcript_fn = get_id(file)
        transcript_fn = os.path.join(transcript_save_dir, f"{transcript_fn}.txt")
        with open(transcript_fn, "w") as f:
            f.write(t)

        if upload_to_s3:
            save_file_to_s3(transcript_fn)


def transcribe_file(
    audio_files: List[str],
    transcript_save_dir: str = "transcripts",
    meta_save_dir: str = "meta",
    upload_to_s3: bool = False,
    is_batch_processing: bool = False,
    delete_files: bool = True,
):
    """
    Transcribe a list of speech files. This method should
    return a list of transcribed text.
    """
    if not os.path.exists(transcript_save_dir):
        os.makedirs(transcript_save_dir)
    if not os.path.exists(meta_save_dir):
        os.makedirs(meta_save_dir)

    # check if speech files are present. Otherwise download from s3
    for f in audio_files:
        if not os.path.exists(f):
            print(f"Downloading {f} from S3")
            save_file_to_s3(f)

    # now
    write_meta(
        audio_files,
        meta_save_dir=meta_save_dir,
        transcript_save_dir=transcript_save_dir,
        upload_to_s3=upload_to_s3,
    )

    # transcribe each speech file with error handling
    if is_batch_processing:
        transcripts, status = transcribe_batch(audio_files)
    else:
        transcripts, status = transcribe_per_files(audio_files)

    write_meta(audio_files, status, meta_save_dir, transcript_save_dir, upload_to_s3)

    write_transcript(transcripts, transcript_save_dir, upload_to_s3)
    
    if delete_files:
        for f in audio_files:
            os.remove(f)
    

def get_parser():
    parser = argparse.ArgumentParser(description="Transcribe a list of speech files")
    parser.add_argument(
        "audio_files", type=str, nargs="+", help="List of speech files to transcribe"
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

    parser.add_argument("--upload_to_s3", action="store_true", help="Upload to S3")
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    transcribe_file(
        args.audio_files,
        args.whisper_model,
        args.transcript_save_dir,
        args.meta_save_dir,
        args.upload_to_s3,
    )
