from typing import List

import pandas as pd

import os

import json

import argparse

from s3utils import load_folder_from_s3, get_list_of_files_s3


def find_transcribed_files(meta_dir: str):
    """
    Scan the metadata directory and return a list of transcribed files. meta dir should corresponds to S3 bucket.
    """
    if not os.path.exists(meta_dir):
        try:
            load_folder_from_s3(s3_folder_path=meta_dir)
        except Exception as e:
            print(f"Failed to load {meta_dir} from S3: {e}")
            return []

    if not os.path.exists(meta_dir):
        os.makedirs(meta_dir)

    transcribed_files = []
    for meta_file in os.listdir(meta_dir):
        if meta_file.endswith(".json"):
            with open(os.path.join(meta_dir, meta_file), "r") as f:
                meta = json.load(f)
                if meta["status"] == "succeded":
                    transcribed_files.append(meta["speech_file"])
    return transcribed_files


def get_durations(
    files: List[str],
    csv_files: str = "world/snapshot.20240606153519/video_list_eng_title.csv",
):

    basenames = [os.path.basename(f) for f in files]
    ids = [os.path.basename(f).split(".")[0] for f in basenames]
    
    # check if csv_files is there
    if not os.path.exists(csv_files):
        load_file_from_s3(csv_files)
    
    file_df = pd.read_csv(csv_file)
    
    durations = []
    for id in ids:
        try:
            duration = file_df[file_df["video_id"] == id]["video_duration_seconds"]
            durations.append(duration)
        except:
            durations.append(None)
        
    return durations


def get_untranscribed_files(
    speech_file_dir: str, meta_dir: str, return_size: bool = True
):
    """
    Get a list of speech files that have not been transcribed.
    """
    transcribed_files = find_transcribed_files(meta_dir)
    all_files, sizes = get_list_of_files_s3(speech_file_dir, return_size=True)

    all_files = all_files[:100]
    sizes = sizes[:100]

    # now get untranscribed_file
    untranscribed_files = []
    untranscribed_sizes = []
    for file, size in zip(all_files, sizes):
        if not file.endswith(".mp4"):
            continue
        if file not in transcribed_files:
            untranscribed_files.append(file)
            untranscribed_sizes.append(size)

    if return_size:
        return untranscribed_files, untranscribed_sizes

    return untranscribed_files


def get_batches(files_dur_in_seconds, speech_files, max_dur_per_batch_in_seconds : float=7200, max_dur_per_file: float=7200):
    # now sort ascending
    sorted_files = sorted(zip(files_dur_in_seconds, speech_files))

    files = [f for _, f in sorted_files]
    durations = [d for d, _ in durations]

    batches = []
    cur_batch = []
    cur_sum = 0
    for f, s in zip(files, durations):
        # skip if duration is too long
        if s > max_dur_per_file:
            continue
        
        if cur_sum + s > max_dur_per_batch_in_seconds:
            if len(cur_batch) > 0:
                batches.append(cur_batch.copy())
        else:
            cur_batch.append(f)
            cur_sum += s

    return batches


def generate_slurm_script(
    speech_file_dir: str,
    transcript_save_dir: str = "results/transcripts",
    meta_save_dir: str = "results/meta",
    submit_jobs: bool = False,
    cache_dir: str = ".cache",
    is_tardis: bool = False,
    upload_to_s3: bool = False,
):
    slurm_script_dir = os.path.join(cache_dir, "slurm_scripts")
    slurm_out_dir = os.path.join(cache_dir, "slurm_out")
    slurm_err_dir = os.path.join(cache_dir, "slurm_err")
    if not os.path.exists(slurm_script_dir):
        os.makedirs(slurm_script_dir)
    if not os.path.exists(slurm_out_dir):
        os.makedirs(slurm_out_dir)
    if not os.path.exists(slurm_err_dir):
        os.makedirs(slurm_err_dir)

    speech_files, file_sizes = get_untranscribed_files(
        speech_file_dir, meta_save_dir, return_size=True
    )

    if is_tardis:
        max_file_size_per_batch = 16e7
    else:
        max_file_size_per_batch = 40e7
    batches = get_batches(
        file_sizes, speech_files, max_size_per_batch=max_file_size_per_batch
    )

    for i, batch in enumerate(batches):
        slurm_fn = os.path.join(slurm_script_dir, f"transcribe_{i}.slurm")
        out_fn = os.path.join(slurm_out_dir, f"transcribe_{i}.out")
        err_fn = os.path.join(slurm_err_dir, f"transcibe_{i}.err")

        with open(slurm_fn, "w") as f:
            f.write("""#!/bin/bash\n\n""")
            f.write(f"#SBATCH --job-name=transcribe_{i}\n")
            f.write(f"#SBATCH --output={out_fn}\n")
            f.write(f"#SBATCH --error={err_fn}\n")
            f.write(f"#SBATCH --time=2:00:00\n")
            f.write(f"#SBATCH --cpus-per-task=8\n")
            if is_tardis:
                f.write(f"#SBATCH --partition=gpu\n")
                f.write(f"#SBATCH --gres=gpu:turing:1\n")
            else:
                f.write(f'#SBATCH --constraint="gpu"\n')
                f.write(f"#SBATCH --gres=gpu:a100:1\n")
            f.write(f"#SBATCH --mem=64G\n\n")

            f.write("module load conda\n")
            f.write("conda activate whisper\n")
            if not upload_to_s3:
                f.write(
                    f"python transcribe.py {' '.join(batch)} --transcript_save_dir {transcript_save_dir} --meta_save_dir {meta_save_dir}"
                )
            else:
                f.write(
                    f"python transcribe.py {' '.join(batch)} --transcript_save_dir {transcript_save_dir} --meta_save_dir {meta_save_dir} --upload_to_s3"
                )

        if submit_jobs:
            os.system(f"sbatch {slurm_fn}")


def get_parser():
    parser = argparse.ArgumentParser(
        description="Generate SLURM scripts for transcribing speech files."
    )
    parser.add_argument(
        "--speech_file_dir", type=str, help="Directory containing speech files."
    )
    parser.add_argument(
        "--transcript_save_dir",
        type=str,
        default="results/transcripts",
        help="Directory to save transcripts.",
    )
    parser.add_argument(
        "--meta_save_dir",
        type=str,
        default="results/meta",
        help="Directory to save metadata.",
    )

    parser.add_argument(
        "--submit_jobs", action="store_true", help="Submit jobs to SLURM."
    )
    parser.add_argument(
        "--is_tardis", action="store_true", help="Use TARDIS cluster instead of RAVEN."
    )
    parser.add_argument(
        "--upload_to_s3", action="store_true", help="Upload transcripts to S3"
    )

    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    generate_slurm_script(
        speech_file_dir=args.speech_file_dir,
        transcript_save_dir=args.transcript_save_dir,
        meta_save_dir=args.meta_save_dir,
        submit_jobs=args.submit_jobs,
        is_tardis=args.is_tardis,
        upload_to_s3=args.upload_to_s3,
    )
