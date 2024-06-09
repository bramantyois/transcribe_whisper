import os

import json

import argparse


def find_transcribed_files(meta_dir: str):
    """
    Scan the metadata directory and return a list of transcribed files.
    """
    transcribed_files = []
    for meta_file in os.listdir(meta_dir):
        if meta_file.endswith(".json"):
            with open(os.path.join(meta_dir, meta_file), "r") as f:
                meta = json.load(f)
                if meta["status"] == "success":
                    transcribed_files.append(meta["speech_file"])
    return transcribed_files


def get_untranscribed_files(speech_file_dir: str, meta_dir: str):
    """
    Get a list of speech files that have not been transcribed.
    """
    transcribed_files = find_transcribed_files(meta_dir)
    speech_files = []
    for speech_file in os.listdir(speech_file_dir):
        if speech_file not in transcribed_files:
            speech_files.append(speech_file)
    return speech_files


def generate_slurm_script(
    speech_file_dir: str,
    transcript_save_dir: str,
    meta_save_dir: str,
    whisper_model: str,
    n_files_per_job: int = 10,
):

    speech_files = get_untranscribed_files(speech_file_dir, meta_save_dir)

    n_jobs = len(speech_files) // n_files_per_job + 1

    for i in range(n_jobs):
        job_script = f"""
            #!/bin/bash
            #SBATCH --job-name=transcribe_{i}
            #SBATCH --output=transcribe_{i}.out
            #SBATCH --error=transcribe_{i}.err
            #SBATCH --time=1:00:00
            #SBATCH --cpus-per-task=2
            #SBATCH --partition gpu
            #SBATCH --gres gpu:1
            #SBATCH --mem=16G
            #SBATCH --partition=short
            #SBATCH --mail-type=END

            module load anaconda/2020.11
            conda activate whisper

            transcribe.py \\
                --speech_files {' '.join(speech_files[i*n_files_per_job:(i+1)*n_files_per_job])} \\
                --whisper_model {whisper_model} \\
                --transcript_save_dir {transcript_save_dir} \\
                --meta_save_dir {meta_save_dir}
            """
        with open(f"transcribe_{i}.slurm", "w") as f:
            f.write(job_script)
