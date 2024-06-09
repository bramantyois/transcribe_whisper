import os

import json

import argparse


def find_transcribed_files(meta_dir: str):
    """
    Scan the metadata directory and return a list of transcribed files.
    """
    if not os.path.exists(meta_dir):
        return []
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
    if not os.path.exists(speech_file_dir):
        return []
    
    transcribed_files = find_transcribed_files(meta_dir)
    speech_files = []
    for speech_file in os.listdir(speech_file_dir):
        # check if .mp4
        if not speech_file.endswith(".mp4"):
            continue
        if speech_file not in transcribed_files:
            speech_files.append(speech_file)
    return speech_files


def generate_slurm_script(
    speech_file_dir: str,
    transcript_save_dir: str ="results/transcripts",
    meta_save_dir: str = "results/meta",
    slurm_script_dir: str = ".cache/slurm_scripts",
    whisper_model: str = "large-v3",
    n_files_per_job: int = 10,
    submit_jobs: bool = False
):
    if not os.path.exists(slurm_script_dir):
        os.makedirs(slurm_script_dir)
    
    speech_files = get_untranscribed_files(speech_file_dir, meta_save_dir)

    n_jobs = len(speech_files) // n_files_per_job + 1

    for i in range(n_jobs):
        f"""#!/bin/bash/
        #SBATCH --job-name=transcribe_{i}
        #SBATCH --output=transcribe_{i}.out
        #SBATCH --error=transcribe_{i}.err
        #SBATCH --time=1:00:00
        #SBATCH --cpus-per-task=2
        #SBATCH --partition=gpu
        #SBATCH --gres=gpu:1
        #SBATCH --mem=16G
        
        module load conda
        conda activate whisper
        
        transcribe.py \\
                --speech_files {' '.join(speech_files[i*n_files_per_job:(i+1)*n_files_per_job])} \\
                --whisper_model {whisper_model} \\
                --transcript_save_dir {transcript_save_dir} \\
                --meta_save_dir {meta_save_dir}"""
        slurm_fn = os.path.join(slurm_script_dir, f"transcribe_{i}.slurm")
        with open(slurm_fn, "w") as f:
            f.write("""#!/bin/bash/\n\n""")
            f.write(f"#SBATCH --job-name=transcribe_{i}\n")
            f.write(f"#SBATCH --output=transcribe_{i}.out\n")
            f.write(f"#SBATCH --error=transcribe_{i}.err\n")
            f.write(f"#SBATCH --time=1:00:00\n")
            f.write(f"#SBATCH --cpus-per-task=2\n")
            f.write(f"#SBATCH --partition=gpu\n")
            f.write(f"#SBATCH --gres=gpu:1\n")
            f.write(f"#SBATCH --mem=16G\n\n")
            
            f.write("module load conda\n")
            f.write("conda activate whisper\n")
            f.write(f"python transcribe.py --speech_files {' '.join(speech_files[i*n_files_per_job:(i+1)*n_files_per_job])} --whisper_model {whisper_model} --transcript_save_dir {transcript_save_dir} --meta_save_dir {meta_save_dir}")
             
        if submit_jobs:
            # Submit the job
            os.system(f"sbatch {slurm_fn}")
            


def get_parser():
    parser = argparse.ArgumentParser(description="Generate SLURM scripts for transcribing speech files.")
    parser.add_argument("--speech_file_dir", type=str, help="Directory containing speech files.")
    parser.add_argument("--transcript_save_dir", type=str, default="results/transcripts", help="Directory to save transcripts.")
    parser.add_argument("--meta_save_dir", type=str, default="results/meta", help="Directory to save metadata.")
    parser.add_argument("--whisper_model", type=str, default="large-v3", help="Whisper model to use.")
    parser.add_argument("--n_files_per_job", type=int, default=10, help="Number of files to transcribe per job.")
    parser.add_argument("--submit_jobs", action="store_true", help="Submit jobs to SLURM.")
    
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    generate_slurm_script(
        speech_file_dir=args.speech_file_dir,
        transcript_save_dir=args.transcript_save_dir,
        meta_save_dir=args.meta_save_dir,
        whisper_model=args.whisper_model,
        n_files_per_job=args.n_files_per_job,
        submit_jobs=args.submit_jobs
    )