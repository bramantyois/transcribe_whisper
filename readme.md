# Transcribe .mp4 file using Whisper
## Setting up Environment
using conda environment is recommended. 
```bash
conda create -n whisper python=3.10
conda activate whisper
```
Install torch 
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```
Install other dependencies
```bash
pip install -r requirements.txt
```
## Running transcription 
to run transcription on a video file, run the following command
```bash
python transcribe.py data/v=_Bs2o5sLVD8.mp4 --whisper_model large-v3 --transcript_save_dir results/transcripts --meta_save_dir results/meta
```
The above command will transcribe the video file `data/v=_Bs2o5sLVD8.mp4` using the `large-v3` model and save the transcript in `results/transcripts` and metadata in `results

## Running transcription on HPC via SLURM
python script `generate_slurm.py` is provided to run transcription on HPC via SLURM. This scripy should be run on the login node (it will automatically run `sbatch ...`)
```bash
python generate_slurm.py --speech_file_dir data --whisper_model large-v3 --transcript_save_dir results/transcripts --meta_save_dir results/meta --n_files_per_job 10 --submit_jobs
```
The above command will transcribe all the files in `data` directory using the `large-v3` model and save the transcript in `results/transcripts` and metadata in `results/meta`. For each job, it will transcribe 10 files. The `--submit_jobs` flag will submit the jobs to the HPC, if not specified, the script will only generate slurm scripts but not submitting. 
```
