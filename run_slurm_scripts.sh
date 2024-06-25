# run sbatch for every files in .cache/slurm_scripts
for file in .cache/slurm_scripts/*; do
    sbatch $file
done