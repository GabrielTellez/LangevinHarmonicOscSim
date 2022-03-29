#!/bin/bash
#SBATCH --time 2-00:00:00
#SBATCH --job-name jupyter-notebook
#SBATCH -o jupyter_log/jupyter-notebook-o%J.log
#SBATCH -e jupyter_log/jupyter-notebook-e%J.log
#SBATCH --cpus-per-task=48

# get tunneling info

port=8879
node=$(hostname -s)
user=$(whoami)

module load anaconda/python3
source ~/bin/conda_init.sh
# activate conda env with numpy numba pandas plotly and jupyter
conda activate harmonic_osc_env
# run jupyter notebook
jupyter-notebook --no-browser --port=${port} --ip=${node}

