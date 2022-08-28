#!/bin/bash
#SBATCH -p t4v2
#SBATCH --exclude=gpu068
#SBATCH --gres=gpu:4
#SBATCH -c 8
#SBATCH --qos=high
#SBATCH --mem=64G
#SBATCH --job-name=expt
#SBATCH --output=out.log
#SBATCH --ntasks=1

source ~/sourcefile
source ~/score2perfenv/bin/activate
date;hostname;pwd;whoami

python main.py --mode=train --config.dataset.path="Store/asap-dataset" --config.dataset.mode="perf2score" --workdir=autoenc_end2end --config=configs.py:autoencoder --config.pt_enc_path=None