#!/bin/bash
# SAM 3D Objects (github.com/facebookresearch/sam-3d-objects, doc/setup.md) installed on Jean Zay: the conda env
# $SCRATCH/conda/envs/sam3d (its environments/default.yml: Python 3.11, CUDA 12.1 toolkit), then the package with its
# [dev], [p3d] and [inference] extras, the CUDA extensions built for the H100 (sm_90) on this GPU-less node, and the
# hydra patch. The prepost partition has internet. The checkpoints are fetched from Hugging Face (gated: an accepted
# account's token in $HF_HOME) into the repo's checkpoints/hf.
#   sbatch launch/slurm/install_sam3d.slurm
#SBATCH -J install_sam3d
#SBATCH -A iyg@cpu
#SBATCH --partition=prepost
#SBATCH --nodelist=jean-zay-pp2              # jean-zay-pp3's proxy answered pypi.org with 403 Forbidden (2026-10-29)
#SBATCH --cpus-per-task=16
#SBATCH -t 10:00:00
#SBATCH -o logs/slurm/%x_%j.out
set -eo pipefail
REPO=/lustre/fswork/projects/rech/iyg/uvx44rt/Tactile-Video2Sim
SRC=$WORK/ext/sam-3d-objects
ENV=$SCRATCH/conda/envs/sam3d
module purge
module load anaconda-py3/2024.06
export PYTHONNOUSERSITE=1 MAX_JOBS=16 FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST="9.0"
cd "$SRC"
if [ ! -x "$ENV/bin/python" ]; then
    echo "=== conda env $ENV ==="
    conda env create -p "$ENV" -f environments/default.yml
fi
PIP="$ENV/bin/python -m pip"
export PATH="$ENV/bin:$PATH" CUDA_HOME="$ENV"
# conda's CUDA toolkit keeps its headers and libraries under targets/: pytorch3d's build does not find
# cuda_runtime_api.h in $CUDA_HOME/include
export CPATH="$ENV/targets/x86_64-linux/include${CPATH:+:$CPATH}"
export LIBRARY_PATH="$ENV/targets/x86_64-linux/lib:$ENV/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
echo "=== pip: [dev], [p3d], [inference] ==="
# doc/setup.md adds https://pypi.ngc.nvidia.com too: Jean Zay's proxy refuses it (500), and pip retries it 5 times per
# package; the NVIDIA packages the requirements pin are on pypi.org as well
export PIP_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cu121"
$PIP install -e '.[dev]' | grep -v "already satisfied"
$PIP install -e '.[p3d]'
export PIP_FIND_LINKS="https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu121.html"
$PIP install -e '.[inference]'
./patching/hydra
echo "=== checkpoints ==="
if [ ! -e checkpoints/hf/pipeline.yaml ]; then
    $PIP install 'huggingface-hub[cli]<1.0'
    "$ENV/bin/hf" download --repo-type model --local-dir checkpoints/hf-download --max-workers 1 facebook/sam-3d-objects
    mv checkpoints/hf-download/checkpoints checkpoints/hf
    rm -rf checkpoints/hf-download
fi
"$ENV/bin/python" -c "import torch, pytorch3d, gsplat, kaolin, spconv; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'pytorch3d', pytorch3d.__version__)"
ls -la checkpoints/hf | head
echo "=== done ==="
