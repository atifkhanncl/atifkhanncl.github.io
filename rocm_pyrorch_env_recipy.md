## The ROCm envioment recipy specific to our nodes with AMD Instinct MI50(32G) GPU cards
```
pip3 cache purge
conda create --name rocm_gpus python=3.10
conda activate rocm_gpus
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/rocm5.6
wget https://raw.githubusercontent.com/wiki/ROCm/pytorch/files/install_kdb_files_for_pytorch_wheels.sh
chmod 777 ./install_kdb_files_for_pytorch_wheels.sh

# replace 'gfx906' with your architecture and 5.7with your preferred ROCm version
export GFX_ARCH=gfx906
export ROCM_VERSION=5.7
./install_kdb_files_for_pytorch_wheels.sh
export HSA_FORCE_FINE_GRAIN_PCIE=1

conda install notebook ipykernel
ipython kernel install --user --name rocm_gpus --display-name "rocm_gpus"

```
