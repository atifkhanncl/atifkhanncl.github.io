## Using ROCm based GPUs <a id="rocm"></a>
Background: The CI has 10 ROCm nodes each of which have 8x  AMD Instinct MI50(32G) GPU cards. These gpus are built upon older chip architecture 'gfx906' which restricts these nodes ROCm software version to 5.7, in addition most of the deep learning (DL) models/pipelines are developed 
for CUDA (NVIDIA) hardware. These two factors make the usage of these nodes/gpus a challengeand and consequently, these nodes are underutilised and often available. 
There are two ways (that I found) of using these nodes for DL model training.

### Using a conda env 
Start the pre-made conda environment called 'rocm_gpus' that exist in scratchc- using:\
```conda activate /mnt/scratchc/ralab/software/rocm_gpus```\
if you want to create this environment from the scratch please see the recipy.\
You're now ready to train models either in interactive mode (jupyter notebook) or by submitting Slurm sbatch.sh job as on CUDA but using **single gpu!** 
#### Training on multiple GPUs










