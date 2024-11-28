## Using ROCm based GPUs <a id="rocm"></a>
Background: The CI has 10 ROCm nodes each of which have 8x  AMD Instinct MI50(32G) GPU cards. These gpus are built upon older chip architecture 'gfx906' which restricts these nodes ROCm software version to 5.7, in addition most of the deep learning (DL) models/pipelines are developed 
for CUDA (NVIDIA) hardware. These two factors make the usage of these nodes/gpus a challenge and and consequently, these nodes are underutilised and often available. 
There are two ways (that I found) of using these nodes for DL model training.

### Using a conda env 
Start the pre-made pytorch conda environment called 'rocm_gpus' that exist in scratchc- using:\
```conda activate /mnt/scratchc/ralab/software/rocm_gpus```\
if you want to create this environment from the scratch please see the recipy.\
You're now ready to train pytorch models either in interactive mode (jupyter notebook) or by submitting Slurm sbatch.sh job as on CUDA but using **a single gpu!** 
#### Training on multiple GPUs
Training models on multiple GPUs require each training batch to be split across GPUs and corresponding gradients to be syncronised. There are different ways of doing this, one of which is using [lightning ai](https://lightning.ai/) package. Below is a describtion of adapting your pytorch code for this including a couple links to toy examples.\
lightning is a wrapper for pytorch's nn.module that simplify and abstract the process of training models. Adapting your pytorch code to lightning can be done in varying levels. Below we describe two extrem levels i.e. minimal and complete lightning adapted.In either case there are five places within your model training that require changes.\

[1] Model:\
Minimal - You can adapt your model to lightining by changing just one thing- most pytorch models are made by extending nn.Module, the only change required is using lightning.LightningModule instead of nn.Module i.e. swap nn.Module with L.LightningModule in the code.\
Complete - To leverage full abstraction functionalities of lightning your pytorch model can be defined as class that can have following methods
- [constructor( )] a constructor function that intialise all parameters/variables
- [forward( )] This define forward pass of the model through all layers and blocks.
- [training_step( )] This defines what should happen in a single training iteration (batch).
- [configure_optimiser( )] this defines optimiser and its parameters.
- [validation_step( )] this is same as training_step( ) but for validation.




[2] Data:  
Minimal - You can train your models without making any change and feed the lightning adapted model with standard train_dataloader, validation_dataloader. 
Complete- But if you do want to use lightning's data abstraction functionalities, you need to create a 'DataModule' class by extending lightning.LightningDataModule class, within there can be methods/functions for following tasks
- [constructor( )] a constructor function that intialise all parameters/variables
- [prepare_data( )] Used for any data-related setup that happens only once, such as loading raw datasets or preprocessing raw data.This is because each GPU will execute the same PyTorch thereby causing duplication. ALL of the code in Lightning makes sure the critical parts are called from ONLY one GPU
- [setup_data( )] Handles any data-related logic that might be dependent on the state of the current process or GPU, such as splitting the data set into train, validation, and test sets, or applying specific transformations.
- [train_dataloader( )] this method allow you to define train loader parameters such as batch size, pin_memory, shuffle etc.
- [validation_dataloader( )] this method allow you to define validation loader parameters such as batch size, pin_memory etc.

[3] Loss:

[4] Optimiser:

[5] Slurm sbatch: 










