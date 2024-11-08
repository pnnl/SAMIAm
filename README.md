# SAM-I-Am zero-shot SAM segmentation for TEM images

This repository debuts the Segment Anything Model (SAM) with an external post-processor to deal with zero-shot TEM image segmentation. Please note that the codebase is still under development.

To run the provided examples, follow these steps:

1. Make sure to set up some of the hardcoded parameters, such as the dataset path, to match your environment.

2. Ensure that the model weights (".pth" files) are present in the repository.

3. Currently, the repository supports processing a batch of images in an unsupervised segmentation process.

4. The runnable Python scripts are provided in the "run.sh" file. You can execute them to run the segmentation process.

## To run with Slurm

If you're using Slurm for job scheduling, you can run the segmentation process with the following command:

```bash
python batch_unsupervised.py --sam=sam_vit_l_0b3195.pth --model=FENet --grid=16 --chipsize=60 --embed=1 --par=0 --dilation=1 --post=1
