# Lossy Image Compression with Conditional Diffusion Models

This repository contains the codebase for our paper on Lossy Image Compression with Conditional Diffusion Models. We provide an off-the-shelf test code for both x-parameterization and epsilon-parameterization.

## Usage

- There are two separate folders, `epsilonparam` and `xparam`, for the epsilon-parameterization and x-parameterization models, respectively. Please use the appropriate folder depending on the model you want to work with. This is because there are some minor differences between the x-param and e-param models, making them incompatible with each other.
- Before running the test code, please read the comments about the arguments in the code to ensure proper usage.
- Note that this test code is provided as a template (with a sanity check example), you may need to write your own dataloader to get the actual results.

## Model Weights

The model weights can be downloaded from [this link](https://drive.google.com/drive/folders/197Wl5cwjaCvrEvggMcyNeHOSxq2rDZ1F?usp=sharing).
- Why the x-param weights are approximately twice as large as the epsilon-param weights? For the x-parameterization, I saved both the exponential moving average (ema) and the latest model. You can ignore the ema model and use the off-the-shelf test code to load the checkpoint.

Please feel free to explore the code and experiment with the models. If you have any questions or encounter any issues, don't hesitate to reach out to us.