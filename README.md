# [Stochastic Sparse Adversarial Attacks](https://arxiv.org/abs/2011.12423)

Manon Césaire, Hatem Hajri, Sylvain Lamprier, Patrick Gallinari\
*IRT SystemX, Sorbonne Université*\
\

This repository contains PyTorch implementation of untargeted FGA and VFGA algorithms.

## Requirements :
To execute the code, please install the packages required in the `requirements_pytorch.txt` file.

&nbsp;&nbsp;&nbsp;&nbsp;
&nbsp;&nbsp;&nbsp;&nbsp;

## How to execute the code :

The algorithms proposed in the paper can be run by using the script `start.py` :

`python start.py --cuda_device <cuda_device> --job <job> --attack <attack> --dataset <dataset> --model <model> --seed <seed> --targeted --sampling <sampling> --batch_size <batch_size> --max_batch <max_batch> --image_choice <image_choice>`  

where :  

- `cuda_device` either `cuda` or `cpu`, specifies CUDA device.  
- `job` either `attack`, `performances` or `visualization`, specifies the action to run.  
- `attack` either `FGA`, `VFGA` or `UA`, switches between the implementations of our attacks.  
- `dataset` either `ImageNet` or `CIFAR10`, selects the dataset on which the job will be performed.  
- `model` either `inception_v3` or `vgg16` when `dataset = ImageNet`  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; either `resnet18` or `vgg19` when `dataset = CIFAR10`  
selects the model on which the job will be performed.  
- `seed` : an integer that specifies the random seed to fix.
- `targeted` : an targeted attack will be executed when this argument is present and an untargeted attack otherwise.
- `sampling` : an integer that specifies the number of samples to generate at each step following the paper.  
- `batch_size` : an integer that specifies the input batch size for processing in parallel.  
- `max_batch` : an integer that specifies how many batches will be performed.  
- `image_choice` : either `best`, `worst` or `random`, switches between images to visualize.  

&nbsp;&nbsp;&nbsp;&nbsp;
&nbsp;&nbsp;&nbsp;&nbsp;

## Job examples :

### Generation of Adversarial Samples

- To generate untargeted FGA adversarial samples against a the trained ResNet18 model over the CIFAR-10 dataset with $`N_S = 10`$ :  

`python start.py --cuda_device "cuda" --job "attack" --attack "FGA" --dataset "CIFAR10" --model "resnet18" --sampling 10 --batch_size 50 --max_batch 100`  

- To generate targeted VFGA adversarial samples against the pre-trained Inception-v3 model over the the ImageNet dataset with $`N_S = 10`$ :  

`python start.py --cuda_device "cuda" --job "attack" --attack "VFGA" --dataset "ImageNet" --model "inception_v3" --targeted --sampling 10 --batch_size 10 --max_batch 100`  

### Analyse Attack Performances 

- To print out the performances of the untargeted FGA attack against the trained ResNet18 model over the the CIFAR-10 dataset with $`N_S = 10`$ :  

`python start.py --cuda_device "cuda" --job "performances" --attack "FGA" --dataset "CIFAR10" --model "resnet18"  --sampling 10 --batch_size 50`

- To print out the performances of the targeted VFGA attack against the pre-trained Inception-v3 model over the the ImageNet dataset with $`N_S = 10`$ :  

`python start.py --cuda_device "cuda" --job "performances" --attack "VFGA" --dataset "ImageNet" --model "inception_v3" --targeted --sampling 10 --batch_size 10`  

### Visualization of Adversarial Samples  

- To display the adversarial sample and its original sample with the best $`L_0`$ score :

`python start.py --cuda_device "cuda" --job "visualization" --attack "VFGA" --dataset "ImageNet" --model "inception_v3" --targeted --sampling 10 --image_choice "best"`  

- To show the adversarial sample and its original sample with a random $`L_0`$ score :  

`python start.py --cuda_device "cuda" --job "visualization" --attack "VFGA" --dataset "ImageNet" --model "inception_v3" --targeted --sampling 10 --image_choice "random"`  

&nbsp;&nbsp;&nbsp;&nbsp;
&nbsp;&nbsp;&nbsp;&nbsp;

## CSV File Structure of the Adversarial Samples :

### Generation

When generating adversarial attacks, a `"results_Name"` folder will be created with `"Name"` the name of the used attack and the category of the used attack (targeted or untargeted), then a sub-folder containing the name of the dataset and the used model. Finally, a last sub-folder of the type `"sampling_NS"` will be created with `"NS"` the size of the generated samples. 

Several CSV files will be created during the generation. There will be as many `"Adversarial_i.csv"`, `"Original_i.csv"``"Labels_i.csv"` and `"Time_i.csv"` files as `"max_batch"`.  

- Each `"Labels_i.csv"` file will contain two columns, the first with the original image label and the second with the adversarial image label.  
- Each `"Original_i.csv"` has dimension `"batch_size"` $`\times`$ the image size (the image size is defined as the total number of pixels). Each row $`j`$ of `"Original_i.csv"` will contain the original image $`j`$ of the batch $`i`$.  
- Each `"Adversarial_i.csv"`  has dimension `"batch_size"` $`\times`$ the image size (the image size is defined as the total number of pixels). Each row $`j`$ of `"Adversarial_i.csv"` will contain the adversarial image $`j`$ of the batch $`i`$.  

### Performances

When analyzing attack performances, a sub-folder `"results"` will be created inside the generation folder. Several CSV files will be recorded :  

- `"Distances.csv"` containing all distances for each batch.  
- `"Time.csv"` containing the run time for each batch.  
- `"Avg_Time.csv"` containing the average time to generate an adversarial sample.  
- `"Performances.csv"` containing  the success rate SR and average distances for the generated adversarial samples.  
- `"Best_Worst.csv"` containing the best and worst $`L_0`$ score among the generated adversarial samples.  

### Visualization  

When visualizing adversarial samples, there are several choices :  

- `"best"` : visualization of the adversarial sample which obtained the best $`L_0`$ score.  
- `"worst"` : visualization of the adversarial sample which obtained the worst $`L_0`$ score.  
- `"random"` : visualization of a random adversarial sample together with its $`L_0`$ score.

The adversarial and original samples will be saved in the `"results"`files in .PNG format.

&nbsp;&nbsp;&nbsp;&nbsp;
&nbsp;&nbsp;&nbsp;&nbsp;

## Remarks and Precautions :  

- This library is implemented to run by batch of images.   
    
- If you decide to generate several times the same attack against the same dataset over the same dataset, previous results will be overwritten too. To avoid that, you only need to rename the folder.  
    
- If you decide to visualize several times an adversarial sample with the parameter \verb "random" \, this image will be overwritten. To avoid that, you only need to rename the image.  

&nbsp;&nbsp;&nbsp;&nbsp;
&nbsp;&nbsp;&nbsp;&nbsp;

## Run Time /Model Propagation Computation :  

If you want to see the run time comparison, you should generate adversarial samples with `batch_size=1` and `max_batch=1,000`, then compute the performances in order to have the average run time comparison for $`1,000`$ images. For example:  

`python start.py --cuda_device "cuda" --job "attack" --attack "VFGA" --dataset "ImageNet" --model "inception_v3" --train_or_test "test" --sampling 10 --batch_size 1 --max_batch 1000`  

`python start.py --cuda_device "cuda" --job "performances" --attack "VFGA"  --dataset "ImageNet" --model "inception_v3" --train_or_test "test" --sampling 10 --batch_size 1` 

&nbsp;&nbsp;&nbsp;&nbsp;
&nbsp;&nbsp;&nbsp;&nbsp;

## Reference :

[1] H. Hajri, M. Césaire, T. Combey, S. Lamprier and P. Gallinari. Stochastic sparse adversarial attacks. _arXiv preprint arXiv:2011.12423_, 2020.
