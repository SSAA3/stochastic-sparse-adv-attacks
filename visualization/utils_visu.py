"""
Useful functions to visualize adversarial samples
"""

import os, sys
sys.path.append(os.path.abspath('../'))

import glob
import torch
import pandas as pd

from attack.Distances import l0_distance
import matplotlib.pyplot as plt


def where_best_worst(path):
    """
    Finding the best L0 score and worst L0 score adversarial samples.
    """
    
    path_result = os.path.join(path, 'results')
    best_worst = pd.read_csv(os.path.join(path_result, 'Best_Worst.csv'))
    distance_files = glob.glob(os.path.join(path, '*{}*.csv'.format('Distances')))

    best_index = worst_index = - 1

    for count, file in enumerate(distance_files):
        l0_values = torch.tensor(pd.read_csv(file).values)[:, 0]

        if float(best_worst['Best L0'].values) in l0_values and best_index==-1:
            best_index = count

        if float(best_worst['Worst L0'].values) in l0_values and worst_index==-1:
            worst_index = count


        if best_index!=-1 and worst_index!=-1:
            break
            
    return best_index, worst_index



def class_dataset(dataset):
    """
    Conversion class id dataset to readable labels.
    """
    
    if dataset=='ImageNet':
        
        dict_class = {}
        Classes = open("./visualization/imagenet1000_clsidx_to_labels.txt")
        
        for line in Classes:
            foo = line.split("':")[0].split(':')
            dict_class[int(foo[0])] = foo[1].split(",\n")[0].split("'")[1]
        

    elif dataset=='CIFAR10':
        
        dict_class = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 
                      5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship', 9: 'truck'}
        
        
    else:
        raise ValueError('Invalid dataset argument')
        
    return dict_class
        

    
def inv_channel(image, **kwargs):
    """
    Inversion of channels in order to visualize images.
    """
    
    image = image.type(torch.float32)
    
    for i in range(len(kwargs['mean'])):

        image[i] = torch.mul(image[i], kwargs['std'][i])
        image[i] = torch.add(image[i], kwargs['mean'][i])
    
    image = image.transpose(0, 2).transpose(0, 1)
    return image  



def input_size(model, dataset):
    """
    Input size of dataset.
    """
    
    if dataset=='ImageNet':
        
        if model=='inception_v3':
            input_size = 299
            
        elif model=='vgg16':
            input_size = 224
        
        else:
            raise ValueError('Invalid model argument')
        
    elif dataset=='CIFAR10':
        input_size = 32
    
    else:
        raise ValueError('Invalid dataset argument')
    
    return input_size



def orig_adv_sample(model, dataset, image_choice, path):
    """
    Allow to save original and aversarial samples of the best, worst or random sample among the dataset.
    """
    
    
    inputsize = input_size(model, dataset)    
    
    if image_choice=='best':
        
        best_index = where_best_worst(path)[0]
        
        original = torch.tensor(pd.read_csv(os.path.join(path, 'Original_{}.csv'.format(best_index))).values)[0].view(3, inputsize, inputsize)
        adversarial = torch.tensor(pd.read_csv(os.path.join(path, 'Adversarial_{}.csv'.format(best_index))).values)[0].view(3, inputsize, inputsize)
        labels = pd.read_csv(os.path.join(path, 'Labels_{}.csv'.format(best_index))).values[0]
    
    
    elif image_choice=='worst':
        
        best_index = where_best_worst(path)[1]
        
        original = torch.tensor(pd.read_csv(os.path.join(path, 'Original_{}.csv'.format(worst_index))).values)[-1].view(3, inputsize, inputsize)
        adversarial =  torch.tensor(pd.read_csv(os.path.join(path, 'Adversarial_{}.csv'.format(worst_index))).values)[-1].view(3, inputsize, inputsize)
        labels = pd.read_csv(os.path.join(path, 'Labels_{}.csv'.format(worst_index))).values[-1]
    
    
    elif image_choice=='random':
        
        path_result = os.path.join(path, 'results')
        if not os.path.exists(path_result):
                os.mkdir(path_result)
        
        distance_files = glob.glob(os.path.join(path, '*{}*.csv'.format('Distances')))
        random_file = int(torch.randint(len(distance_files), (1,)))
        random_index = torch.randint(pd.read_csv(distance_files[random_file]).shape[0], (1,))
        
        original = torch.tensor(pd.read_csv(os.path.join(path, 'Original_{}.csv'.format(random_file))).values)[random_index].view(3, inputsize, inputsize)
        adversarial =  torch.tensor(pd.read_csv(os.path.join(path, 'Adversarial_{}.csv'.format(random_file))).values)[random_index].view(3, inputsize, inputsize)
        labels = pd.read_csv(os.path.join(path, 'Labels_{}.csv'.format(random_file))).values[random_index]
    
    
    else:
        raise ValueError('Invalid index choice argument')
        
    return original, adversarial, labels



def proba_images(model, dataset, arg_model, image_choice, path, device):
    """
    Compute probabilities for given images.
    """
    
    inputsize = input_size(arg_model, dataset)
    original, adversarial, labels = orig_adv_sample(arg_model, dataset, image_choice, path)
    
    original = original.to(device)
    adversarial = adversarial.to(device)
    
    init_proba = model(original.view(1, 3, inputsize, inputsize).float())
    end_proba = model(adversarial.view(1, 3, inputsize, inputsize).float())
    
    prob_init_orig = init_proba[0, labels[0]]
    proba_end_orig = end_proba[0, labels[0]]
    proba_end_adv = end_proba[0, labels[1]]
    
    return prob_init_orig, proba_end_orig, proba_end_adv



def plot_samples(model, dataset, arg_model, image_choice, path, device, **kwargs):
    """
    Plot original and adversarial samples.
    """
    
    path_result = os.path.join(path, 'results')
    
    original, adversarial, labels = orig_adv_sample(arg_model, dataset, image_choice, path)
    prob_init_orig, proba_end_orig, proba_end_adv = proba_images(model, dataset, arg_model, image_choice, path, device)
    
    pix_nb = int(l0_distance(original.view(1, -1), adversarial.view(1, -1)))
    
    dict_class = class_dataset(dataset)
    
    plt.figure(figsize=(12, 6))

    plt.subplot(121)
    plt.imshow(inv_channel(original,  **kwargs))
    plt.title('Orginal Sample : \n\n\n {} : {:.5f}'.format(dict_class[labels[0]], prob_init_orig))
    plt.axis('off')

    plt.subplot(122)
    plt.imshow(inv_channel(adversarial,  **kwargs))
    plt.title('Adversarial Sample : {} pixel(s) \n\n {} : {:.5f} \n {} : {:.5f}'.format(pix_nb, dict_class[labels[1]], proba_end_adv,
                                                                                        dict_class[labels[0]], proba_end_orig,))
    plt.axis('off')

    plt.savefig(os.path.join(path_result, '{}_{}_{}.PNG'.format(image_choice, dataset, arg_model)))
    plt.show()
    
    print('Plot saved !')