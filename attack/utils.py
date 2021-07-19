"""
Useful functions to use attacks and generate adversarial samples
"""

import numpy as np
import torch, torchvision
import torchvision.transforms as transforms
import torchvision.models as models


def model_with_softmax(model):
    
    """
    Computing the softmax activation function to the neural network's output.

    :param model: An instance of torchvision.models class.

    :return:
        The softmax output tensor.
    """
    
    return lambda x: torch.nn.functional.softmax(model(x), dim=1)


def normalization(input_, mean, std, device):
    
    """
    Computing Theta_Max and Theta_Min vector afin normalization of dataset

    :param input_: The input tensor.
    :param mean: Mean vector of normalization.
    :param std: Std vector of normalization.
    :param device: Speficies CUDA device.

    :returns:
        The normalized input.
    """
    
    Mean = torch.tensor(mean).view(3, 1, 1).to(device)
    Std = torch.tensor(std).view(3, 1, 1).to(device)
    
    Input = (input_ - Mean) / Std
    
    return Input


def model_with_normalization(model, mean, std, device):
    
    """
    Computing the normalization function to input before to apply the neural network.

    :param model: An instance of torchvision.models class.
    :param mean: Mean vector of normalization.
    :param std: Std vector of normalization.
    :param device: Speficies CUDA device.

    :return:
        The output tensor with normalized inputs.
    """
    
    return lambda x: model(normalization(x, mean, std, device))



def compute_gradients(model, input_, target_class, Hessian=False):
    
    """
    Computing the gradients of a function with respect to given inputs.

    :param model: An instance of torchvision.models class.
    :param input_: The input tensor.
    :param target_class: Label to target for the gradient computation.

    :return:
        The gradient tensor.
    """
    
    assert input_.requires_grad
    
    target_class=target_class.view(-1)
    
    output = model(input_)[range(target_class.size()[0]), target_class]
    Grads = torch.autograd.grad(output, input_, grad_outputs=torch.ones_like(output), create_graph=True)
    
    if Hessian:
        Hess = torch.autograd.grad(Grads[0], input_, grad_outputs=torch.ones_like(Grads[0]))
        return Hess[0]
    
    else:
        return Grads[0]


def random_classes(labels, nb_classes, nb_input, device):
    
    """
    Computing a random targeted class chosen for each input.

    :param labels: The labels of inputs.
    :param nb_classes: Number of classes for the dataset.
    :param nb_input: Number of inputs considered.

    :return:
        The random targeted class chosen to reach.
    """
    
    index_list = torch.arange(nb_classes).repeat(nb_input,1).to(device)
    available_indexes = (index_list != labels.view(-1, 1))
    choice_list = index_list[available_indexes].view(nb_input, -1)
    y_target = torch.tensor([np.random.choice(choice_list[i].cpu().numpy()) for i in range(nb_input)]).to(device)
    
    return y_target


class PropagationCounter:
    
    # see https://github.com/jeromerony/adversarial-library/blob/cd4f2bc8fd5b5116409f8d249b1a46702889ca0d/adv_lib/utils/utils.py
    
    def __init__(self, type: str):
        self.type = type
        self.reset()

    def __call__(self, *args, **kwargs):
        if self.type == 'forward':
            batch_size = len(args[1][0])
        elif self.type == 'backward':
            batch_size = len(args[1][1])
        self.num_samples_called += batch_size

    def reset(self):
        self.num_samples_called = 0