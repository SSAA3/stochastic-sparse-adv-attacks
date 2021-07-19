"""
Folded gaussian attack (FGA) algorithm
"""

import os, sys
sys.path.append(os.path.abspath('../'))

import torch
from attack.utils import compute_gradients
from attack.Distances import compute_distances


def FGA_untargeted(model, input_, labels, labels_out, nb_classes, clip_min=0., clip_max=1., max_iter=5, sampling=10, device=-1):
    
    """
    Non-Targeted Folded Gaussian Attack

    :param model: An instance of torchvision.models class.
    :param input_: The input tensor.
    :param labels: The label tensor.
    :param nb_classes: The classes number of inputs.
    :param clip_min: Minimum float value for features.
    :param clip_max: Maximum float value for features.
    :param sampling: Size of the gaussian sample.
    :param max_iter: Maximum number of perturbed features.
    :param device: Speficies CUDA device.

    :returns:
        The adversarial sample to the input tensor.
        The Label output tensor.
        The Number of iterations for each input.
    """
    
    input_.requires_grad = True
    nb_input, depth, width, heigth = input_.size()
    input_size = width * heigth * depth

    search_domain = torch.ones_like(input_).view(-1, input_size)
    
    iter_ = torch.tensor([0]).to(device)
    cond = torch.tensor(True).to(device)
    
    # Initialize tensors to stock results
    stock_init_input, stock_adv_input = torch.tensor([]).to(device), torch.tensor([]).to(device)
    stock_init_label, stock_adv_label = torch.tensor([]).to(device), torch.tensor([]).to(device)
    
    init_input = torch.autograd.Variable(input_.clone()).to(device)
    Dist = torch.tensor([]).to(device)
    
    initial_index = torch.arange(nb_input)
    final_index = []
    
    # The initial labels of inputs are chosen as the targets
    target = labels
    
    while torch.logical_and(torch.le(iter_, torch.tensor([max_iter]).to(device)), cond):
        
        with torch.no_grad():
            # One hot for target
            target_onehot = torch.FloatTensor(nb_input, nb_classes).to(device)
            target_onehot = target_onehot.zero_().scatter_(1, target.view(-1, 1), 1)
    
            # Class of the current input : C
            pred = torch.argmax(output, dim=1) if (iter_>0) else labels
            pred_onehot = torch.FloatTensor(nb_input, nb_classes).to(device)
            pred_onehot = pred_onehot.zero_().scatter_(1, pred.view(-1, 1), 1)

            input_reshaped = input_.view(-1, input_size)
        
        # Computing the gradients of the probability function p_c with respect to the inputs
        jacobian = compute_gradients(model, input_, pred)
        
        with torch.no_grad():
            jacobian = jacobian.view(-1, input_size)
            saliency = torch.mul((clip_max - input_reshaped), jacobian)
            saliency += (1 - search_domain) * input_size * torch.max(torch.abs(saliency), axis=1).values.view(-1, 1)

            # Finding the most vulnerable feature
            p = torch.argmin(saliency, dim=1)
            p_onehot = torch.FloatTensor(nb_input, input_size).to(device)
            p_onehot = p_onehot.zero_().scatter_(1, p.view(-1, 1), 1)

            # Running conditions of non-targeted FGA :
            #    - Initial class = Current class
            #    - Number of modified features >= 2
            cond = torch.sum(target_onehot * pred_onehot, axis=1).eq(1) & (torch.ge(torch.sum(search_domain, axis = 1), 2))
            cond_float = cond.type(torch.float32).view(-1, 1)
            
            # Applying gaussian noise to the most vulnerable pixel
            values = torch.sum(p_onehot * input_reshaped, axis=1).view(-1, 1)
            values = ((clip_max - clip_min) * (clip_max - values)) * torch.abs(torch.randn(nb_input, sampling, device=device))
            values = p_onehot.view(nb_input, 1, -1).expand(nb_input, sampling, -1) * (values * cond_float).view(nb_input, sampling, 1)
            
            x_modif = input_reshaped.view(nb_input, 1, -1).expand(nb_input, sampling, -1) + values
            
            # Keeping images where pixels <= clip_max
            leq_clipmax = (x_modif <= clip_max).prod(2)
            
            # Keeping only 1 image per sample where pixels > clip_max
            x_modif = x_modif.view(-1, depth, width, heigth)           
            ind2 = torch.arange(x_modif.size()[0]).to(device)
            ind2 = torch.where(leq_clipmax.view(-1).bool(), 0, ind2).view(nb_input, sampling)
            imax = ind2.argmax(1).view(-1)
            leq_clipmax[range(len(leq_clipmax)),imax] = 1
            
            # In leq_clipmax, we have 1 for the images on which we want to apply model, 0 for the others
            # Computing the next probabilities and next labels
            leq_clipmax = leq_clipmax.view(-1).to(device)
            next_proba = torch.ones(nb_input, sampling, nb_classes, device=device).view(-1, nb_classes)
            
            next_proba[leq_clipmax.bool()] =  model(x_modif[leq_clipmax.bool()])
            next_proba = next_proba.view(nb_input, sampling, nb_classes)            
            next_pred = torch.argmax(next_proba, axis=2).type(torch.int64)

            # Finding the best modifications
            target_proba = torch.transpose(next_proba, 2, 1)
            target_proba = (target_proba * target_onehot.view(nb_input, nb_classes, 1)).sum(axis=1)
            best_ind = torch.argmin(target_proba, axis=1).type(torch.int64)

            # Updating the best modifications            
            input_ = x_modif.view(-1, sampling, input_size)[range(nb_input), best_ind.long(), :]
            input_ = input_.view(-1, depth, width, heigth)
            input_ = torch.autograd.Variable(input_, requires_grad=True)

            # Updating the next labels
            next_pred_out = next_pred[range(nb_input), best_ind.long()]
            next_pred_onehot = torch.FloatTensor(nb_input, nb_classes).to(device)
            next_pred_onehot = next_pred_onehot.zero_().scatter_(1, next_pred_out.view(-1, 1), 1)

            # Updating the next running conditions
            notdone = torch.sum(target_onehot * next_pred_onehot, axis=1).eq(1) & (torch.ge(torch.sum(search_domain, axis = 1), 2)) 
            cond = torch.any(notdone)

            iter_ = torch.add(iter_, 1)
            search_domain -= p_onehot * cond_float
            search_domain = search_domain[notdone]

            ## Stock : 
            done = torch.logical_not(notdone)
            if torch.any(done):
                Dist = torch.cat([Dist, compute_distances(input_[done], init_input[done], done.sum())], dim=0)
                
                stock_init_input = torch.cat([stock_init_input, init_input[done]])
                stock_adv_input = torch.cat([stock_adv_input, input_[done]])
                stock_init_label = torch.cat([stock_init_label, target[done]])
                stock_adv_label = torch.cat([stock_adv_label, next_pred_out[done]])
                
                final_index += initial_index[done].tolist()
                initial_index = initial_index[notdone]

            if torch.any(notdone):
                input_ = torch.autograd.Variable(input_[notdone], requires_grad=True)
                output = model(input_)
                nb_input = torch.sum(notdone)
            
                init_input = init_input[notdone]
                target = target[notdone]

    return stock_init_input, stock_adv_input, stock_init_label, stock_adv_label, Dist, final_index
