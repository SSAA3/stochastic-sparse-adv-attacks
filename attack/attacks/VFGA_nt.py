"""
Voting folded gaussian attack (VFGA) algorithm
"""

import os, sys
sys.path.append(os.path.abspath('../'))

import torch
from attack.utils import compute_gradients
from attack.Distances import compute_distances


def VFGA_untargeted(model, input_, labels, labels_out, nb_classes, clip_min=0., clip_max=1., max_iter=5, sampling=10, device=-1):
    
    """
    Non-Targeted Voting Folded Gaussian Attack

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
            saliency_plus = torch.mul((clip_max - input_reshaped), jacobian)
            saliency_plus += (1 - search_domain) * input_size * torch.max(torch.abs(saliency_plus), axis=1).values.view(-1, 1)
            
            saliency_minus = torch.mul((input_reshaped - clip_min), jacobian)
            saliency_minus -= (1 - search_domain) * input_size * torch.max(torch.abs(saliency_minus), axis=1).values.view(-1, 1)

            # Finding the two most vulnerable features
            p_plus = torch.argmin(saliency_plus, dim=1)
            p_onehot_plus = torch.FloatTensor(nb_input, input_size).to(device)
            p_onehot_plus = p_onehot_plus.zero_().scatter_(1, p_plus.view(-1, 1), 1)
            
            p_minus = torch.argmax(saliency_minus, dim=1)
            p_onehot_minus = torch.FloatTensor(nb_input, input_size).to(device)
            p_onehot_minus = p_onehot_minus.zero_().scatter_(1, p_minus.view(-1, 1), 1)

            # Running conditions of non-targeted FGA :
            #    - Initial class = Current class
            #    - Number of modified features >= 2
            cond = torch.sum(target_onehot * pred_onehot, axis=1).eq(1) & (torch.ge(torch.sum(search_domain, axis = 1), 2))
            cond_float = cond.type(torch.float32).view(-1, 1)
            
            # Applying gaussian noise to the two most vulnerable pixels
            values_plus = torch.sum(p_onehot_plus * input_reshaped, axis=1).view(-1, 1)
            values_plus = ((clip_max - clip_min) * (clip_max - values_plus)) * torch.abs(torch.randn(nb_input, sampling, device=device))
            values_plus = p_onehot_plus.view(nb_input, 1, -1).expand(nb_input, sampling, -1) * (values_plus * cond_float).view(nb_input, sampling, 1)
            
            values_minus = torch.sum(p_onehot_minus * input_reshaped, axis=1).view(-1, 1)
            values_minus = ((clip_max - clip_min) * (values_minus - clip_min)) * torch.abs(torch.randn(nb_input, sampling, device=device))
            values_minus = p_onehot_minus.view(nb_input, 1, -1).expand(nb_input, sampling, -1) * (values_minus * cond_float).view(nb_input, sampling, 1)

            # Modifications of initial inputs for increasing perturbations
            x_modif_plus = input_reshaped.view(nb_input, 1, -1).expand(nb_input, sampling, -1) + values_plus
            leq_clipmax = (x_modif_plus <= clip_max).prod(2)
            
            ## Keeping only 1 image per sample where pixels > clip_max
            x_modif_plus = x_modif_plus.view(-1, depth, width, heigth)           
            ind2_plus = torch.arange(x_modif_plus.size()[0]).to(device)
            ind2_plus = torch.where(leq_clipmax.view(-1).bool(), 0, ind2_plus).view(nb_input, sampling)
            imax_plus = ind2_plus.argmax(1).view(-1)
            leq_clipmax[range(len(leq_clipmax)), imax_plus] = 1
            x_modif_plus = torch.clip(x_modif_plus, clip_min, clip_max) 
            
            # Modifications of initial inputs for decreasing perturbations
            x_modif_minus = input_reshaped.view(nb_input, 1, -1).expand(nb_input, sampling, -1) - values_minus
            geq_clipmin = (x_modif_minus >= clip_min).prod(2)
            
            ## Keeping only 1 image per sample where pixels < clip_min
            x_modif_minus = x_modif_minus.view(-1, depth, width, heigth)           
            ind2_minus = torch.arange(x_modif_minus.size()[0]).to(device)
            ind2_minus = torch.where(geq_clipmin.view(-1).bool(), 0, ind2_minus).view(nb_input, sampling)
            imax_minus = ind2_minus.argmax(1).view(-1)
            geq_clipmin[range(len(geq_clipmin)), imax_minus] = 1
            x_modif_minus = torch.clip(x_modif_minus, clip_min, clip_max) 

            x_modif = torch.cat([x_modif_plus, x_modif_minus], dim=1).view(2 * sampling * nb_input, -1)
            x_modif = x_modif.view(-1, depth, width, heigth)

            # Computing the next probabilities and next labels
            next_proba_plus = torch.ones(nb_input, sampling, nb_classes, device=device).view(-1, nb_classes)
            next_proba_plus[leq_clipmax.view(-1).bool()] =  model(x_modif_plus[leq_clipmax.view(-1).bool()])
            next_proba_plus = next_proba_plus.view(nb_input * sampling, nb_classes)

            next_proba_minus = torch.ones(nb_input, sampling, nb_classes, device=device).view(-1, nb_classes)
            next_proba_minus[geq_clipmin.view(-1).bool()] =  model(x_modif_minus[geq_clipmin.view(-1).bool()])
            next_proba_minus = next_proba_minus.view(nb_input * sampling, nb_classes)
            
            next_proba = torch.cat([next_proba_plus, next_proba_minus], dim=1).view(nb_input, 2 * sampling, nb_classes)
            next_pred = torch.argmax(next_proba, axis=2).type(torch.int64)

            # Finding the best modifications
            target_proba = torch.transpose(next_proba, 2, 1)
            target_proba = (target_proba * target_onehot.view(nb_input, nb_classes, 1)).sum(axis=1)
            best_ind = torch.argmin(target_proba, axis=1).type(torch.int64)

            # Updating the best modifications
            input_ = x_modif.view(-1, 2 * sampling, input_size)[range(nb_input), best_ind.long(), :]
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
            search_domain -= (p_onehot_plus * torch.logical_not(best_ind.ge(sampling)).view(-1, 1) + p_onehot_minus * best_ind.ge(sampling).view(-1, 1)) * cond_float
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
