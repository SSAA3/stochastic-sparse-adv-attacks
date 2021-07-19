"""
Generate adversarial samples against the chosen model and dataset, then save results in CSV files
"""

import os
import torch
import numpy as np
import pandas as pd
from attack.utils import PropagationCounter


def generate_attacks(model, dataset_loader, attack_arg, targeted_arg, sampling, max_batch, device, path, **kwargs):
    
    done = 0
    
    if kwargs['data'] == 'ImageNet':
        target = torch.tensor(np.load('./attack/Target_ImageNet.npy'), dtype=torch.int64).to(device)
    
    elif kwargs['data'] == 'CIFAR10':
        target = torch.tensor(np.load('./attack/Target_CIFAR10.npy'), dtype=torch.int64).to(device)
    
    else:
        raise ValueError('Invalid dataset argument')
    
    
    for batch_x, batch_y in dataset_loader:
        
        with torch.no_grad():
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            # Correctly predicted images
            correct_pred = torch.argmax(model(batch_x), dim=1).eq(batch_y)
            batch_x, batch_y = batch_x[correct_pred], batch_y[correct_pred]
            batch_size = batch_x.size()[0]
            
        if batch_size!=0:
            
            with torch.no_grad():
                
                if targeted_arg:
                    
                    batchsize = kwargs['batchsize']
                    labels_out = target[batchsize*done:batchsize*(done+1)][correct_pred]
                    
                    if attack_arg == 'FGA':
                        from attack.attacks.FGA_t import FGA_targeted
                        attack = FGA_targeted

                    elif attack_arg == 'VFGA':
                        from attack.attacks.VFGA_t import VFGA_targeted
                        attack = VFGA_targeted

                    elif attack_arg == 'UA':
                        from attack.attacks.UA_t import UA_targeted
                        attack = UA_targeted

                    else:
                        raise ValueError('Invalid attack argument')
                    
                
                else:
                    
                    labels_out = None

                    if attack_arg == 'FGA':
                        from attack.attacks.FGA_nt import FGA_untargeted
                        attack = FGA_untargeted

                    elif attack_arg == 'VFGA':
                        from attack.attacks.VFGA_nt import VFGA_untargeted
                        attack = VFGA_untargeted

                    elif attack_arg == 'UA':
                        from attack.attacks.UA_nt import UA_untargeted
                        attack = UA_untargeted

                    else:
                        raise ValueError('Invalid attack argument')

            forward_counter, backward_counter = PropagationCounter('forward'), PropagationCounter('backward')
            kwargs['model_without'].register_forward_hook(forward_counter)
            kwargs['model_without'].register_backward_hook(backward_counter)
            fwd, bwd = [], [] 
            
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            
            start.record()
            original, adversarial, label_in, label_out, dist, final_index = attack(model, batch_x, batch_y, labels_out=labels_out, nb_classes=kwargs['nb_classes'], 
                                                                                   max_iter=kwargs['max_iter'], sampling=sampling, device=device)
            end.record()
            torch.cuda.synchronize()
            time = (start.elapsed_time(end)) / 1000  # times are in milliseconds
            
            fwd.append(forward_counter.num_samples_called / len(batch_x))
            bwd.append(backward_counter.num_samples_called / len(batch_x))
            forward_counter.reset(), backward_counter.reset()
            
        
            with torch.no_grad():
                batch_in = pd.DataFrame(original.view(original.size()[0], -1).cpu().detach().numpy())
                batch_in.to_csv(os.path.join(path, 'Original_{}.csv'.format(done)), index=False)

                batch_out = pd.DataFrame(adversarial.view(adversarial.size()[0], -1).cpu().detach().numpy())
                batch_out.to_csv(os.path.join(path, 'Adversarial_{}.csv'.format(done)), index=False)

                Distances = pd.DataFrame(dist.cpu().detach().numpy(), columns=['L0', 'L1', 'L2'])
                Distances.to_csv(os.path.join(path, 'Distances_{}.csv'.format(done)), index=False)

                Labels =  pd.DataFrame({'Labels in' : label_in.cpu().type(torch.int64).detach().numpy(), 
                                        'Labels out': label_out.cpu().type(torch.int64).detach().numpy()})
                Labels.to_csv(os.path.join(path, 'Labels_{}.csv'.format(done)), index=False)
                
                Final_index = pd.DataFrame({'Final Index': final_index})
                Final_index.to_csv(os.path.join(path, 'Final_index_{}.csv'.format(done)), index=False)
                
                time_ = torch.tensor([time]).to(device)
                Time = pd.DataFrame(time_.cpu().detach().numpy())
                Time.to_csv(os.path.join(path, 'Time_{}.csv'.format(done)), index=False)
                
                Model_Prop = pd.DataFrame({'Forward' : fwd, 'Backward': bwd})
                Model_Prop.to_csv(os.path.join(path, 'Model_Prop_{}.csv'.format(done)), index=False)

            done += 1

            print('Done : {} / {}'.format(done, max_batch))
            
            
        else:
            pass

        if done==max_batch:
            break