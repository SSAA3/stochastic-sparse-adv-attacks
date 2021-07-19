"""
Compute of the performances of the attacks for a specific model and dataset
"""

import os, glob
import torch
import argparse
import pandas as pd


def get_merged_csv(flist, **kwargs):
    return pd.concat([pd.read_csv(f, **kwargs) for f in flist], ignore_index=True)


def compute_performances(arg_model, dataset, arg_attack, targeted_arg, sampling, batch_size, **kwargs):
    
    path = os.path.join('results_{}_{}'.format(arg_attack, targeted_arg), 
                        '{}_{}'.format(dataset, arg_model), 
                        'sampling_{}'.format(sampling))
    
    path_result = os.path.join(path, 'results')
    if not os.path.exists(path_result):
            os.mkdir(path_result)
            
    
    for x in ['Distances', 'Time', 'Model_Prop']:
        fmask = os.path.join(path, '*{}*.csv'.format(x))

        df = get_merged_csv(glob.glob(fmask), index_col=None)
        df.to_csv(os.path.join(path_result, '{}.csv'.format(x)), index=False)
    
    
    Distance = pd.read_csv(os.path.join(path_result, 'Distances.csv'))
    Time = pd.read_csv(os.path.join(path_result, 'Time.csv'))
    Model_Prop = pd.read_csv(os.path.join(path_result, 'Model_Prop.csv'))
    
    
    # Compute distances performances
    performances = pd.DataFrame(Distance.mean(axis=0)).transpose() 
    performances = performances.rename(columns=lambda s: 'Avg_' + s, index=lambda s:'{}_{}'.format(dataset, arg_model))
    
    performances['Med_L0'] = Distance['L0'].median()
    
    performances['SR'] = (Distance['L0'] < kwargs['max_iter']).sum() / Distance['L0'].count() * 100
    performances = performances.reindex(columns = ['SR', 'Avg_L0', 'Med_L0', 'Avg_L1', 'Avg_L2'])
    performances.to_csv(os.path.join(path_result, 'Performances.csv'), index=False)
    
    print('-------- Performances of {} - {} attack --------'.format(arg_attack, targeted_arg))
    print('--------------------------------------------')
    print('---- Dataset : {}'.format(dataset))
    print('---- Model : {}'.format(arg_model))
    print('---- Ns = {}'.format(sampling))
    print('-------------------------------------------- \n')
    print(performances)
    
    best_worst = pd.DataFrame({ 'Best L0' : [Distance['L0'].min()], 'Worst L0' : [Distance['L0'].max()]})
    best_worst.rename(index={0:'{}_{}'.format(dataset, arg_model)})
    best_worst.to_csv(os.path.join(path_result, 'Best_Worst.csv'), index=False)
    
    print('\n Best and Worst L0 : \n')
    print(best_worst)
    
    if batch_size == 1:
        T = float(Time.mean().values)
        print('\n Average time to generate an adversarial example : {:.3f} seconds \n'.format(T))
        save_time = pd.DataFrame({ 'Avg Time' : [T]})
        save_time.to_csv(os.path.join(path_result, 'Avg_Time.csv'), index=False)
        
        MP = Model_Prop.mean(axis=0).values
        print('\n Average forward and backward numbers - avg F : {:.3f} and avg B :  {:.3f}'.format(MP[0], MP[1]))
