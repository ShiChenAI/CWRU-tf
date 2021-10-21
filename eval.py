from tqdm import tqdm
import numpy as np
import argparse
import os
from pathlib import Path
import tensorflow as tf
from networks import Cov1DModel, single_loss
from datasets import CWRUDataset, CWRUDataloader
from utils import Params, cal_acc, generate_classifier, increment_dir, cal_classifier_acc

def get_args():
    parser = argparse.ArgumentParser(description='CWRU data training.')
    parser.add_argument('--cfg', type=str, default='./cfg/cwru.yml', help='The path to the configuration file.')
    parser.add_argument('--log-dir', type=str, default='./log/exp11', help='The directory of log files.')
    parser.add_argument('--weight-dir', type=str, default='./log/exp11/weights', help='The directory of log files.')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--k', type=int, default=10, help='K-fold cross validation.')

    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    cfg, log_dir, weight_dir, batch_size, k = \
        args.cfg, args.log_dir, args.weight_dir, args.batch_size, args.k
    
    params = Params(cfg)
    data_dir, fault_flags, time_steps, channels, threshold, m1, m2 = \
        params.data_dir, params.fault_flags, params.time_steps, params.channels, params.threshold, params.m1, params.m2 

    print('Log directory: {}'.format(log_dir))

    results_dir = os.path.join(log_dir, 'results')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    dataset = CWRUDataset(data_dir=data_dir, 
                          fault_flags=fault_flags, 
                          time_steps=time_steps, 
                          channels=channels,
                          k=k)

    accs = {}
    for val_idx in range(k):
        print('Start {0}-Fold Cross-Validation: {1}'.format(k, val_idx+1))
        faults_classifiers = generate_classifier(fault_flags, dataset, val_idx, batch_size)
        for fault_flag in fault_flags:
            if fault_flag == 'Normal':
                continue
            results_path = os.path.join(results_dir, '{}_{}_results.txt'.format(fault_flag, val_idx))
            weight_path = os.path.join(weight_dir, '{}_{}.h5'.format(fault_flag, val_idx))
            model = Cov1DModel()
            model.build(input_shape=(batch_size*2, time_steps, channels))
            model.load_weights(weight_path)
            faults_classifiers[fault_flag]['model'] = model

        print('\nEvaluating fold {0}...\n'.format(val_idx+1))
        cur_accs = cal_classifier_acc(fault_flags, faults_classifiers, threshold)
        print('\nEvaluating results of fold {0}:\n'.format(val_idx+1))
        for fault_flag, acc in cur_accs.items():
            print('{0}: {1:.4f}\n'.format(fault_flag, acc))
            if fault_flag in accs.keys():
                accs[fault_flag].append(acc)
            else:
                accs[fault_flag] = [acc]

    print('\nFinal evaluating...\n')
    acc_path = os.path.join(results_dir, 'avg_acc.txt')
    for fault_flag, v in accs.items():
        avg_acc = sum(v) / len(v)
        results_str = '{0}: {1:.4f}\n'.format(fault_flag, avg_acc)
        print(results_str)
        with open(acc_path, 'a') as f:
            f.write(results_str)