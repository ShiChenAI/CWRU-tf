import numpy as np
from tqdm import tqdm
import yaml
import glob
from pathlib import Path
from datasets import CWRUDataloader

class Params:
    def __init__(self, project_file):
        self.params = yaml.safe_load(open(project_file).read())

    def __getattr__(self, item):
        return self.params.get(item, None)

def cal_acc(**kwargs):
    mode = kwargs.get('mode', 'train')
    threshold = kwargs.get('threshold', [0.5, 0.1])
    pred = kwargs.get('pred', None)
    if mode == 'train':
        batch_size = pred.shape[0]//2 
        pred = np.sum(np.squeeze(pred, axis=2), axis=1)
        pos_sum = np.sum(pred[:batch_size])
        neg_sum = np.sum(pred[batch_size:])
        pos_mask = np.where(pred < threshold[0], 0 ,1)
        neg_mask = np.where(pred < threshold[1], 0 ,1)
        #pred = np.where(pred < threshold, 0 ,1)

        return (np.sum(pos_mask[:batch_size]==1) + np.sum(neg_mask[batch_size:]==0)) / len(pred), pos_sum, neg_sum
    elif mode == 'eval':
        abnormal_flag = kwargs.get('abnormal_flag', None)
        tp = 0
        total = 0
        for fault_flag, p in pred.items():
            p = np.sum(np.squeeze(p, axis=2), axis=1)
            p = np.where(p < threshold, 0 ,1)
            if fault_flag == abnormal_flag:
                # Positive
                tp += np.sum(p==1)
            else:
                # Negative
                tp += np.sum(p==0)
            total += len(p)

        return tp / total

def cal_classifier_acc(fault_flags, faults_classifiers, threshold=0.6):
    accs = {}
    for fault_flag in fault_flags:
        if fault_flag == 'Normal':
            continue

        if fault_flag == 'OR@12':
            ccc = 0
        data_loader = faults_classifiers[fault_flag]['test_loader']
        pos_model = faults_classifiers[fault_flag]['model']
        neg_models = []
        for k, v in faults_classifiers.items():
            if k == 'Normal' or k == fault_flag:
                continue

            neg_models.append(v['model'])

        tp = 0
        total = 0
        process = tqdm(enumerate(data_loader), total=data_loader.gen_len())
        for step, data in process:
            batch = data['pos_data']
            pred = pos_model(batch)
            pred = np.sum(np.squeeze(pred, axis=2), axis=1)
            for neg_model in neg_models:
                neg_pred = neg_model(batch)
                neg_pred = np.sum(np.squeeze(neg_pred, axis=2), axis=1)
                pred = np.vstack((pred, neg_pred))

            ab = np.where(pred>0, 1, 0)[0]
            max_idxs = np.argmax(pred, axis=0)
            tp += len(np.intersect1d(np.where(max_idxs==0)[0], np.where(ab==1)[0]))
            total += len(max_idxs)
            acc = len(np.intersect1d(np.where(max_idxs==0)[0], np.where(ab==1)[0])) / len(max_idxs)

            postfix = '[{0} samples evaluation] Step: {1:4d}, Val acc: {2:.4f}'.format(fault_flag, step+1, acc)
            process.set_postfix_str(postfix)

        fault_acc = tp / total
        accs[fault_flag] = fault_acc

    return accs

def generate_classifier(fault_flags, dataset, val_idx, batch_size):
    faults_classifiers = {}
    for fault_flag in fault_flags:
        if fault_flag == 'Normal':
            continue
        abnormal_flags = [fault_flag]
        print('[fold: {0}] Generating data...[fault_flag: {1}]'.format(val_idx+1, fault_flag))
        datasets = dataset.generate_datasets(abnormal_flags, val_idx)
        faults_classifiers[fault_flag] = {'model': None,
                                          'train_loader': CWRUDataloader(datasets['train'], batch_size), 
                                          'test_loader': CWRUDataloader(datasets['test'], batch_size)}

    return faults_classifiers

def increment_dir(dir, comment=''):
    # Increments a directory runs/exp1 --> runs/exp2_comment

    n = 0  # number
    dir = str(Path(dir))  # os-agnostic
    d = sorted(glob.glob(dir + '*'))  # directories
    if len(d):
        n = max([int(x[len(dir):x.find('_') if '_' in x else None]) for x in d]) + 1  # increment
    return dir + str(n) + ('_' + comment if comment else '')
