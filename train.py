from tqdm import tqdm
import numpy as np
import argparse
import os
from pathlib import Path
import tensorflow as tf
from networks import Cov1DModel, single_loss, Transformer, AttentionCov1DModel
from datasets import CWRUDataset, CWRUDataloader
from utils import Params, cal_acc, generate_classifier, increment_dir, cal_classifier_acc

def get_args():
    parser = argparse.ArgumentParser(description='CWRU data training.')
    parser.add_argument('--cfg', type=str, default='./cfg/cwru.yml', help='The path to the configuration file.')
    parser.add_argument('--log-dir', type=str, default='./log', help='The directory of log files.')
    parser.add_argument('--name', default='', help='renames results.txt to results_name.txt if supplied')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--k', type=int, default=10, help='K-fold cross validation.')
    parser.add_argument('--optim', type=str, default='adam', help='The optimizer to use (Adam or SGD).')
    parser.add_argument('--model', type=str, default='cnn', help='1DCNN/Transformer.')

    return parser.parse_args()

def train(abnormal_flag, faults_classifiers, epochs, m1, m2, batch_size, time_steps, channels, optim_type, threshold, results_path):

    train_loader = faults_classifiers[abnormal_flag]['train_loader']
    test_dataloader = faults_classifiers[abnormal_flag]['test_loader']

    model = Cov1DModel()
    model.build(input_shape=(batch_size*2, time_steps, channels))

    # モデルの内容を出力
    model.summary()

    if optim_type == 'adam':
        optimizer = tf.keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    elif optim_type == 'nadam':
        optimizer = tf.keras.optimizers.Nadam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
    elif optim_type == 'sgd':
        optimizer = tf.keras.optimizers.SGD(lr=0.0001, decay=1e-4, momentum=0.8, nesterov=True)

    best_acc = 0
    best_diff = 0
    best_model = None
    for epoch in range(epochs):
        process = tqdm(enumerate(train_loader), total=train_loader.gen_len())
        train_losses = []
        train_accs = []
        for step, data in process:
            pos_data, neg_data = data['pos_data'], data['neg_data']
            batch = tf.concat([pos_data, neg_data], 0)

            loss = 0.0 #stepごとにlossを初期化
            with tf.GradientTape() as t:
                pred = tf.zeros([batch_size*2, 0, 1])
                pred = model(batch)
                loss = single_loss(pred, m1, m2)
                train_losses.append(loss)
                acc, pos_sum, neg_sum = cal_acc(mode='train', threshold=threshold, pred=pred)
                train_accs.append(acc)
            
            d = t.gradient(loss, model.trainable_weights)
            optimizer.apply_gradients(zip(d, model.trainable_weights))

            postfix = 'Step: {0:4d}, Train loss: {1:.4f}, Train acc: {2:.4f}, Positive score: {3:.4f}, Negative score: {4:.4f}'.format(step+1, loss, acc, pos_sum, neg_sum)
            process.set_postfix_str(postfix)

        # Eval
        #val_acc = eval(model, abnormal_flag, faults_classifiers, m1, m2, threshold)
        # Test
        val_losses = []
        val_accs = []
        pos_scores = []
        neg_scores = []
        process = tqdm(enumerate(test_dataloader), total=test_dataloader.gen_len())
        for step, data in process:
            pos_data, neg_data = data['pos_data'], data['neg_data']
            batch = tf.concat([pos_data, neg_data], 0)
            pred = model(batch)
            loss = single_loss(pred, m1, m2)
            val_losses.append(loss)
            acc, pos_sum, neg_sum = cal_acc(mode='train', threshold=threshold, pred=pred)
            val_accs.append(acc)
            pos_scores.append(pos_sum)
            neg_scores.append(neg_sum)

            postfix = 'Step: {0:4d}, Val loss: {1:.4f}, Val acc: {2:.4f}, Positive score: {3:.4f}, Negative score: {4:.4f}'.format(step+1, loss, acc, sum(pos_scores)/len(pos_scores), sum(neg_scores)/len(neg_scores))
            process.set_postfix_str(postfix)

        if best_acc < sum(val_accs)/len(val_accs):
            best_acc = sum(val_accs)/len(val_accs)
            best_model = model

        results_str = 'Epoch: {0:4d}, Train loss: {1:.4f}, Train acc: {2:.4f}, Val loss: {3:.4f}, Val acc: {4:.4f}\n'.format(epoch+1, sum(train_losses)/len(train_losses), sum(train_accs)/len(train_accs), sum(val_losses)/len(val_losses), sum(val_accs)/len(val_accs))
        print(results_str)
        with open(results_path, 'a') as f:
            f.write(results_str)

    return best_model

def batch_train(**kwargs):
    m = kwargs.get('m', 'cnn')
    faults_classifiers = kwargs.get('faults_classifiers', None)
    epochs = kwargs.get('epochs', 1)
    m1 = kwargs.get('m1', 0.1)
    m2 = kwargs.get('m2', 0.1)
    batch_size = kwargs.get('batch_size', 8)
    time_steps = kwargs.get('time_steps', 200)
    channels = kwargs.get('channels', 2)
    optim_type = kwargs.get('optim_type', 'adam')
    threshold = kwargs.get('threshold', None)
    acc_path = kwargs.get('acc_path', None)

    accs = {}
    for k in faults_classifiers.keys():
        if m == 'cnn':
            model = Cov1DModel()
            model.build(input_shape=(batch_size*2, time_steps, channels))
        elif m == 'att_cnn':
            model = AttentionCov1DModel()
            model.build(input_shape=(batch_size*2, time_steps, channels))
        elif m == 'transformer':
            num_layers = kwargs.get('num_layers', 4)
            num_heads = kwargs.get('num_heads', 2)
            d_model = kwargs.get('d_model', 2)
            hidden_layer_shape = kwargs.get('hidden_layer_shape', 2048)
            max_pos_encoding = kwargs.get('max_pos_encoding', 100000)
            rate = kwargs.get('rate', 0.1)
            model = Transformer(num_layers=num_layers, num_heads=num_heads, d_model=d_model, 
                                hidden_layer_shape=hidden_layer_shape, max_pos_encoding=max_pos_encoding, rate=rate)
        
        faults_classifiers[k]['model'] = model

        if optim_type == 'adam':
            optimizer = tf.keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        elif optim_type == 'nadam':
            optimizer = tf.keras.optimizers.Nadam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
        elif optim_type == 'sgd':
            optimizer = tf.keras.optimizers.SGD(lr=0.0001, decay=1e-4, momentum=0.8, nesterov=True)

        faults_classifiers[k]['optimizer'] = optimizer

    best_avg_acc = 0
    best_accs = None
    for epoch in range(epochs):
        for k in faults_classifiers.keys():
            train_loader = faults_classifiers[k]['train_loader']
            process = tqdm(enumerate(train_loader), total=train_loader.gen_len())
            train_losses = []
            train_accs = []
            pos_scores = []
            neg_scores = []
            for step, data in process:
                pos_data, neg_data = data['pos_data'], data['neg_data']
                batch = tf.concat([pos_data, neg_data], 0)

                loss = 0.0 #stepごとにlossを初期化
                with tf.GradientTape() as t:
                    pred = tf.zeros([batch_size*2, 0, 1])
                    pred = faults_classifiers[k]['model'](batch)
                    loss = single_loss(pred, m1, m2)
                    train_losses.append(loss)
                    acc, pos_sum, neg_sum = cal_acc(mode='train', threshold=threshold, pred=pred)
                    train_accs.append(acc)
                    pos_scores.append(pos_sum)
                    neg_scores.append(neg_sum)
                
                d = t.gradient(loss, faults_classifiers[k]['model'].trainable_weights)
                faults_classifiers[k]['optimizer'].apply_gradients(zip(d, faults_classifiers[k]['model'].trainable_weights))

                postfix = 'Fault: {0}, Step: {1:4d}, Train loss: {2:.4f}, Train acc: {3:.4f}, Positive score: {4:.4f}, Negative score: {5:.4f}'.format(k, step+1, sum(train_losses)/len(train_losses), sum(train_accs)/len(train_accs), sum(pos_scores)/len(pos_scores), sum(neg_scores)/len(neg_scores))
                process.set_postfix_str(postfix)

            # Eval
            val_losses = []
            val_accs = []
            pos_scores = []
            neg_scores = []
            test_dataloader = faults_classifiers[k]['test_loader']
            process = tqdm(enumerate(test_dataloader), total=test_dataloader.gen_len())
            for step, data in process:
                pos_data, neg_data = data['pos_data'], data['neg_data']
                batch = tf.concat([pos_data, neg_data], 0)
                pred = faults_classifiers[k]['model'](batch)
                loss = single_loss(pred, m1, m2)
                val_losses.append(loss)
                acc, pos_sum, neg_sum = cal_acc(mode='train', threshold=threshold, pred=pred)
                val_accs.append(acc)
                pos_scores.append(pos_sum)
                neg_scores.append(neg_sum)

                postfix = 'Fault: {0}, Step: {1:4d}, Val loss: {2:.4f}, Val acc: {3:.4f}, Positive score: {4:.4f}, Negative score: {5:.4f}'.format(k, step+1, sum(val_losses)/len(val_losses), sum(val_accs)/len(val_accs), sum(pos_scores)/len(pos_scores), sum(neg_scores)/len(neg_scores))
                process.set_postfix_str(postfix)

        cur_accs = cal_classifier_acc(fault_flags, faults_classifiers, threshold)
        accs = []
        for fault_flag, acc in cur_accs.items():
            results_str = 'Epoch: {0:4d}, fault: {1}: {2:.4f}\n'.format(epoch+1, fault_flag, acc)
            print(results_str)
            with open(acc_path, 'a') as f:
                f.write(results_str)
            accs.append(acc)
        avg_acc = sum(accs) / len(accs)
        if avg_acc > best_avg_acc:
            best_avg_acc = avg_acc
            best_accs = cur_accs

    return best_accs


if __name__ == '__main__':
    args = get_args()
    cfg, log_dir, name, epochs, batch_size, k, optim_type, m = \
        args.cfg, args.log_dir, args.name, args.epochs, args.batch_size, args.k, args.optim, args.model
    
    params = Params(cfg)
    data_dir, fault_flags, time_steps, channels, threshold, m1, m2 = \
        params.data_dir, params.fault_flags, params.time_steps, params.channels, params.threshold, params.m1, params.m2 
    if m == 'transformer':
        num_layers, num_heads, d_model, hidden_layer_shape, max_pos_encoding, rate = \
            params.num_layers, params.num_heads, params.d_model, params.hidden_layer_shape, params.max_pos_encoding, params.rate 

    log_dir = increment_dir(Path(log_dir) / 'exp', name)
    print('Log directory: {}'.format(log_dir))
    weight_dir = os.path.join(log_dir, 'weights')
    if not os.path.exists(weight_dir):
        os.makedirs(weight_dir)
    results_dir = os.path.join(log_dir, 'results')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    dataset = CWRUDataset(data_dir=data_dir, 
                           fault_flags=fault_flags, 
                           time_steps=time_steps, 
                           channels=channels,
                           k=k)

    batch_eval = True
    if batch_eval:
        final_accs = {}
        for val_idx in range(k):
            print('Start {0}-Fold Cross-Validation: {1}'.format(k, val_idx+1))
            faults_classifiers = generate_classifier(fault_flags, dataset, val_idx, batch_size)
            acc_path = os.path.join(results_dir, 'avg_acc.txt')
            if m == 'cnn' or m == 'att_cnn':
                accs = batch_train(m=m, 
                                   faults_classifiers=faults_classifiers, 
                                   epochs=epochs, 
                                   m1=m1, 
                                   m2=m2, 
                                   batch_size=batch_size, 
                                   time_steps=time_steps, 
                                   channels=channels, 
                                   optim_type=optim_type, 
                                   threshold=threshold, 
                                   acc_path=acc_path)
            elif m == 'transformer':
                accs = batch_train(m=m, 
                                   faults_classifiers=faults_classifiers, 
                                   epochs=epochs, 
                                   m1=m1, 
                                   m2=m2, 
                                   batch_size=batch_size, 
                                   time_steps=time_steps, 
                                   channels=channels, 
                                   optim_type=optim_type, 
                                   threshold=threshold, 
                                   acc_path=acc_path, 
                                   num_layers=num_layers, 
                                   num_heads=num_heads, 
                                   d_model=d_model, 
                                   hidden_layer_shape=hidden_layer_shape,
                                   max_pos_encoding=max_pos_encoding, 
                                   rate=rate)
            for fault_flag, acc in accs.items():
                if fault_flag in final_accs.keys():
                    final_accs[fault_flag].append(acc)
                else:
                    final_accs[fault_flag] = [acc]
            break
        print('\nFinal evaluating...\n')
        final_acc_path = os.path.join(results_dir, 'final_avg_acc.txt')
        for fault_flag, v in final_accs.items():
            avg_acc = sum(v) / len(v)
            results_str = '{0}: {1:.4f}\n'.format(fault_flag, avg_acc)
            print(results_str)
            with open(final_acc_path, 'a') as f:
                f.write(results_str)
    else:
        accs = {}
        for val_idx in range(k):
            print('Start {0}-Fold Cross-Validation: {1}'.format(k, val_idx+1))
            faults_classifiers = generate_classifier(fault_flags, dataset, val_idx, batch_size)
            for fault_flag in fault_flags:
                if fault_flag == 'Normal':
                    continue
                results_path = os.path.join(results_dir, '{}_{}_results.txt'.format(fault_flag, val_idx))
                model = train(fault_flag, faults_classifiers, epochs, m1, m2, batch_size, time_steps, channels, optim_type, threshold, results_path)
                save_path = os.path.join(weight_dir, '{}_{}.h5'.format(fault_flag, val_idx))
                model.save_weights(save_path)
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
