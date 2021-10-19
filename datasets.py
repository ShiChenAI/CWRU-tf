import os
import glob
import random
import math
import numpy as np
from tqdm import tqdm
from scipy.io import loadmat

class CWRUDataset(object):
    def __init__(self, data_dir, fault_flags, time_steps, channels, k):
        self.data_dir = data_dir # CWRUデータフォルダ
        self.fault_flags = fault_flags # 故障種別データフラッグ
        #self.abnormal_flags = abnormal_flags # 異常データフラッグ
        self.time_steps = time_steps # データ長
        self.channels = channels
        self.k = k # k-分割交差検証のサブセット数
        self.val_idx = 0 # 検証サブセット

        # 正常/異常データの読み取り
        #self.normal_samples, self.abnormal_samples = self._get_samples()
        self.samples = self._get_samples()

    def _get_normal_flags(self, abnormal_flags):
        return [x for x in self.fault_flags if x not in abnormal_flags]
  
    def _mat2arr(self, mat_path):
        mat = loadmat(mat_path)

        if len(mat.keys()) > 4:       
            return np.hstack([mat[key] for key in list(mat.keys())[3:-1]])
        else:
            return mat[list(mat.keys())[3]]

    def _split_files(self, flags):
        kf_samples = {}
        for flag in flags:
            files_dir = os.path.join(self.data_dir, flag)
            samples = []
            for f in tqdm(glob.glob('{}/*'.format(files_dir)), desc='Splitting CWRU files ({})'.format(flag)):
                arr = self._mat2arr(f)
                if arr.shape[1] == 1:
                    continue
                else:
                    arr = arr[:, :self.channels]
                sample_num = len(arr) // self.time_steps
                for idx in range(sample_num):
                    sample = arr[idx*self.time_steps: (idx+1)*self.time_steps]
                    samples.append(sample)

            random.shuffle(samples)
            k_sample_num = int(math.ceil(len(samples) / float(self.k)))
            kf_samples[flag] = [samples[i:i+k_sample_num] for i in range(0, len(samples), k_sample_num)]

        return kf_samples

    def _get_samples(self):
        return self._split_files(self.fault_flags)

    def __norm(self, data):
        """正規化
        """

        return (data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0))

    def generate_datasets(self, abnormal_flags, val_idx):
        self.val_idx = val_idx
        
        # 正常データフラッグ(正常なデータと異常データフラッグを除くすべての故障種別データフラッグ)
        normal_flags = self._get_normal_flags(abnormal_flags)

        self.normal_samples_train = None
        self.normal_samples_test = None
        for normal_flag in normal_flags:
            print('Normal flag: {}'.format(normal_flag))
            for idx, s in enumerate(self.samples[normal_flag]):
                if idx != self.val_idx:
                    for sample in tqdm(s, desc='Generating normal training data ({})'.format(idx+1)):
                        if self.normal_samples_train is None:
                            self.normal_samples_train = sample[np.newaxis, :, :]
                        else:
                            self.normal_samples_train = np.vstack((self.normal_samples_train, sample[np.newaxis, :, :]))
                else:
                    for sample in tqdm(s, desc='Generating normal testing data ({})'.format(idx+1)):
                        if self.normal_samples_test is None:
                            self.normal_samples_test = sample[np.newaxis, :, :]
                        else:
                            self.normal_samples_test = np.vstack((self.normal_samples_test, sample[np.newaxis, :, :]))

        self.normal_samples_train = self.__norm(self.normal_samples_train)
        self.normal_samples_test = self.__norm(self.normal_samples_test)

        self.abnormal_samples_train = None
        self.abnormal_samples_test = None
        for abnormal_flag in abnormal_flags:
            print('Abnormal flag: {}'.format(abnormal_flag))
            for idx, s in enumerate(self.samples[abnormal_flag]):
                if idx != self.val_idx:
                    for sample in tqdm(s, desc='Generating abnormal training data ({})'.format(idx+1)):
                        if self.abnormal_samples_train is None:
                            self.abnormal_samples_train = sample[np.newaxis, :, :]
                        else:
                            self.abnormal_samples_train = np.vstack((self.abnormal_samples_train, sample[np.newaxis, :, :]))
                else:
                    for sample in tqdm(s, desc='Generating abnormal testing data ({})'.format(idx+1)):
                        if self.abnormal_samples_test is None:
                            self.abnormal_samples_test = sample[np.newaxis, :, :]
                        else:
                            self.abnormal_samples_test = np.vstack((self.abnormal_samples_test, sample[np.newaxis, :, :]))
        
        self.abnormal_samples_train = self.__norm(self.abnormal_samples_train)
        self.abnormal_samples_test = self.__norm(self.abnormal_samples_test)

        return {'train': [self.normal_samples_train, self.abnormal_samples_train],
                'test': [self.normal_samples_test, self.abnormal_samples_test]}

class CWRUDataloader(object):
    def __init__(self, dataset, batch_size):
        self.normal_samples = dataset[0]
        self.abnormal_samples = dataset[1]
        self.batch_size = batch_size
        self.batch_ids = [batch_size, batch_size]

    def __iter__(self): 
        return self
 
    def __next__(self):  
        if self.batch_ids[0] <= len(self.normal_samples):
            if self.batch_ids[1] > len(self.abnormal_samples):
                self.batch_ids[1] = self.batch_size

            neg_samples = self.normal_samples[self.batch_ids[0]-self.batch_size:self.batch_ids[0], :, :]
            pos_samples = self.abnormal_samples[self.batch_ids[1]-self.batch_size:self.batch_ids[1], :, :]

            self.batch_ids[0] += self.batch_size
            self.batch_ids[1] += self.batch_size

            return {'pos_data': pos_samples, 'neg_data': neg_samples}
        else:
            self.batch_ids = [self.batch_size, self.batch_size]
            np.random.shuffle(self.normal_samples)
            np.random.shuffle(self.abnormal_samples)
            raise StopIteration

    def gen_len(self):
        return len(self.normal_samples) // self.batch_size 