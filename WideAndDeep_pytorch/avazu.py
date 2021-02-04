import shutil
import struct
from collections import defaultdict
from pathlib import Path

import lmdb
import numpy as np
import torch.utils.data
from tqdm import tqdm


class AvazuDataset(torch.utils.data.Dataset):
    """
    Avazu Click-Through Rate Prediction Dataset

    Dataset preparation
        Remove the infrequent features (appearing in less than threshold instances) and treat them as a single feature

    :param dataset_path: avazu train path
    :param cache_path: lmdb cache path
    :param rebuild_cache: If True, lmdb cache is refreshed
    :param min_threshold: infrequent feature threshold，如果某一特征值出现的次数少于min_threshold的特征，则去掉，统一当成一个特征其它

    Reference
        https://www.kaggle.com/c/avazu-ctr-prediction
    """

    def __init__(self, dataset_path=None, cache_path='.avazu', rebuild_cache=False, min_threshold=4):
        self.NUM_FEATS = 22
        self.min_threshold = min_threshold
        
        #如果cache文件不存在，重新构建
        if rebuild_cache or not Path(cache_path).exists():
            shutil.rmtree(cache_path, ignore_errors=True)
            if dataset_path is None:
                raise ValueError('create cache: failed: dataset_path is None')
            self.__build_cache(dataset_path, cache_path)
            
        #如果已经存在或者构建完毕，则读取
        self.env = lmdb.open(cache_path, create=False, lock=False, readonly=True)
        with self.env.begin(write=False) as txn:
            self.length = txn.stat()['entries'] - 1
            self.field_dims = np.frombuffer(txn.get(b'field_dims'), dtype=np.uint32)

    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:
            np_array = np.frombuffer(
                txn.get(struct.pack('>I', index)), dtype=np.uint32).astype(dtype=np.long)
                #返回x,y
        return np_array[1:], np_array[0]

    def __len__(self):
        return self.length

    def __build_cache(self, path, cache_path):
        feat_mapper, defaults = self.__get_feat_mapper(path)
        with lmdb.open(cache_path, map_size=int(1e11)) as env:
        #[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],shape=(1,22)
            field_dims = np.zeros(self.NUM_FEATS, dtype=np.uint32)
            #填充field_dims，即每个特征的特征值种类数，多加1，应该是将出现频率少的归为其它类，即多加的一个编号
            for i, fm in feat_mapper.items():
                field_dims[i - 1] = len(fm) + 1
                
            #先将field_dims写入cache
            with env.begin(write=True) as txn:
                txn.put(b'field_dims', field_dims.tobytes())
                
            #利用yield机制批量写入cache，不至于占用内存太大
            for buffer in self.__yield_buffer(path, feat_mapper, defaults):
                with env.begin(write=True) as txn:
                    for key, value in buffer:
                        txn.put(key, value)

    def __get_feat_mapper(self, path):
        #嵌套的defaultdict字典
        feat_cnts = defaultdict(lambda: defaultdict(int))
        with open(path) as f:
            f.readline()
            pbar = tqdm(f, mininterval=1, smoothing=0.1)
            pbar.set_description('Create avazu dataset cache: counting features')
            for line in pbar:
                values = line.rstrip('\n').split(',')
                if len(values) != self.NUM_FEATS + 2:
                    continue
                for i in range(1, self.NUM_FEATS + 1):
                #将除去id和click的22个特征存入字典feat_cnts中，字典的第1个key为1，对应value=hour特征，即第三个特征，以此类推
                    feat_cnts[i][values[i + 1]] += 1
        feat_mapper = {i: {feat for feat, c in cnt.items() if c >= self.min_threshold} for i, cnt in feat_cnts.items()}
        # #eg：{1: {'14102100': 0}, 2: {'1007': 0, '1002': 1, '1010': 2, '1005': 3, '1001': 4}
        feat_mapper = {i: {feat: idx for idx, feat in enumerate(cnt)} for i, cnt in feat_mapper.items()}
        #表示22个特征中，去掉频率小的值后，每个特征的值的种类数
        #eg:{1: 1, 2: 5, 3: 2, 4: 137, 5: 122, 6: 11, 7: 86, 8: 20, 9: 10,
        defaults = {i: len(cnt) for i, cnt in feat_mapper.items()}
        return feat_mapper, defaults
    
    #生成的是一个生成器，每10万条记录返回一次
    def __yield_buffer(self, path, feat_mapper, defaults, buffer_size=int(1e5)):
        item_idx = 0
        buffer = list()
        with open(path) as f:
            f.readline()
            pbar = tqdm(f, mininterval=1, smoothing=0.1)
            pbar.set_description('Create avazu dataset cache: setup lmdb')
            for line in pbar:
                values = line.rstrip('\n').split(',')
                if len(values) != self.NUM_FEATS + 2:
                    continue
                    
                #多1位是要存放y值
                np_array = np.zeros(self.NUM_FEATS + 1, dtype=np.uint32)
                #0位置先存放的是click，即y值
                np_array[0] = int(values[1])
                #接下来存放剩下的22个x特征
                for i in range(1, self.NUM_FEATS + 1):
                    #出现频率少于阈值的默认归到一个其它类别里面，例如：一共有0，1，2三个编号，那么找不到的将是defaults[i]=3，因为一共是3个
                    np_array[i] = feat_mapper[i].get(values[i+1], defaults[i])
                 #(struct.pack('>I', item_idx)将数据转换位字节流
                buffer.append((struct.pack('>I', item_idx), np_array.tobytes()))
                item_idx += 1
                if item_idx % buffer_size == 0:
                    yield buffer
                    buffer.clear()
            yield buffer
