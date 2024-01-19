
import json
import copy
import numpy as np
import collections
import random
import h5py

from .liver import Dataset as BaseDataset

class Hdf5Reader:
    def __init__(self, path):
        try:
            self.file = h5py.File(path, "r")
        except Exception:
            print('{} not found!'.format(path))
            self.file = None

    def __getitem__(self, key):
        data = {'id': key}
        if self.file is None:
            return data
        group = self.file[key]
        for k in group:
            data[k] = group[k]
        return data

class FileManager:
    def __init__(self, files):
        self.files = {}
        for k, v in files.items():
            self.files[k] = Hdf5Reader(v["path"])

    def __getitem__(self, key):
        p = key.find('/')
        if key[:p] in self.files:
            ret = self.files[key[:p]][key[p+1:]]
            ret['id'] = key.replace('/', '-')
            return ret
        elif '/' in self.files:
            ret = self.files['/'][key]
            ret['id'] = key.replace('/', '-')
            return ret
        else:
            raise KeyError('{} not found'.format(key))


class Dataset(BaseDataset):
    def __init__(self, split_path, paired=False, task=None, batch_size=None):
        with open(split_path, 'r') as f:
            config = json.load(f)
        self.files = FileManager(config['files'])
        self.subset = {}

        for k, v in config['subsets'].items():
            self.subset[k] = {}
            for entry in v:

                self.subset[k][entry] = self.files[entry]

        self.paired = paired

        def convert_int(key):
            try:
                return int(key)
            except ValueError as e:
                return key
        #schemes[1] = "train": 1.0 ,schemes[2] = "val": 1.0
        self.schemes = dict([(convert_int(k), v)
                             for k, v in config['schemes'].items()])

        #subset
        #key:train,val ;  value:train_data,val_data
        for k, v in self.subset.items():
            print('Number of data in {} is {}'.format(k, len(v)))

        self.task = task
        if self.task is None:
            self.task = config.get("task", "registration")
        if not isinstance(self.task, list):
            self.task = [self.task]

        self.image_size = config.get("image_size", [160, 160, 160])
        self.segmentation_class_value = config.get(
            'segmentation_class_value', None)

        if 'atlas' in config:
            if isinstance(config['atlas'],list):
                self.atlas = []
                for sub_atlas in config['atlas']:
                    self.atlas.append(self.files[sub_atlas])
            else:
                self.atlas = self.files[config['atlas']]
        else:
            self.atlas = None

        self.batch_size = batch_size

    def center_crop(self, volume):
        slices = [slice((os - ts) // 2, (os - ts) // 2 + ts) if ts < os else slice(None, None)
                  for ts, os in zip(self.image_size, volume.shape)]
        volume = volume[slices]

        ret = np.zeros(self.image_size, dtype=volume.dtype)
        slices = [slice((ts - os) // 2, (ts - os) // 2 + os) if ts > os else slice(None, None)
                  for ts, os in zip(self.image_size, volume.shape)]
        ret[slices] = volume

        return ret

    @staticmethod
    def generate_atlas(atlas, sets, loop=False ,add_train_data=False):
        sets = copy.copy(sets)
        #np.random.seed(1234)
        while True:
            if isinstance(atlas,list):
                if add_train_data:
                    atlas_add=atlas[:]
                    id = random.randint(0,len(atlas)-1)
                    atlas_lft = atlas_add.pop(id)
                    sets.extend(atlas_add)
            if loop:
                np.random.shuffle(sets)
            for d in sets:
                if add_train_data:
                    print('atlas_lft:',len(atlas_lft))
                    yield d,atlas_lft
                else:
                    print('atlas:',len(atlas))
                    yield d,atlas
            if not loop:
                break

    def generator(self, subset, batch_size=None, loop=False, pair_train=False, adj=False, if_seg=False ,add_train_data=False):
        if batch_size is None:
            batch_size = self.batch_size
        valid_mask = np.ones([2], dtype=np.bool)#task2
        scheme = self.schemes[subset]
        if 'registration' in self.task:
            if self.atlas is not None and pair_train == False:
                generators, fractions = zip(*[(self.generate_atlas(self.atlas, list(
                    self.subset[k].values()), loop ,add_train_data), fraction) for k, fraction in scheme.items()])#fraction:1.0
            else:
                generators, fractions = zip(
                    *[(self.generate_pairs(list(self.subset[k].values()), loop, adj), fraction) for k, fraction in scheme.items()])

            while True:
                imgs = [batch_size] + self.image_size + [1]
                ret = dict()
                ret['voxelT1'] = np.zeros(imgs, dtype=np.float32)
                ret['atlasT1'] = np.zeros(imgs, dtype=np.float32)

                ret['seg1'] = np.zeros(imgs, dtype=np.float32)
                ret['seg2'] = np.zeros(imgs, dtype=np.float32)
                ret['point1'] = np.ones(
                    (batch_size, np.sum(valid_mask), 3), dtype=np.float32) * (-1)
                ret['point2'] = np.ones(
                    (batch_size, np.sum(valid_mask), 3), dtype=np.float32) * (-1)

                ret['id1'] = np.empty((batch_size), dtype='<U40')
                ret['id2'] = np.empty((batch_size), dtype='<U40')

                ret['pseudo_label'] = np.zeros(imgs, dtype=np.float32)

                i = 0
                flag = True
                cc = collections.Counter(np.random.choice(range(len(fractions)), size=[
                                         batch_size, ], replace=True, p=fractions))
        
                nums = [cc[i] for i in range(len(fractions))]
                for gen, num in zip(generators, nums):
                    assert not self.paired or num % 2 == 0
                    for t in range(num):
                        try:
                            while True:
                                d1, D2 = next(gen)
                                break
                        except StopIteration:
                            flag = False
                            break
                        if isinstance(D2,list):
                            Ret = []
                            for d2 in D2:
                                if 'segmentation' in d1:
                                    ret['seg1'][i, ..., 0] = d1['segmentation']
                                if 'segmentation' in d2:
                                    ret['seg2'][i, ..., 0] = d2['segmentation']
                                if 'point' in d1:
                                    ret['point1'][i] = d1['point'][...][valid_mask]
                                if 'point' in d2:
                                    ret['point2'][i] = d2['point'][...][valid_mask]

                                ret['voxelT1'][i, ..., 0], ret['atlasT1'][i, ...,0] = d1['volumeT1'], d2['volumeT1']
                                ret['id1'][i] = d1['id']
                                ret['id2'][i] = d2['id']
                                if 'pseudo_label' in d1:
                                    ret['pseudo_label'][i, ..., 0] = d1['pseudo_label']
                                Ret.append(ret)
                                
                                ret = copy.deepcopy(ret)
                            if loop or if_seg:
                                Ret = Ret[random.randint(0,len(Ret)-1)]
                        
                        else:
                            # print('多图谱转换为单图谱')
                            d2 = D2
                            if 'segmentation' in d1:
                                ret['seg1'][i, ..., 0] = d1['segmentation']
                            if 'segmentation' in d2:
                                ret['seg2'][i, ..., 0] = d2['segmentation']
                            if 'point' in d1:
                                ret['point1'][i] = d1['point'][...][valid_mask]
                            if 'point' in d2:
                                ret['point2'][i] = d2['point'][...][valid_mask]

                            ret['voxelT1'][i, ..., 0], ret['atlasT1'][i, ...,0] = d1['volumeT1'], d2['volumeT1']
                            ret['id1'][i] = d1['id']
                            ret['id2'][i] = d2['id']
                            if 'pseudo_label' in d1:
                                ret['pseudo_label'][i, ..., 0] = d1['pseudo_label']
                            Ret = ret
                        i += 1

                if flag:
                    assert i == batch_size
                    # print('RET lens:',len(Ret))
                    yield Ret
                else:
                    # print('RET lens:',len(Ret))
                    yield Ret
                    break
