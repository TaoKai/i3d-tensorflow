import os, sys
import numpy as np
import cv2

class MyData(object):
    def __init__(self, path, batch_size=16):
        self.batch_size = batch_size
        self.train_list, self.test_list = self.get_data_list(path)
        self.tr_len = self.train_list.shape[0]
        self.te_len = self.test_list.shape[0]
        self.tr_ind = 0
        self.te_ind = 0
    
    def get_test_num(self):
        return self.te_len//self.batch_size+1

    def next_test(self):
        if self.te_ind+self.batch_size>=self.te_len:
            data_batch = self.get_data_batch(self.test_list[self.te_ind:])
            self.te_ind = 0
            return data_batch
        else:
            data_batch = self.get_data_batch(self.test_list[self.te_ind:self.te_ind+self.batch_size])
            self.te_ind += self.batch_size
            return data_batch

    def get_train_num(self):
        return self.tr_len//self.batch_size+1

    def next_train(self):
        if self.tr_ind+self.batch_size>=self.tr_len:
            data_batch = self.get_data_batch(self.train_list[self.tr_ind:])
            self.tr_ind = 0
            return data_batch
        else:
            data_batch = self.get_data_batch(self.train_list[self.tr_ind:self.tr_ind+self.batch_size])
            self.tr_ind += self.batch_size
            return data_batch

    def get_data_batch(self, records):
        recs = list(records[:, 0])
        res = map(self.get_data_mat, recs)
        res = list(res)
        imgs = np.stack(res, axis = 0)
        labels = records[:, 1].astype(np.int32)
        names = records[:, 2]
        return imgs, labels, names

    def get_data_mat(self, rec):
        imgs = np.load(rec).astype(np.float32)
        imgs = (imgs-127.5)/128
        #print(imgs.shape)
        return imgs

    def get_data_list(self, path):
        test_path = os.path.join(path, 'test')
        train_path = os.path.join(path, 'train')
        test_mat = np.load(os.path.join(path, 'test.npy'))
        train_mat = np.load(os.path.join(path, 'train.npy'))
        test_mat = list(test_mat)
        train_mat = list(train_mat)
        test_list = []
        train_list = []
        for t in test_mat:
            p = os.path.join(test_path, t[0])
            l = int(t[1])
            ln = str(t[2])
            test_list.append((p, l, ln))
        for t in train_mat:
            p = os.path.join(train_path, t[0])
            l = int(t[1])
            ln = str(t[2])
            train_list.append((p, l, ln))
        return np.array(train_list), np.array(test_list)


if __name__ == '__main__':
    data = MyData('./data')
    for i in range(data.get_test_num()):
        bt = data.next_test()
        print(i, bt[0].shape, bt[1], bt[2])
    for i in range(data.get_train_num()):
        bt = data.next_train()
        print(i, bt[0].shape, bt[1], bt[2])
