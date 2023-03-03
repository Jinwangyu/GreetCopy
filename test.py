# _*_coding:utf-8 _*_
# @Time: 2023/3/2 20:58
# @Author: Jinwangyu
# @File: test
import numpy as np
import os


np.random.seed(0)

L1 = np.random.randn(3, 3)
np.random.seed(0)

L2 = np.random.randn(3, 3)
print(L1)
print(L2)

dataset_name = 'cora'
path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '.', 'data', dataset_name)
print(path)


if __name__ == '__main__':
    print('')
