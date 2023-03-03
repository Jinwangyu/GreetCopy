# _*_coding:utf-8 _*_
# @Time: 2023/3/2 21:16
# @Author: Jinwangyu
# @File: data_loader
import os
import dgl
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import remove_self_loops


def load_data(dataset_name):
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '.', 'data', dataset_name)
    dataset = Planetoid(path, dataset_name)
    print(dataset)
    data = dataset[0]
    print(data)

    edges = remove_self_loops(data.edge_index)[0]

    features = data.x
    [nnodes, nfeats] = features.shape





if __name__ == '__main__':
    load_data('cora')
