# _*_coding:utf-8 _*_
# @Time: 2023/3/2 19:37
# @Author: Jinwangyu
# @File: main.py
import argparse
import torch
import numpy as np
import random
import dgl

# 固定随机种子
def setup_seed(seed):
    # 分别为cpu和所有Gpu设置随机种子
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True   # 使用GPU时需要设置确保返回的卷积算法是确定的
    # 固定生成随机ndarray的种子
    np.random.seed(seed)
    # 固定生成随机数x的种子
    random.seed(seed)
    # 固定dgl中的随机种子
    dgl.seed(seed)
    dgl.random.seed(seed)


def main(args):
    setup_seed(0)

    print("")


if __name__ == '__main__':
    '''
        cora 
        -dataset cora -ntrials 10 -sparse 0 -epochs 400 -cl_batch_size 0 -nlayers_proj 1 -alpha 0.5 -k 20 -maskfeat_rate_1 0.8 
        -maskfeat_rate_2 0.1 -dropedge_rate_1 0.8 -dropedge_rate_2 0.8 -lr_disc 0.001 -margin_hom 0.5 -margin_het 0.5 
        -cl_rounds 2 -eval_freq 5

    '''
    parser = argparse.ArgumentParser()

    # ESSENTIAL
    parser.add_argument('-dataset', type=str, default='cora',
                        choices=['cora', 'citeseer', 'pubmed', 'chameleon', 'squirrel', 'actor', 'cornell',
                                 'texas', 'wisconsin', 'computers', 'photo', 'cs', 'physics', 'wikics'])
    parser.add_argument('-ntrials', type=int, default=10)
    parser.add_argument('-sparse', type=int, default=0)
    parser.add_argument('-eval_freq', type=int, default=5)
    parser.add_argument('-epochs', type=int, default=400)
    parser.add_argument('-lr_gcl', type=float, default=0.001)
    parser.add_argument('-lr_disc', type=float, default=0.001)
    parser.add_argument('-cl_rounds', type=int, default=2)
    parser.add_argument('-w_decay', type=float, default=0.0)
    parser.add_argument('-dropout', type=float, default=0.5)

    # DISC Module - Hyper-param
    '''
    -alpha 为高通滤波强度
    -margin_hom -margin_het  pivot-anchored ranking loss中两个视角下损失项的超参
    '''
    parser.add_argument('-alpha', type=float, default=0.5)
    parser.add_argument('-margin_hom', type=float, default=0.5)
    parser.add_argument('-margin_het', type=float, default=0.5)

    # GRL Module - Hyper-param
    '''
    -nlayers_enc -nlayers_proj 为编码器、投影网络层数
    -k 对比损失knn邻居节点数量
    -maskfeat和dropedge为双通道对比时图扰动率
    '''
    parser.add_argument('-nlayers_enc', type=int, default=2)
    parser.add_argument('-nlayers_proj', type=int, default=1, choices=[1, 2])
    parser.add_argument('-emb_dim', type=int, default=128)
    parser.add_argument('-proj_dim', type=int, default=128)
    parser.add_argument('-cl_batch_size', type=int, default=0)
    parser.add_argument('-k', type=int, default=20)
    parser.add_argument('-maskfeat_rate_1', type=float, default=0.1)
    parser.add_argument('-maskfeat_rate_2', type=float, default=0.5)
    parser.add_argument('-dropedge_rate_1', type=float, default=0.5)
    parser.add_argument('-dropedge_rate_2', type=float, default=0.1)

    args = parser.parse_args()

    print(args)




