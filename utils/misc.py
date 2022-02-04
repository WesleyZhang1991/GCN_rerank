from __future__ import division, print_function
import os
import glob
import numpy as np
# import mxnet as mx
# from mxnet.metric import EvalMetric


class DotDict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def clean_immediate_checkpoints(model_dir, prefix, final_epoch):
    ckpts = glob.glob(os.path.join(model_dir, "%s*.params" % prefix))

    for ckpt in ckpts:
        ckpt_name = os.path.basename(ckpt)
        epoch_idx = int(ckpt_name[:ckpt_name.rfind(".")].split("-")[-1])
        if epoch_idx < final_epoch:
            os.remove(ckpt)


def euclidean_dist(x, y, eps=1e-12, squared=False):
    m, n = x.shape[0], y.shape[0]
    xx = mx.nd.power(x, 2).sum(axis=1, keepdims=True).broadcast_to([m, n])
    yy = mx.nd.power(y, 2).sum(axis=1, keepdims=True).broadcast_to([n, m]).T
    dist = xx + yy
    dist = dist - 2 * mx.nd.dot(x, y.T)
    dist = mx.nd.clip(dist, eps, np.inf)
    return dist if not squared else mx.nd.sqrt(dist)

