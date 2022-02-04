"""Eval for GCR."""

import time
import numpy as np
import torch

import utils.cmc as cmc
from scipy.spatial import distance

# pylint: disable=invalid-name, too-many-locals, line-too-long
def mergesetfeat4(_cfg, X, labels):
    """Run GCR for one iteration."""

    FIRST_MERGE = True
    start_time = time.time()
    labels_cam = labels[:, 1]
    unique_labels_cam = np.unique(labels_cam)
    index_dic = {item: [] for item in unique_labels_cam}
    for labels_index, item in enumerate(labels_cam):
        index_dic[item].append(labels_index)

    beta1 = _cfg.GCR.BETA1
    beta2 = _cfg.GCR.BETA2
    k1 = _cfg.GCR.K1
    k2 = _cfg.GCR.K2
    scale = _cfg.GCR.SCALE
    # print('K1 is {},  beta1 is {}, K2 is {}, beta2 is {}, scale is {}\n'.format(k1, beta1, k2, beta2, scale))

    # compute global feat
    if _cfg.GCR.MODE == 'fixA' and FIRST_MERGE:
        sim = X.dot(X.T)
        np.save('temp', sim)
        FIRST_MERGE = False
    elif _cfg.GCR.MODE == 'fixA' and not FIRST_MERGE:
        sim = np.load('temp.npy')
    else:
        sim = X.dot(X.T)
        if _cfg.GCR.MODE == 'no-norm':
            X2 = np.square(X).sum(axis=1)
            r2 = -2. * np.dot(X, X.T) + X2[:, None] + X2[None, :]
            dist = np.clip(r2, 0., float(np.inf))

    if scale != 1.0:
        if _cfg.GCR.WITH_GPU:
            rank = gpu_argsort(-sim)
        else:
            rank = np.argsort(-sim, axis=1)    # time consuming
        S = np.zeros(sim.shape)
        for i in range(0, X.shape[0]):
            if _cfg.GCR.MODE != 'no-norm':
                S[i, rank[i, :k1]] = np.exp(sim[i, rank[i, :k1]]/beta1)
                S[i, i] = np.exp(sim[i, i]/beta1)    # this is duplicated???
            else:
                S[i, rank[i, :k1]] = np.exp(-(dist[i, rank[i, :k1]])/beta1)
                S[i, i] = np.exp(-0.0/beta1)    # this is duplicated???
        # TODO: not sure if this equals to draft...
        if _cfg.GCR.MODE == 'sym':
            S = 0.5 * (S + S.T)
        temp = np.sum(S, axis=1)
        # print(temp.min(), temp.max(), temp.shape)
        temp = np.sum(S, axis=0)
        # print(temp.min(), temp.max(), temp.shape)
        D_row = np.sqrt(1. / np.sum(S, axis=1))
        D_col = np.sqrt(1. / np.sum(S, axis=0))
        L = np.outer(D_row, D_col) * S
        global_X = L.dot(X)
    else:
        global_X = 0.0

    if scale != 0.0:
        # compute cross camera feat
        for i in range(0, X.shape[0]):
            tmp = sim[i, i]
            sim[i, index_dic[labels[i, 1]]] = -2
            sim[i, i] = tmp
        if _cfg.GCR.MODE == 'no-norm':
            for i in range(0, X.shape[0]):
                tmp = dist[i, i]
                dist[i, index_dic[labels[i, 1]]] = 1000
                dist[i, i] = tmp

        if _cfg.GCR.WITH_GPU:
            rank = gpu_argsort(-sim)
        else:
            rank = np.argsort(-sim, axis=1)    # time consuming
        S = np.zeros(sim.shape)
        for i in range(0, X.shape[0]):
            if _cfg.GCR.MODE != 'no-norm':
                S[i, rank[i, :k2]] = np.exp(sim[i, rank[i, :k2]] / beta2)
                S[i, i] = np.exp(sim[i, i]/beta2)    # this should not be ommited
            else:
                S[i, rank[i, :k2]] = np.exp(-dist[i, rank[i, :k2]] / beta2)
                S[i, i] = np.exp(-0.0/beta2)    # this should not be ommited
        if _cfg.GCR.MODE == 'sym':
            S = 0.5 * (S + S.T)
        D_row = np.sqrt(1. / np.sum(S, axis=1))
        D_col = np.sqrt(1. / np.sum(S, axis=0))
        L = np.outer(D_row, D_col) * S
        cross_X = L.dot(X)
    else:
        cross_X = 0.0

    X = scale*cross_X+(1-scale)*global_X
    if _cfg.GCR.MODE != 'no-norm':
        X /= np.linalg.norm(X, ord=2, axis=1, keepdims=True)
    if _cfg.COMMON.VERBOSE:
        print(f'round time {time.time()-start_time} s')
    return X

def mergesetfeat4_localk(_cfg, X, labels):
    """Run GCR for one iteration."""

    labels_cam = labels[:, 1]
    unique_labels_cam = np.unique(labels_cam)
    index_dic = {item: [] for item in unique_labels_cam}
    for labels_index, item in enumerate(labels_cam):
        index_dic[item].append(labels_index)

    cross_index_dic = {}
    full_index = np.arange(0, len(labels))
    for item in index_dic:
        #cross_index_dic[item] = full_index
        cross_index_dic[item] = np.setdiff1d(full_index, index_dic[item])
    beta1 = _cfg.GCR.BETA1
    beta2 = _cfg.GCR.BETA2
    k1 = _cfg.GCR.K1
    k2 = _cfg.GCR.K2
    scale = _cfg.GCR.SCALE
    # print('K1 is {},  beta1 is {}, K2 is {}, beta2 is {}, scale is {}\n'.format(k1, beta1, k2, beta2, scale))
    x_sim = X.dot(X.T)
    for i in range(0, X.shape[0]):
        cross_indexes = cross_index_dic[labels[i, 1]]
        good_sim_indexes = np.argwhere(x_sim[:, i] > 0).flatten()
        cross_knn_indexes = np.intersect1d(cross_indexes, good_sim_indexes)
        global_knn_indexes = np.intersect1d(full_index, good_sim_indexes)
        global_knn_indexes = np.setdiff1d(global_knn_indexes, i)
        cross_sim = x_sim[cross_knn_indexes, i]
        global_sim = x_sim[global_knn_indexes, i]
        # compute global feat
        idx = np.argsort(-global_sim)
        selected_knn_indexes = np.concatenate((global_knn_indexes[idx[:k1]], np.array(i).reshape(-1)))
        knnX = X[selected_knn_indexes, :]
        S = x_sim[selected_knn_indexes, :]
        S = S[:, selected_knn_indexes]
        S = np.exp(S / beta1)
        D = np.sqrt(1. / np.sum(S, axis=1))
        L = np.outer(D, D) * S
        global_feat = L[-1, :].dot(knnX)

        # compute cross feat
        idx = np.argsort(-cross_sim)
        selected_knn_indexes = np.concatenate((cross_knn_indexes[idx[:k2]], np.array(i).reshape(-1)))
        knnX = X[selected_knn_indexes, :]
        S = x_sim[selected_knn_indexes, :]
        S = S[:, selected_knn_indexes]
        S = np.exp(S / beta2)
        D = np.sqrt(1. / np.sum(S, axis=1))
        L = np.outer(D, D) * S
        cross_feat = L[-1, :].dot(knnX)

        X[i, :] = scale * cross_feat + (1 - scale) * global_feat
    X /= np.linalg.norm(X, ord=2, axis=1, keepdims=True)
    return X


def mergesetfeat1_notrk(_cfg, P, neg_vector, in_feats, in_labels):
    """Mergesetfeat1 notrk"""

    out_feats = []
    for i in range(in_feats.shape[0]):
        camera_id = in_labels[i, 1]
        if _cfg.PVG.OPERATION == 'P_neg':
            feat = in_feats[i] - neg_vector[camera_id]
            feat = P[camera_id].dot(feat)
        elif _cfg.PVG.OPERATION == 'neg':
            feat = in_feats[i] - neg_vector[camera_id]
        elif _cfg.PVG.OPERATION == 'P':
            feat = in_feats[i]
            feat = P[camera_id].dot(feat)
        elif _cfg.PVG.OPERATION == 'none':
            feat = in_feats[i]

        # random neg vec
        # rand_vec = np.random.random((512,))
        # rand_vec = rand_vec / np.linalg.norm(rand_vec, ord=2) * 0.1
        # feat = in_feats[i] - rand_vec

        feat = feat/np.linalg.norm(feat, ord=2)
        out_feats.append(feat)
    out_feats = np.vstack(out_feats)
    return out_feats


def mergesetfeat1(_cfg, P, neg_vector, in_feats, in_labels, in_tracks):
    """Run PVG."""

    trackset = np.unique(in_tracks)
    # trackset = list(set(list(in_tracks)))
    out_feats = []
    out_labels = []
    track_index_dic = {item: [] for item in trackset}
    for track_index, item in enumerate(in_tracks):
        track_index_dic[item].append(track_index)

    for track in trackset:
        indexes = track_index_dic[track]
        camera_id = in_labels[indexes, 1][0]

        if _cfg.PVG.OPERATION == 'P_neg':
            feat = np.mean(in_feats[indexes], axis=0) - neg_vector[camera_id]
            feat = P[camera_id].dot(feat)
        elif _cfg.PVG.OPERATION == 'neg':
            feat = np.mean(in_feats[indexes], axis=0) - neg_vector[camera_id]
        elif _cfg.PVG.OPERATION == 'P':
            feat = np.mean(in_feats[indexes], axis=0)
            feat = P[camera_id].dot(feat)
        elif _cfg.PVG.OPERATION == 'none':
            feat = np.mean(in_feats[indexes], axis=0)

        feat = feat/np.linalg.norm(feat, ord=2)
        label = in_labels[indexes][0]
        out_feats.append(feat)
        out_labels.append(label)
        # if len(out_feats) % 100 == 0:
        #     print('%d/%d' %(len(out_feats), len(trackset)))
    out_feats = np.vstack(out_feats)
    out_labels = np.vstack(out_labels)
    return out_feats, out_labels


def compute_P_all(gal_feats, gal_labels, la):
    """Compute P and neg for all data(global cameras)."""

    X = gal_feats
    neg_vector_all = np.mean(X, axis=0).astype('float32')
    # la = 0.04
    P_all = np.linalg.inv(X.T.dot(X)+X.shape[0]*la*np.eye(X.shape[1])).astype('float32')
    neg_vector = {}
    u_labels = np.unique(gal_labels[:, 1])
    P = {}
    for label in u_labels:
        P[label] = P_all
        neg_vector[label] = neg_vector_all
    return P, neg_vector

def compute_P2(gal_feats, gal_labels, la):
    """Compute P2 on gal data????"""

    X = gal_feats
    neg_vector = {}
    u_labels = np.unique(gal_labels[:, 1])
    P = {}
    for label in u_labels:
        curX = gal_feats[gal_labels[:, 1] == label, :]
        # curX = gal_feats
        neg_vector[label] = np.mean(curX, axis=0).astype('float32')
        P[label] = np.linalg.inv(curX.T.dot(curX)+curX.shape[0]*la*np.eye(X.shape[1])).astype('float32')
    return P, neg_vector


def run_pvg(_cfg, all_data):
    """Run pvg."""

    # start_time = time.time()
    [prb_feats, prb_labels, prb_tracks, gal_feats, gal_labels, gal_tracks] = all_data
    if _cfg.PVG.ENABLE_PVG:
        if _cfg.PVG.STATICS_LEVEL == 'intra_camera':
            P, neg_vector = compute_P2(gal_feats, gal_labels, la=_cfg.PVG.LA)
        elif _cfg.PVG.STATICS_LEVEL == 'all':
            P, neg_vector = compute_P_all(gal_feats, gal_labels, la=_cfg.PVG.LA)

        if _cfg.COMMON.DATASET == 'mars':
            prb_feats, prb_labels = mergesetfeat1(_cfg, P, neg_vector, prb_feats, prb_labels, prb_tracks)
            gal_feats, gal_labels = mergesetfeat1(_cfg, P, neg_vector, gal_feats, gal_labels, gal_tracks)
        else:
            prb_feats = mergesetfeat1_notrk(_cfg, P, neg_vector, prb_feats, prb_labels)
            gal_feats = mergesetfeat1_notrk(_cfg, P, neg_vector, gal_feats, gal_labels)
    # print(f'{time.time() - start_time} s for PVG')
    return prb_feats, prb_labels, gal_feats, gal_labels


def run_gcr(_cfg, all_data):
    """Run GCR."""

    [prb_feats, prb_labels, _, gal_feats, gal_labels, _] = all_data
    prb_n = len(prb_labels)
    data = np.vstack((prb_feats, gal_feats))
    labels = np.concatenate((prb_labels, gal_labels))
    if _cfg.GCR.ENABLE_GCR:
        for gal_round in range(_cfg.GCR.GAL_ROUND):
            if _cfg.GCR.MODE != 'localk':
                data = mergesetfeat4(_cfg, data, labels)
            else:
                data = mergesetfeat4_localk(_cfg, data, labels)
    prb_feats_new = data[:prb_n, :]
    gal_feats_new = data[prb_n:, :]
    return prb_feats_new, gal_feats_new


def gcrv_image(_cfg, all_data):
    """GCRV image port."""

    [prb_feats, prb_labels, prb_tracks, gal_feats, gal_labels, gal_tracks] = all_data
    prb_feats, prb_labels, gal_feats, gal_labels = run_pvg(_cfg, all_data)
    all_data = [prb_feats, prb_labels, prb_tracks, gal_feats, gal_labels, gal_tracks]
    prb_feats, gal_feats = run_gcr(_cfg, all_data)
    sims = cmc.ComputeEuclid(prb_feats, gal_feats, 1)
    return sims, prb_feats, gal_feats


def gcrv_video(_cfg, all_data):
    """GCRV video port."""

    [prb_feats, _, _, gal_feats, _, _] = all_data
    prb_feats, gal_feats = run_gcr(_cfg, all_data)
    sims = cmc.ComputeEuclid(prb_feats, gal_feats, 1)
    return sims, prb_feats, gal_feats


def gpu_argsort(temp):
    """Use torch for faster argsort."""

    temp = torch.from_numpy(temp).to('cuda').half()
    rank = torch.argsort(temp, dim=1).cpu().numpy()
    return rank


def debug_time():
    """Find faster argsort op."""

    temp = np.random.random((20000, 20000)).astype('float32')
    start_time = time.time()
    rank_1 = np.argsort(-temp, axis=1)
    print(rank_1[:10])
    print(time.time()-start_time)

    start_time = time.time()
    rank_2 = gpu_argsort(-temp)
    print(rank_2[:10])
    print(time.time()-start_time)
    np.testing.assert_array_equal(rank_1, rank_2)


if __name__ == '__main__':
    debug_time()
