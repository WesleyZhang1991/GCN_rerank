

from __future__ import print_function, division, absolute_import
import numpy as np

from sklearn.preprocessing import normalize
from tqdm import tqdm

from utils.evaluation import eval_rank_list

# pylint: disable=invalid-name, too-many-locals
def eval_func(ranklist, q_pids, g_pids, q_camids, g_camids, max_rank=50):
    """Evaluation with market1501 metric
        Key: for each query identity, its gallery images from the same camera view are discarded.
    """
    num_q = len(ranklist)
    num_g = len(ranklist[0])
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))

    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0.  # number of valid query
    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        line_ranklist = ranklist[q_idx]
        orig_cmc = []
        for temp_g_index in line_ranklist:
            if g_pids[int(temp_g_index)] == q_pid and g_camids[int(temp_g_index)] != q_camid:
                orig_cmc.append(1)
            if g_pids[int(temp_g_index)] != q_pid:
                orig_cmc.append(0)

        # compute cmc curve
        orig_cmc = np.array(orig_cmc)
        debug_str = f'q_idx: {q_idx}, q_pid: {q_pid}, q_camid: {q_camid}, '

        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        if num_rel == 0:
            AP = 0.0
        else:
            AP = tmp_cmc.sum() / num_rel
        debug_str += f'AP: {AP}, num_rel: {num_rel}'
        # print(debug_str)
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)
    print(all_cmc, mAP)
    return all_cmc, mAP


def LBR(_cfg, query_features, gallery_features, prb_labels, gal_labels):
    """LBR."""

    # for MSMT
    # temperature = 0.02
    # top_k = 150
    temperature = 0.3
    top_k = 100

    # l2 normalize
    query_features = normalize(query_features)
    gallery_features = normalize(gallery_features)

    # Start to propagation
    rank_list = []
    for i in tqdm(range(query_features.shape[0])):
        q_feat = query_features[i]

        # Initial P2G affinity vector
        y_0 = np.dot(gallery_features, q_feat)

        rank_index = np.argsort(-y_0)
        top_k_index = rank_index[:top_k]

        g_feats = gallery_features[top_k_index, :]

        # G2G affinity matrix
        W = np.dot(g_feats, g_feats.T)
        W = np.exp(W / temperature)
        W = W / W.sum(axis=1, keepdims=True)

        g_feats = np.dot(W, g_feats)

        # recompute top-k ranking list
        y = np.dot(g_feats, q_feat)

        rank_index[:top_k] = top_k_index[np.argsort(-y)]
        rank_list.append(rank_index)

    data = np.vstack((query_features, gallery_features))
    labels = np.concatenate((prb_labels, gal_labels))

    # evaluation by original inplementation.
    OFFICIAL = False
    if OFFICIAL:
        print("Evaluating...")
        query_cam_ids = prb_labels[:, 1]
        gallery_cam_ids = gal_labels[:, 1]
        query_ids = prb_labels[:, 0]
        gallery_ids = gal_labels[:, 0]
        # this might be wrong...
        eval_rank_list(rank_list, query_ids, query_cam_ids, gallery_ids, gallery_cam_ids)
        # eval_func(rank_list, query_ids, gallery_ids, query_cam_ids, gallery_cam_ids, max_rank=50)
    else:
        # to cope with our framework.
        q_num = query_features.shape[0]
        g_num = gallery_features.shape[0]
        q_g_dist = np.zeros((q_num, g_num), dtype=np.float32) + 10.0
        for i, rank_index in enumerate(rank_list):
            for k in range(top_k):
                dist = (k+1) / float(top_k)
                j = rank_index[k]
                q_g_dist[i, j] = dist
        return q_g_dist

if __name__ == '__main__':
    pass
