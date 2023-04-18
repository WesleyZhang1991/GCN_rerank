from __future__ import print_function, division, absolute_import
import numpy as np

from sklearn.preprocessing import normalize
from tqdm import tqdm

# pylint: disable=invalid-name, too-many-locals
def QE(_cfg, query_features, gallery_features, prb_labels, gal_labels):
    """QE."""

    top_k = 5

    # l2 normalize
    query_features = normalize(query_features)
    gallery_features = normalize(gallery_features)

    # Start to propagation
    for i in tqdm(range(query_features.shape[0])):
        q_feat = query_features[i]

        # Initial P2G affinity vector
        y_0 = np.dot(gallery_features, q_feat)
        rank_index = np.argsort(-y_0)
        top_k_index = rank_index[:top_k]
        g_feats = gallery_features[top_k_index, :]
        g_feats = np.concatenate([g_feats, q_feat[np.newaxis, :]], axis=0)
        # print(g_feats.shape)
        query_features[i] = np.mean(g_feats, axis=0)

    gallery_features = normalize(gallery_features)
    data = np.vstack((query_features, gallery_features))
    labels = np.concatenate((prb_labels, gal_labels))
    q_g_dist = 1.0 - np.dot(query_features, gallery_features.T)

    return q_g_dist


if __name__ == '__main__':
    pass
