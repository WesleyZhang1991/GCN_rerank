"""Eval for GCRV."""

import pickle
import os
import logging
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn import manifold

# pylint: disable=invalid-name, invalid-name, too-many-locals, line-too-long
def get_log(log_name):
    """Get log."""

    log_dir = os.path.dirname(log_name)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
    rootLogger = logging.getLogger()
    fileHandler = logging.FileHandler(log_name, 'w+')    # w+ for overwrite
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)
    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    rootLogger.addHandler(consoleHandler)
    rootLogger.setLevel(logging.INFO)
    return rootLogger


def mergesetfeat(in_feats, in_labels, in_tracks):
    """Just mean feature of a tracklet, baseline."""

    trackset = np.unique(in_tracks)
    out_feats = []
    out_labels = []
    track_index_dic = {item: [] for item in trackset}
    for track_index, item in enumerate(in_tracks):
        track_index_dic[item].append(track_index)

    # for track in sorted(trackset):
    for track in trackset:
        indexes = track_index_dic[track]
        feat = np.mean(in_feats[indexes], axis=0)    # these lines too slow
        feat = feat/np.linalg.norm(feat, ord=2)
        label = in_labels[indexes][0]
        # print(in_labels[indexes])
        out_feats.append(feat)
        out_labels.append(label)
    out_feats = np.vstack(out_feats)
    out_labels = np.vstack(out_labels)
    return out_feats, out_labels


def get_data(feature_dic):
    """Get image/video data."""

    data_feature = []
    data_label = []
    data_track = []
    set_id = set()
    for file_name in sorted(feature_dic.keys()):
        short_name = '_'.join(file_name.split('.')[0].split('_')[:3])
        feat = feature_dic[file_name]
        feat = feat/np.linalg.norm(feat, ord=2)
        try:
            rawid, cam, _ = short_name.split('_')
            rawid = int(rawid)
            cam = int(cam)
        except ValueError:    # distractor
            rawid = -1
            cam = -1
        set_id.add(rawid)
        data_feature.append(feat)
        data_label.append([rawid, cam])
        data_track.append(short_name)
    data_feature = np.array(data_feature)
    return data_feature, np.array(data_label), np.array(data_track)


def main():
    """Main method."""

    config_file = './config/market.yml'
    cfg.merge_from_file(config_file)
    cfg.freeze()

    test_probe_feature = pickle.load(open(cfg.COMMON.PRB_PKL, 'rb'), encoding='latin1')
    test_gallery_feature = pickle.load(open(cfg.COMMON.GAL_PKL, 'rb'), encoding='latin1')
    if cfg.COMMON.DATASET == 'mars':
        test_gallery_feature.update(test_probe_feature)
    prb_feats, prb_labels, prb_tracks = get_data(test_probe_feature)
    gal_feats, gal_labels, gal_tracks = get_data(test_gallery_feature)

    prb_feats, prb_labels = mergesetfeat(prb_feats, prb_labels, prb_tracks)
    gal_feats, gal_labels = mergesetfeat(gal_feats, gal_labels, gal_tracks)


if __name__ == '__main__':
    main()
