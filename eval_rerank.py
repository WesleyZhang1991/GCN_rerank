"""Eval for GCRV and other re-rerank method."""

import pickle
import time
from multiprocessing import Pool
import sys

import numpy as np
import tqdm

from config.defaults import get_cfg_defaults
from ecn import ECN
from gcrv import gcrv_image, gcrv_video, run_pvg
from lbr import LBR
from qe import QE
import utils.cmc as cmc
from utils.io import get_data, get_log, mergesetfeat

import argparse


def get_image_sims(_cfg, all_data):
    """Get image sims."""

    [prb_feats, prb_labels, _, gal_feats, gal_labels, _] = all_data
    if _cfg.COMMON.RERANK_TYPE == 'baseline':
        sims = cmc.ComputeEuclid(prb_feats, gal_feats, 1)
        print(sims)
    elif _cfg.COMMON.RERANK_TYPE == 'k_reciprocal':
        sims = cmc.ReRank(prb_feats, gal_feats)
    elif _cfg.COMMON.RERANK_TYPE == 'ecn':
        sims = ECN(prb_feats, gal_feats)
        sims = sims.T
    elif _cfg.COMMON.RERANK_TYPE == 'ecn_orig':
        sims = ECN(prb_feats, gal_feats, method='origdist')
        sims = sims.T
    elif _cfg.COMMON.RERANK_TYPE == 'gcrv_ecn':
        sims, prb_feats, gal_feats = gcrv_image(_cfg, all_data)
        sims = ECN(prb_feats, gal_feats)
        sims = sims.T
    elif _cfg.COMMON.RERANK_TYPE == 'lbr':
        sims = LBR(_cfg, prb_feats, gal_feats, prb_labels, gal_labels)
    elif _cfg.COMMON.RERANK_TYPE == 'qe':
        sims = QE(_cfg, prb_feats, gal_feats, prb_labels, gal_labels)
    elif _cfg.COMMON.RERANK_TYPE == 'gcrv':
        sims, _, _ = gcrv_image(_cfg, all_data)
    elif _cfg.COMMON.RERANK_TYPE == 'gcrv_post':
        sims, prb_feats, gal_feats = gcrv_image(_cfg, all_data)
        sims = cmc.ReRank(prb_feats, gal_feats, lambda_value=0.8)
    else:
        print('unknown rerank type')
        sims = None

    if _cfg.COMMON.VERBOSE:
        print('sims shape, ', sims.shape)
    return sims, gal_labels, prb_labels


def get_video_sims(_cfg, all_data):
    """Get video sims."""

    [prb_feats, prb_labels, prb_tracks, gal_feats, gal_labels,
     gal_tracks] = all_data
    if _cfg.COMMON.DATASET == 'mars':
        gal_feats = np.concatenate([gal_feats, prb_feats], axis=0)
        gal_labels = np.concatenate([gal_labels, prb_labels], axis=0)
        gal_tracks = np.concatenate([gal_tracks, prb_tracks], axis=0)
    # merge video data
    if 'gcrv' in _cfg.COMMON.RERANK_TYPE and _cfg.PVG.ENABLE_PVG:
        all_data = [prb_feats, prb_labels, prb_tracks,
                    gal_feats, gal_labels, gal_tracks]
        prb_feats, prb_labels, gal_feats, gal_labels = run_pvg(_cfg, all_data)
    else:
        prb_feats, prb_labels = mergesetfeat(prb_feats, prb_labels, prb_tracks)
        gal_feats, gal_labels = mergesetfeat(gal_feats, gal_labels, gal_tracks)

    if _cfg.COMMON.RERANK_TYPE == 'baseline':
        sims = cmc.ComputeEuclid(prb_feats, gal_feats, 1)
    elif _cfg.COMMON.RERANK_TYPE == 'k_reciprocal':
        sims = cmc.ReRank(prb_feats, gal_feats)
    elif _cfg.COMMON.RERANK_TYPE == 'ecn':
        sims = ECN(prb_feats, gal_feats)
        sims = sims.T
    elif _cfg.COMMON.RERANK_TYPE == 'ecn_orig':
        sims = ECN(prb_feats, gal_feats, method='origdist')
        sims = sims.T
    elif _cfg.COMMON.RERANK_TYPE == 'gcrv_ecn':
        all_data = [prb_feats, prb_labels, prb_tracks,
                    gal_feats, gal_labels, gal_tracks]
        sims, prb_feats, gal_feats = gcrv_video(_cfg, all_data)
        sims = ECN(prb_feats, gal_feats)
        sims = sims.T
    elif _cfg.COMMON.RERANK_TYPE == 'lbr':
        sims = LBR(_cfg, prb_feats, gal_feats, prb_labels, gal_labels)
    elif _cfg.COMMON.RERANK_TYPE == 'qe':
        sims = QE(_cfg, prb_feats, gal_feats, prb_labels, gal_labels)
    elif _cfg.COMMON.RERANK_TYPE == 'gcrv':
        all_data = [prb_feats, prb_labels, prb_tracks,
                    gal_feats, gal_labels, gal_tracks]
        sims, _, _ = gcrv_video(_cfg, all_data)
    elif _cfg.COMMON.RERANK_TYPE == 'gcrv_post':
        all_data = [prb_feats, prb_labels, prb_tracks,
                    gal_feats, gal_labels, gal_tracks]
        sims, prb_feats, gal_feats = gcrv_video(_cfg, all_data)
        sims = cmc.ReRank(prb_feats, gal_feats, lambda_value=0.8)
    else:
        print('unknown rerank type')
        sims = None

    return sims, gal_labels, prb_labels


def get_sims(_cfg, all_data):
    """Get sims."""

    start_time = time.time()
    if _cfg.COMMON.DATASET == 'mars':
        sims, gal_labels, prb_labels = get_video_sims(_cfg, all_data)
    else:
        sims, gal_labels, prb_labels = get_image_sims(_cfg, all_data)
    if _cfg.COMMON.VERBOSE:
        print(f'time for get sim: {time.time()-start_time}')
    return sims, gal_labels, prb_labels


def eval_rerank_methods(_cfg, all_data):
    """Eval rerank"""

    sims, gal_labels, prb_labels = get_sims(_cfg, all_data)
    # print(sims.shape, prb_labels.shape, gal_labels.shape)
    r_cmc = cmc.GetRanks(sims, prb_labels, gal_labels, 10, True)
    r_map = cmc.GetMAP(sims, prb_labels, gal_labels, True)

    log_name = f'./LOG/{_cfg.COMMON.DATASET}/{_cfg.COMMON.RERANK_TYPE}.log'
    root_logger = get_log(log_name)
    log_str = f'eval {_cfg.COMMON.DATASET} with {_cfg.COMMON.RERANK_TYPE}'
    root_logger.info(log_str)
    root_logger.info(_cfg)
    result = 'rank1: %.2f  mAP: %.2f' % (r_cmc[0]*100.0, r_map*100.0)
    root_logger.info(result)
    root_logger.info('-----------------------------------------------')


def main():
    """Main method."""

    parser = argparse.ArgumentParser(description="Rerank Evaluation")
    parser.add_argument(
        "--config_file", default="", help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)
    args = parser.parse_args()

    cfg = get_cfg_defaults()
    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    test_probe_feature = pickle.load(open(cfg.COMMON.PRB_PKL, 'rb'),
                                     encoding='latin1')
    test_gallery_feature = pickle.load(open(cfg.COMMON.GAL_PKL, 'rb'),
                                       encoding='latin1')
    gal_feats, gal_labels, gal_tracks = get_data(test_gallery_feature)
    prb_feats, prb_labels, prb_tracks = get_data(test_probe_feature)
    all_data = [prb_feats, prb_labels, prb_tracks,
                gal_feats, gal_labels, gal_tracks]

    eval_rerank_methods(cfg, all_data)


if __name__ == '__main__':
    main()
