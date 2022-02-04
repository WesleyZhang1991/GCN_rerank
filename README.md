# [ICASSP] Graph Convolution for Re-ranking in Person Re-identification
The official repository for GCR rerank, a GCN-based reranking method for both image and video re-ID.

## Environment

We use python 3.7/torch 1.6/torchvision 0.7.0.

## Extracted features
We provide Market1501/MARS features from [reid-strong-baseline](https://github.com/michuanhaohao/reid-strong-baseline) at [Google Drive](https://drive.google.com/drive/folders/1iYt4n88dv2Bmn5ccOfhYo4aVCZUgWIaB?usp=sharing).

## Command Lines
Run GCRV rerank with basic settings on Market1501
```
python eval_rerank.py --config_file=config/market.yml
```
Run PVG only
```
python eval_rerank.py --config_file=config/market.yml PVG.ENABLE_PVG True GCR.ENABLE_GCR False
```
Run GCR only
```
python eval_rerank.py --config_file=config/market.yml PVG.ENABLE_PVG False GCR.ENABLE_GCR True
```
RUN GCRV on video reid dataset(MARS)
```
python eval_rerank.py --config_file=config/mars.yml
```
Run other rerank methods: (baseline, k_reciprocal, ecn, ecn_orig, lbr, qe)
```
python eval_rerank.py --config_file=config/market.yml COMMON.RERANK_TYPE baseline
```

## Thanks
State-of-the-art reranking method inlucidng [K_reciprocal](https://github.com/zhunzhong07/person-re-ranking), [ECN](https://github.com/pse-ecn/expanded-cross-neighborhood), [LBR](https://github.com/CoinCheung/SFT-ReID)

## Citation

If you find our work useful in your research, please consider citing:
```
@inproceedings{zhang2022graph,
 title={Graph Convolution for Re-ranking in Person Re-identification},
 author={Zhang, Yuqi and Qian Qi and Liu Chong and Chen, Weihua and Wang Fan and Li Hao and Jin Rong},
 journal={arXiv preprint arXiv:2107.02220},
 year={2022}
}
```
