# Graph Convolution Based Efficient Re-Ranking for Visual Retrieval, submitted to TMM
The official repository for GCR rerank, a GCN-based reranking method for image re-ID, video re-ID, and image retrieval.

## Environment

We use python 3.7/torch 1.6/torchvision 0.7.0.

## Datasets
image re-ID: Market, Duke, MSMT, CUHK03

video re-ID: MARS

image retrieval: ROxford, RParis

## Extracted features
We provide Market1501/MARS features from [reid-strong-baseline](https://github.com/michuanhaohao/reid-strong-baseline) at [Google Drive](https://drive.google.com/drive/folders/13pzDLmdbal2SpVCIaa4yczx7aPJK8yVx?usp=share_link).

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
