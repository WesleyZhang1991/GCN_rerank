from yacs.config import CfgNode as CN

_C = CN()
# -----------------------------------------------------------------------------
# COMMON_PARAM
# -----------------------------------------------------------------------------
_C.COMMON = CN()
_C.COMMON.DATASET = 'mars'
_C.COMMON.RERANK_TYPE = 'gcrv'    # 'baseline', 'gcrv', 'k_reciprocal', 'ecn'
_C.COMMON.GAL_PKL = "./features/mars/test_gallery_reid.pkl"
_C.COMMON.PRB_PKL = "./features/mars/test_probe_reid.pkl"
_C.COMMON.USE_RERANK = False
_C.COMMON.VERBOSE = True

# -----------------------------------------------------------------------------
# PVG PARAM
# -----------------------------------------------------------------------------
_C.PVG = CN()
_C.PVG.ENABLE_PVG = True
_C.PVG.CACHE_P = False
_C.PVG.LA = 0.06
_C.PVG.STATICS_LEVEL = 'intra_camera'    # 'intra_camera', 'all'
_C.PVG.OPERATION = 'P_neg'    # 'P', 'neg', 'P_neg'

# -----------------------------------------------------------------------------
# GCR PARAM
# -----------------------------------------------------------------------------
_C.GCR = CN()
_C.GCR.ENABLE_GCR = False
_C.GCR.BETA1 = 0.08
_C.GCR.BETA2 = 0.08
_C.GCR.SCALE = 0.7
_C.GCR.K1 = 45
_C.GCR.K2 = 90
_C.GCR.GAL_ROUND = 3
_C.GCR.MODE = 'nonsym'
_C.GCR.WITH_GPU = True

def get_cfg_defaults():
  """Get a yacs CfgNode object with default values for my_project."""
  # Return a clone so that the defaults will not be altered
  # This is for the "local variable" use pattern
  return _C.clone()
