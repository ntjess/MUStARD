from typing import Sequence, Any

import pandas as pd
import numpy as np
from skimage.exposure import equalize_hist
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

import ngram_heatmap
from plot_data import getSarcasmDf
from utilitys import widgets

def bigramSimilarity(condList: Sequence[str], sarcasmDf=None):
  if sarcasmDf is None:
    sarcasmDf = getSarcasmDf(context=True, returnBert=False)
  scores = pd.DataFrame(columns=condList, index=condList)
  imgs = {}
  allToks = np.unique(ngram_heatmap.tokenize(sarcasmDf['text'], 1))
  tokToIdxMapping = pd.Series(np.arange(len(allToks)), allToks)
  for cond in tqdm(condList, 'Getting bigram images'):
    df = sarcasmDf.query(cond)
    imgs[cond] = ngram_heatmap.makeBigramImage(
      df, 1, allToks=allToks, tokToIdxMapping=tokToIdxMapping)
  numConds = len(condList)
  ssimProgbar = tqdm(desc='Calculating ssim', total=int(((numConds+1)*numConds)/2))
  for ii, cond in enumerate(condList):
    for jj in range(ii+1):
      cmpCond = condList[jj]
      score = ssim(imgs[cond], imgs[cmpCond])
      score = np.round(score, 3)
      scores.at[cond, cmpCond] = score
      scores.at[cmpCond, cond] = score
      ssimProgbar.update()
  scores.to_csv('./data/bigram_sim_scores.csv')
  return scores

sar_ctxScores = bigramSimilarity(['context & sarcasm', 'context & (~sarcasm)',
                  '(~context) & sarcasm', '(~context) & (~sarcasm)'])
bp = 1