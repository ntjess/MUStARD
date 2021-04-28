from collections import defaultdict
from typing import Any

import nltk
import pandas as pd
import numpy as np
import re
import pyqtgraph as pg
import cv2 as cv
from skimage import morphology as morph

from plot_data import imgDir, makeSarcasmRatioPlot

file = pd.read_csv(r'/home/ntjess/Desktop/Git/MUStARD/data/sarcasm_data.csv')

df = pd.DataFrame(file, columns= ['text','speaker', 'sarcasm'])

def tokenize(textSer: pd.Series, ngramSize=2):
  toTokenize = textSer.str.lower().replace('[^a-z ]', '', regex=True).str.strip()
  toTokenize = ('BEGIN ' + toTokenize + ' END').str.cat(sep=' ').split()
  return list(nltk.ngrams(toTokenize, ngramSize))

def makeFreqSer(value: Any=None, corpus: pd.DataFrame=df, whichCol='speaker'):
  if value:
    miniCorpus = corpus.loc[df[whichCol] == value, 'text']
  else:
    miniCorpus = corpus['text']
  valueNgrams = tokenize(miniCorpus)
  freqDict = defaultdict(int)
  for tok in valueNgrams: freqDict[tok] += 1
  # END BEGIN is common occurrence that is unnecessary / false
  freqDict.pop(('END', 'BEGIN'), None)
  return pd.Series(freqDict).sort_values(ascending=False)

allBigrams = makeFreqSer()

def makePlots(df=df, imgDir=imgDir, whichCol='speaker'):
  uniqueLbls = np.unique(df[whichCol])
  numLbls = len(uniqueLbls)
  nrows = np.sqrt(numLbls).astype(int)
  ncols = np.ceil(numLbls/nrows)
  outGrid = pg.GraphicsLayoutWidget()

  allToks = np.unique(tokenize(df['text'], 1))
  tokToIdxMapping = pd.Series(np.arange(len(allToks)), allToks)
  imgDownsize = 0.15
  imgSideLen = int(len(allToks)*imgDownsize)
  for ii, curLbl in enumerate(uniqueLbls):
    freqSer = makeFreqSer(curLbl, df, whichCol)
    pltItem: pg.PlotItem = outGrid.addPlot(title=curLbl)
    curToks = np.vstack(freqSer.index)
    img = np.zeros((imgSideLen, imgSideLen), dtype='uint8')
    rowIdxs = (tokToIdxMapping[curToks[:,0]]*imgDownsize).astype(int)
    colIdxs = (tokToIdxMapping[curToks[:,1]]*imgDownsize).astype(int)
    rowIdxs = np.clip(rowIdxs, 0, imgSideLen-1)
    colIdxs = np.clip(colIdxs, 0, imgSideLen-1)
    img[rowIdxs, colIdxs] = freqSer.values
    img = cv.morphologyEx(img, cv.MORPH_DILATE, morph.disk(3)).astype(float)
    img = np.log10(img + 1)
    imgItem = pg.ImageItem(img)
    pltItem.addItem(imgItem)
    imgItem.save(str(imgDir/f'{curLbl} heatmap.jpg'))
    # pltItem.addLegend()
    # for clr in np.unique(curColors):
    #   pltItem.addItem(HoverScatter([], [], brush=clr, pen=clr, name=mapping[clr]))

    if ii % ncols == ncols-1:
        outGrid.nextRow()
  return outGrid

if __name__ == '__main__':
  app = pg.mkQApp()
  sarcasmRatio = makeSarcasmRatioPlot(df, False)
  badSpeakers = sarcasmRatio[(sarcasmRatio < 0.001) | (sarcasmRatio > 0.999)].index
  membership = ~np.isin(df['speaker'], badSpeakers)
  # Sarcasm by speaker
  origDf = df
  df = df[membership]
  curDir = imgDir / 'sarcasm heatmaps'
  curDir.mkdir(exist_ok=True)
  # for membership in df['sarcasm'], ~df['sarcasm']:
  #   grid = makePlots(df[membership], curDir)
  #   fullSize = app.primaryScreen().size()
  #   grid.resize(fullSize)
  #   pic = grid.grab()
  #   assert pic.save(str(curDir/'speaker heatmaps.jpg'))
  #   curDir = imgDir / 'not sarcasm heatmaps'
  #   curDir.mkdir(exist_ok=True)

  # Speaker by sarcasm
  curDir = imgDir/'combined'
  curDir.mkdir(exist_ok=True)
  df = origDf
  grid = makePlots(df, curDir, 'sarcasm')
  fullSize = app.primaryScreen().size()
  grid.resize(fullSize)
  pic = grid.grab()
  assert pic.save(str(curDir/'speaker heatmaps.jpg'))