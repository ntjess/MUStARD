from collections import defaultdict
from typing import Any

import nltk
import pandas as pd
import numpy as np
import pyqtgraph as pg
import cv2 as cv
from skimage import morphology as morph

from plot_data import imgDir, getSarcasmDf, saveGrid

app = pg.mkQApp()


def tokenize(textSer: pd.Series, ngramSize=2):
  toTokenize = textSer.str.lower().replace('[^a-z ]', '', regex=True).str.strip()
  toTokenize = ('BEGIN ' + toTokenize + ' END').str.cat(sep=' ').split()
  return list(nltk.ngrams(toTokenize, ngramSize))

def makeFreqSer(corpus: pd.DataFrame, value: Any=None, whichCol='speaker'):
  if value:
    miniCorpus = corpus.loc[corpus[whichCol] == value, 'text']
  else:
    miniCorpus = corpus['text']
  valueNgrams = tokenize(miniCorpus)
  freqDict = defaultdict(int)
  for tok in valueNgrams: freqDict[tok] += 1
  # END BEGIN is common occurrence that is unnecessary / false
  freqDict.pop(('END', 'BEGIN'), None)
  return pd.Series(freqDict).sort_values(ascending=False)

def makePlots(df, imgDir=imgDir, whichCol='speaker', resizeRatio=0.25):
  uniqueLbls = np.unique(df[whichCol])
  numLbls = len(uniqueLbls)
  nrows = np.sqrt(numLbls).astype(int)
  ncols = np.ceil(numLbls/nrows)
  outGrid = pg.GraphicsLayoutWidget()

  allToks = np.unique(tokenize(df['text'], 1))
  tokToIdxMapping = pd.Series(np.arange(len(allToks)), allToks)
  imgSideLen = int(len(allToks)*resizeRatio)
  for ii, curLbl in enumerate(uniqueLbls):
    freqSer = makeFreqSer(df, curLbl, whichCol)
    pltItem: pg.PlotItem = outGrid.addPlot(title=curLbl)
    pltItem.getViewBox().setAspectLocked(True)
    curToks = np.vstack(freqSer.index)
    img = np.zeros((imgSideLen, imgSideLen), dtype='uint8')
    rowIdxs = (tokToIdxMapping[curToks[:,0]]*resizeRatio).astype(int)
    colIdxs = (tokToIdxMapping[curToks[:,1]]*resizeRatio).astype(int)
    rowIdxs = np.clip(rowIdxs, 0, imgSideLen-1)
    colIdxs = np.clip(colIdxs, 0, imgSideLen-1)
    img[rowIdxs, colIdxs] = freqSer.values
    img = cv.morphologyEx(img, cv.MORPH_DILATE, morph.disk(3)).astype(float)
    img = np.log10(img + 1)
    imgItem = pg.ImageItem(img)
    pltItem.addItem(imgItem)
    pltItem.vb.autoRange(padding=0)
    imgItem.save(str(imgDir/f'{curLbl} heatmap.jpg'))
    # pltItem.addLegend()
    # for clr in np.unique(curColors):
    #   pltItem.addItem(HoverScatter([], [], brush=clr, pen=clr, name=mapping[clr]))

    if ii % ncols == ncols-1:
        outGrid.nextRow()
  return outGrid

def makeAllHeatmapPlots(df: pd.DataFrame):
  origDf = df.copy()
  curDir = imgDir / 'sarcasm heatmaps'
  curDir.mkdir(exist_ok=True)

  for membership in df['sarcasm'], ~df['sarcasm']:
    grid = makePlots(df[membership], curDir)
    saveGrid(grid, curDir/'combined heatmaps.jpg')
    curDir = imgDir / 'not sarcasm heatmaps'
    curDir.mkdir(exist_ok=True)

  # Speaker by sarcasm
  curDir = imgDir / 'combined'
  curDir.mkdir(exist_ok=True)
  df = origDf
  grid = makePlots(df, curDir, 'sarcasm')
  saveGrid(grid, curDir / 'combined heatmaps.jpg')

def main():
  df = getSarcasmDf(returnBert=False)
  makeAllHeatmapPlots(df)

if __name__ == '__main__':
  main()