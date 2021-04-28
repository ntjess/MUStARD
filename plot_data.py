# from nltk.grammar import cfg_demo
# from data_loader import DataLoader
from pathlib import Path

import pyqtgraph as pg

app = pg.mkQApp()
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import random_projection
from sklearn.cluster import FeatureAgglomeration
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from config import Config

from utilitys.widgets import HoverScatter
from utilitys import PrjParam
from ast import literal_eval
import pickle as pkl

imgDir = Path('./proj_images')

def saveGrid(grid: pg.GraphicsLayoutWidget, saveFile):
  fullSize = app.primaryScreen().size()
  grid.resize(fullSize)
  pic = grid.grab()
  assert pic.save(str(saveFile))

def makeDataPlots(plotFeats, sarcasmDf, labelCol, colorCol, featureType):
  # -----
  # SARCASM PER SPEAKER PLOTS
  # -----
  plotFeats = plotFeats[:, :2].T
  labels = sarcasmDf[labelCol].to_numpy()
  colors = sarcasmDf[colorCol].to_numpy()
  if len(colors) == 0:
    return
  colors, mapping = PrjParam(name=colorCol).toNumeric(colors, offset=False, returnMapping=True)
  uniqueLbls = np.unique(labels)

  numLbls = len(uniqueLbls)
  nrows = np.sqrt(numLbls).astype(int)
  ncols = np.ceil(numLbls/nrows)
  outGrid = pg.GraphicsLayoutWidget(title=f'{colorCol.title()} per {labelCol.title()}, {featureType}')

  for ii, curLbl in enumerate(uniqueLbls):
    membership = labels == curLbl
    data = plotFeats[:, membership]
    curColors = colors[membership]
    pltItem: pg.PlotItem = outGrid.addPlot(title=curLbl)
    spi = HoverScatter(*data, pen=curColors, brush=curColors)
    pltItem.setLabel('bottom', 'Feature 1')
    pltItem.setLabel('left', 'Feature 2')
    pltItem.addItem(spi)
    # pltItem.addLegend()
    # for clr in np.unique(curColors):
    #   pltItem.addItem(HoverScatter([], [], brush=clr, pen=clr, name=mapping[clr]))

    if ii % ncols == ncols-1:
        outGrid.nextRow()

  return outGrid

def makeSarcasmRatioPlot(sarcasmDf, save=True):
  speakers = np.unique(sarcasmDf['speaker'])
  ratioData = {}
  for speaker in speakers:
    speakerDf = sarcasmDf.loc[sarcasmDf['speaker'] == speaker, 'sarcasm']
    ratioData[speaker] = speakerDf.sum()/len(speakerDf)
  ratioSer = pd.Series(ratioData).sort_values(ascending=False)
  if save:
    ratioSer.plot.bar()
    plt.title('Ratio of sarcastic to non-sarcastic sentences')
    plt.tight_layout()
    plt.savefig(imgDir/'sarcasmRatio.pdf')
  return ratioSer

def makeSpeakerGridPlots(sarcasmDf, bertFeats=None, show=False):
  tformFile = './data/transformData.pkl'
  if bertFeats is None:
    with open(tformFile, 'rb') as ifile:
      dataMap = pkl.load(ifile)
  else:
    print('Regenerating transform data...')
    dataMap = {
      'PCA': PCA().fit_transform(bertFeats),
      'TSNE': TSNE().fit_transform(bertFeats),
      'Agglomeration': FeatureAgglomeration().fit_transform(bertFeats),
      'Gaussian Projection': random_projection.GaussianRandomProjection(2).fit_transform(bertFeats),
      'Sparse Projection': random_projection.SparseRandomProjection(2).fit_transform(bertFeats)
    }
    with open(tformFile, 'wb') as ofile:
      pkl.dump(dataMap, ofile)

  for combo in ('speaker', 'sarcasm'), ('sarcasm', 'speaker'):
    for tform in dataMap:
      tfData = dataMap[tform]
      grid = makeDataPlots(tfData, sarcasmDf, *combo, tform)
      if show:
        grid.show()
      title = grid.windowTitle()
      saveGrid(grid, imgDir/f'{title}.jpg')

def getSarcasmDf(optimalSpeakers=True, noContext=True, returnBert=True):
  if returnBert:
    df = pd.read_csv('./data/bert_plus_sarcasm.csv')
  else:
    df = pd.read_csv('./data/sarcasm_data.csv')
  membership = np.ones(len(df), dtype=bool)
  if optimalSpeakers:
    speakers = df['speaker']
    sarcasmRatio = makeSarcasmRatioPlot(df, False)
    badSpeakers = sarcasmRatio[(sarcasmRatio < 0.001) | (sarcasmRatio > 0.999)].index
    membership &= (~np.isin(speakers, badSpeakers))
    membership &= ~(speakers.str.contains('PERSON'))
  if noContext:
    membership &= (~df['context'])

  df = sarcasmDf = df[membership]
  if returnBert:
    sarcasmDf = df.drop('bert', axis=1)
    bertFeats = np.row_stack(df['bert'].apply(literal_eval))
    return sarcasmDf, bertFeats
  return sarcasmDf

def main():
  from ngram_heatmap import makeAllHeatmapPlots
  REGEN_BERT = True
  USE_CONTEXT = True
  print('Getting sarcasm data...')
  sarcasmDf = getSarcasmDf(noContext=USE_CONTEXT, returnBert=REGEN_BERT)
  if REGEN_BERT:
    sarcasmDf, bertFeats = sarcasmDf
  else:
    bertFeats = None
  print('Making ratio plot...')
  makeSarcasmRatioPlot(sarcasmDf)
  print('Making speaker feature plots...')
  makeSpeakerGridPlots(sarcasmDf, bertFeats)
  print('Making heatmap plots...')
  makeAllHeatmapPlots(sarcasmDf)
  print('Done!')

if __name__ == '__main__':
  main()