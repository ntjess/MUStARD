# from nltk.grammar import cfg_demo
# from data_loader import DataLoader
from pathlib import Path

from sklearn.cluster import FeatureAgglomeration

from config import Config

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn import svm, random_projection
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pyqtgraph as pg
from utilitys.widgets import HoverScatter
app = pg.mkQApp()
from utilitys import PrjParam
from ast import literal_eval
from pyqtgraph.Qt import QtWidgets, QtCore, QtGui
import sweetviz
import pickle as pkl

imgDir = Path('./proj_images')
imgDir.mkdir(exist_ok=True)
cfg = Config()
BERT_COL = 7
# dl = DataLoader(cfg)
# bertFeats = np.array([data[BERT_COL] for data in dl.data_input])
# np.save('./data/bertFeats.npy', bertFeats)
# bertFeats = np.load('./data/bertFeats.npy')
# bertFeats = np.delete(bertFeats, 308, 1)
df = pd.read_csv('./data/bert_plus_sarcasm.csv')
membership = df['speaker'].apply(str.isalpha)
df = df[membership]
bertFeats = np.row_stack(df['bert'].apply(literal_eval))
sarcasmDf = df.drop('bert', axis=1)

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

def makeSarcasmRatioPlot(sarcasmDf):
  speakers = np.unique(sarcasmDf['speaker'])
  ratioData = {}
  for speaker in speakers:
    speakerDf = sarcasmDf.loc[sarcasmDf['speaker'] == speaker, 'sarcasm']
    ratioData[speaker] = speakerDf.sum()/len(speakerDf)
  pd.Series(ratioData).sort_values(ascending=False).plot.bar()
  plt.title('Ratio of sarcastic to non-sarcastic sentences')
  plt.tight_layout()
  plt.savefig(imgDir/'sarcasmRatio.pdf')

def makeSpeakerGridPlots():
  tformFile = './data/transformData.pkl'
  try:
    with open(tformFile, 'rb') as ifile:
      dataMap = pkl.load(ifile)
  except Exception:
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
        grid.show()
        title = grid.windowTitle()
        print(grid.windowTitle())
        # Resize for good aspect ratio before saving
        fullSize = app.primaryScreen().size()
        grid.resize(fullSize)
        pic = grid.grab()
        assert pic.save(str(imgDir/f'{title}.jpg'))

makeSarcasmRatioPlot(sarcasmDf)