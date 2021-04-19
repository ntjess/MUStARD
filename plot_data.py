# from nltk.grammar import cfg_demo
# from data_loader import DataLoader
from config import Config

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn import svm
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pyqtgraph as pg
from utilitys.widgets import HoverScatter
app = pg.mkQApp()

cfg = Config()
BERT_COL = 7
# dl = DataLoader(cfg)
# bertFeats = np.array([data[BERT_COL] for data in dl.data_input])
# np.save('./data/bertFeats.npy', bertFeats)
# bertFeats = np.load('./data/bertFeats.npy')
# bertFeats = np.delete(bertFeats, 308, 1)
bertFeats = pd.read_csv('./data/sarcasm_bert.csv', index_col=0).to_numpy()
sarcasmDf = pd.read_csv('./data/sarcasm_data.csv')
membership = (sarcasmDf['show'] == 'BBT').values
bertFeats = bertFeats[membership]
sarcasmDf = sarcasmDf[membership]
# labels = np.array(dl.data_output)
# np.save('./data/labels.npy', labels)
# labels = np.load('./data/labels.npy')
speakerLabels = sarcasmDf['speaker'].to_numpy()
speakers = np.unique(speakerLabels).tolist()
labels = np.fromiter((speakers.index(l) for l in speakerLabels), dtype=int)

pca = PCA()
pcaTransformed = pca.fit_transform(bertFeats)

lda = LDA()
ldaTransformed = lda.fit_transform(bertFeats, labels)

tsne = TSNE()
tsneTransformed = tsne.fit_transform(bertFeats, labels)

pw = pg.PlotWidget()
spi = HoverScatter(*tsneTransformed[:,:2].T, pen=labels, brush=labels, data=speakerLabels)
pw.addItem(spi)
pw.show()

coefs = np.sort(np.abs(lda.coef_)).flatten()
importances = coefs/coefs.max()
plt.hist(importances, 100)

toPlot = ldaTransformed[:10000]
plt.scatter(*toPlot[:, :2].T)
plt.show()


def makeDataPlots(bertFeats, labels):
  gw = pg.GraphicsLayoutWidget()
  curPlt = gw.nextCol()