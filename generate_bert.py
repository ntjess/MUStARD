from pathlib import Path
from typing import Union, List

import numpy as np
import pandas as pd
from bert_embedding import BertEmbedding
import json

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from tqdm import tqdm

def parseSarcasmJson(filename: Union[str, Path], save=True):
  filename = Path(filename)
  with open(filename, 'r') as ifile:
    data = json.load(ifile)

  outData = []
  for sample in data.values():
    transformed = {
      'text': sample['utterance'],
      'speaker': sample['speaker'],
      'context': False,
      'sarcasm': sample['sarcasm'],
      'show': sample['show']
    }
    outData.append(transformed)
    for line, speaker in zip(sample['context'], sample['context_speakers']):
      transformed = {
        'text': line,
        'speaker': speaker,
        'context': True,
        'sarcasm': sample['sarcasm'],
        'show': sample['show']
      }
      outData.append(transformed)
  df = pd.DataFrame(outData)
  if save:
    df.to_csv(filename.with_suffix('.csv'), index=False)
  return df

def getBertEmbeddings(text: List[str], firstWordOnly=True, saveName: Union[str, Path]=None):
  bert_embedding = BertEmbedding()
  result = bert_embedding(text)
  indices = [r[0] for r in result]
  data = [r[1] for r in result]
  if firstWordOnly:
    outIdxs = []
    outData = []
    for ii, idx in enumerate(indices):
      for jj, word in enumerate(idx):
        if word.isalnum():
          outIdxs.append(word)
          outData.append(data[ii][jj])
          break
    indices = outIdxs
    data = outData
  else:
    indices = np.concatenate(indices)
  data = np.row_stack(data)
  out = pd.DataFrame(index=indices, data=data)
  if saveName:
    out.to_csv(saveName)
  return out

if __name__ == '__main__':
  sample1Df = parseSarcasmJson('./data/sarcasm_data.json')
  text = sample1Df['text'].tolist()
  berts = getBertEmbeddings(text, saveName='./data/sarcasm_bert.csv')
  # berts = pd.read_csv('./data/sarcasm_bert.csv', index_col=0)
  # lda = LinearDiscriminantAnalysis()
  # ldaTransformed = lda.fit_transform(berts)

