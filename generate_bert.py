from pathlib import Path
from typing import Union, List

import numpy as np
import pandas as pd
import json

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from tqdm import tqdm
import jsonlines

DATA_PATH_JSON = "./data/sarcasm_data.json"
BERT_TARGET_EMBEDDINGS = "./data/bert-output.jsonl"
BERT_CONTEXT_EMBEDDINGS = "./data/bert-output-context.jsonl"


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

def createCustomBertEmbeddings(text: List[str], firstWordOnly=True, saveName: Union[str, Path]=None):
  from bert_embedding import BertEmbedding
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

def getContextBertFeatures(dataset=None):
  if dataset is None:
    dataset = json.load(open(DATA_PATH_JSON))
  # Prepare context video length list
  length=[]
  for idx, ID in enumerate(dataset.keys()):
    length.append(len(dataset[ID]["context"]))

  # Load BERT embeddings
  with jsonlines.open(BERT_CONTEXT_EMBEDDINGS) as reader:
    context_utterance_embeddings=[]
    # Visit each context utterance
    for obj in tqdm(reader):

      CLS_TOKEN_INDEX = 0
      features = obj['features'][CLS_TOKEN_INDEX]

      bert_embedding_target = []
      for layer in [0,1,2,3]:
        bert_embedding_target.append(np.array(features["layers"][layer]["values"]))
      bert_embedding_target = np.mean(bert_embedding_target, axis=0)
      context_utterance_embeddings.append(np.copy(bert_embedding_target))

  # Checking whether total context features == total context sentences
  assert(len(context_utterance_embeddings)== sum(length))

  # Rearrange context features for each target utterance
  # cumulative_length = np.cumsum(length)

  # end_index = cumulative_length.tolist()
  # start_index = [0]+end_index[:-1]

  # final_context_bert_features = []
  # for start, end in zip(start_index, end_index):
  #   local_features = []
  #   for idx in range(start, end):
  #     local_features.append(context_utterance_embeddings[idx])
  #   final_context_bert_features.append(local_features)

  return context_utterance_embeddings

def getTargetBertFeatures():
  text_bert_embeddings = []
  with jsonlines.open(BERT_TARGET_EMBEDDINGS) as reader:
      
    # Visit each target utterance
    for obj in tqdm(reader):

      CLS_TOKEN_INDEX = 0
      features = obj['features'][CLS_TOKEN_INDEX]

      bert_embedding_target = []
      for layer in [0,1,2,3]:
        bert_embedding_target.append(np.array(features["layers"][layer]["values"]))
      bert_embedding_target = np.mean(bert_embedding_target, axis=0)
      text_bert_embeddings.append(np.copy(bert_embedding_target))

  return text_bert_embeddings

def generateDataCsv():
  parseSarcasmJson(DATA_PATH_JSON)

def generateCustomBertCsv():
  sample1Df = pd.read_csv(Path(DATA_PATH_JSON).with_suffix('.csv'))
  text = sample1Df['text'].tolist()
  createCustomBertEmbeddings(text, saveName='./data/sarcasm_bert.csv')

def generateMustardBert():
  tgt = getTargetBertFeatures()
  ctx = getContextBertFeatures()
  df = pd.read_csv(Path(DATA_PATH_JSON).with_suffix('.csv'))
  sortedDf = df.sort_values(by=['context'])
  bertFeats = np.row_stack([tgt, ctx]).tolist()
  sortedDf['bert'] = bertFeats
  sortedDf.to_csv('./data/bert_plus_sarcasm.csv')

generateMustardBert()