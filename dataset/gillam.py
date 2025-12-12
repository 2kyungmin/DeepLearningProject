#!/usr/bin/env python
# coding: utf-8

# In[122]:


import sys
import os
sys.path.append('..')

from pathlib import Path
from common.utils import *
import numpy as np
import matplotlib.pyplot as plt
import import_ipynb
import csv
import pickle

data_dir = os.path.dirname(os.path.abspath(__file__))

key_file = {
    'train': f'{data_dir}/gillam_train.csv',
    'test': f'{data_dir}/gillam_test.csv',
    'dev': f'{data_dir}/gillam_dev.csv'
}

vocab_file = 'gillam.vocab.pkl'

def get_save_file_path(data_type, max_len=None):
    save_path = data_dir+'/'
    if max_len == None:
        save_path += f'{data_type}.npy'
    else:
        save_path += f'{data_type}{max_len}.npy'
    return save_path

# In[123]:


def load_vocab():

    '''
    1. train.csv파일을 읽어 파일명을 가져옴
    2. extract_utterances()으로 "CHI"의 plaintext를 word_to_id, id_to_word로 변환
    3. vocab파일에 저장
    4. 파일 존재시, 불러오기
    Returns: 
        word_to_id, id_to_word
    '''

    if os.path.exists(vocab_file):
        with open(vocab_file, 'rb') as f:
            word_to_id, id_to_word = pickle.load(f)
        return word_to_id, id_to_word

    word_to_id = {}
    id_to_word = {}
    word_to_id["<UNK>"] = 0
    id_to_word[0] = "<UNK>"
    data_type = 'train'
    words = []

    with open(key_file[data_type], 'r') as f:
        dr = csv.DictReader(f)
        for line in dr:
            tmp_path = data_dir+'/'+line['filename']
            utterances = extract_utterances(tmp_path, ["CHI"])
            for utt in utterances[:]:
                words.extend(utt.clean_text.replace('\n', '<eos>').strip().split())
        
    for i, word in enumerate(words):
        if word not in word_to_id:
            tmp_id = len(word_to_id)
            word_to_id[word] = tmp_id
            id_to_word[tmp_id] = word

    with open(vocab_file, 'wb') as f:
        pickle.dump((word_to_id, id_to_word), f)

    return word_to_id, id_to_word 


# In[124]:
def load_data(data_type):
    '''
    Args: 
        data_type: 'train' or 'test' or 'dev'
    Returns: 
        corpus, word_to_id, id_to_word
    
    '''
    if data_type not in ['train', 'test', 'dev']:
        raise ValueError(f'data_type: {data_type}')

    save_path = get_save_file_path(data_type)

    word_to_id, id_to_word = load_vocab()
    if os.path.exists(save_path):
        corpus = np.load(save_path, allow_pickle=True)
        return corpus, word_to_id, id_to_word

    # CSV파일을 읽어 data file명 찾기
    with open(key_file[data_type], 'r') as f:
        tmp = []
        dr = csv.DictReader(f)
        for line in dr:
            words = []
            tmp_path = data_dir+'/'+line['filename']   # 원본 데이터의 파일 위치
            utterances = extract_utterances(tmp_path, ["CHI"])    # 아이의 발화만 정리
            for utt in utterances[:]:
                words.extend(utt.clean_text.replace('\n', '<eos>').strip().split())
            # word_to_id에 없는 단어는 <UNK>
            unk_id = word_to_id["<UNK>"]
            tmp_words = [word_to_id.get(w, unk_id) for w in words]
            # SLI 아동이면 corpus 마지막에 '0', TD 아동은 '1' 추가
            tmp_words.extend('0' if line['group'] == 'SLI' else '1') 
            tmp.append(tmp_words)

    corpus = np.array(tmp, dtype=object)
    np.save(save_path, corpus)
    return corpus, word_to_id, id_to_word    

# %%
def load_data_len(data_type, max_len=256):
    '''    
    :param data_type: 'train' 'test', 'dev'
    :param max_len: 너무 긴 아동의 인터뷰는 len길이만큼 자름
        남은 인터뷰가 짧다면 앞의 단어를 가져와 max_len을 맞춤
    :returns: corpus, word_to_id, id_to_word
    '''
    if data_type not in ['train', 'test', 'dev']:
        raise ValueError(f'data_type: {data_type}')

    save_path = get_save_file_path(data_type, max_len)

    word_to_id, id_to_word = load_vocab()
    if os.path.exists(save_path):
        corpus = np.load(save_path, allow_pickle=True)
        return corpus, word_to_id, id_to_word

    # CSV파일을 읽어 data file명 찾기
    with open(key_file[data_type], 'r') as f:
        tmp = []
        dr = csv.DictReader(f)
        for line in dr:
            words = []
            tmp_path = data_dir+'/'+line['filename']   # 원본 데이터의 파일 위치
            utterances = extract_utterances(tmp_path, ["CHI"])    # 아이의 발화만 정리
            for utt in utterances[:]:
                words.extend(utt.clean_text.replace('\n', '<eos>').strip().split())
            # word_to_id에 없는 단어는 <UNK>
            unk_id = word_to_id["<UNK>"]
            word_ids = [word_to_id.get(w, unk_id) for w in words]
            # SLI 아동이면 corpus 마지막에 '0', TD 아동은 '1' 추가
            label = 0 if line['group'] == 'SLI' else 1
            total_len = len(word_ids)
            for start in range(0, total_len, max_len):
                chunk = word_ids[start:start + max_len]
                tmp.append(chunk+[label])
                
            if total_len % max_len!=0:
                last_chunk = word_ids[-max_len:]
                tmp.pop()                           # 마지막 요소 제거
                tmp.append(last_chunk + [label])    # 새로운 chunk 추가
            
    corpus = np.array(tmp, dtype=object)
    np.save(save_path, corpus)
    return corpus, word_to_id, id_to_word    

# %%
# corpus, _, _ = load_data_len("train", 32)
# print(corpus[0])
# print(corpus[2])
# print(max([len(x) for x in corpus]))
# print(os.path.dirname(os.path.abspath(__file__)))
# print(get_save_file_path('train', 256))

# In[125]:

if __name__ == '__main__':
    for data_type in ('train', 'dev', 'test'):
        load_data(data_type)

