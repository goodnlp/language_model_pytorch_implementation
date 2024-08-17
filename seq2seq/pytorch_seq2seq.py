# -*- coding: utf-8 -*-
"""pytorch_seq2seq.ipynb
# reference

## reference to lstm : https://colah.github.io/posts/2015-08-Understanding-LSTMs/

## reference to seq2seq: https://github.com/bentrevett/pytorch-seq2seq
"""



"""# data"""

import pandas as pd

def load_data(path):
    with open(path, 'r', encoding='utf-8') as f:
        # replace '\n' with ' '
        data = pd.Series(f.readlines())
        data = data.str.replace('\n', ' ')
    return data

data_ZH = load_data('/content/drive/MyDrive/goodnlp_github_data/language_model_implementation/english_chinese_translation_dataset/chinese.zh')
data_EN = load_data('/content/drive/MyDrive/goodnlp_github_data/language_model_implementation/english_chinese_translation_dataset/english.en')

data_EN = data_EN[:10000]
data_ZH = data_ZH[:10000]

from transformers import BertTokenizer, BertModel
import numpy as np

class DataPreprocess(object):
  def __init__(self,path=None, language="english"):
    self.path=path
    if language=="english":
      self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    elif language=="chinese":
      self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

  def preprocess_wiki(self, res):
    vocab=set()
    length=[]
    sentence_arr=[]
    for i in range(len(res)):
      text=res[i]
      tokenized_text = self.tokenizer.tokenize(text)
      # remove sentence which is too short, too long
      #if 5<= len(tokenized_text)<=100:
      sentence_arr.append(tokenized_text)
      length.append(len(tokenized_text))
      vocab.update(tokenized_text)

    v2i={v: i for i, v in enumerate(sorted(vocab), start=1)}
    v2i['<PAD>']=0
    v2i["<EOS>"] = len(v2i) # <BOS> as start of sequence ,<EOS> as end of sequence
    v2i["<BOS>"] = len(v2i) # the total number of tokens should include these special tokens: len(v2i)

    i2v = {i: v for v, i in v2i.items()}
    return sentence_arr, v2i, i2v, max(length)

  def token_to_idx(self,sentence_arr, v2i):
    sentence_idx=[]
    for i in range(len(sentence_arr)):
      sentence_idx.append([v2i['<BOS>']]+[v2i[item] for item in sentence_arr[i]]+[v2i['<EOS>']])
    return sentence_idx

  # add a pad_zero function to align the sentences of various length
  def pad_zero(self, seqs, max_len):
      PAD_ID = 0
      padded = np.full((len(seqs), max_len), fill_value=PAD_ID, dtype=np.int32)
      for i, seq in enumerate(seqs):
          padded[i, :len(seq)] = seq
      return padded

  def get_idx_sentence(self):
    sentence_arr, v2i, i2v, max_len= self.preprocess_wiki(self.path) #input is part of wiki data, for demo usage
    sentence_idx = self.token_to_idx(sentence_arr, v2i)
    # define idx for padding
    PAD_ID= v2i['<PAD>']
    # there is <GO> and <SEP> at start and ending of sentence, so the full length should be 100+2=102
    sentence_idx_padded = self.pad_zero(sentence_idx,max_len+2)
    return sentence_idx_padded.tolist(), v2i

ut_en = DataPreprocess(data_EN, language="english")
sent_idx_en, v2i_en = ut_en.get_idx_sentence()

ut_zh = DataPreprocess(data_ZH, language="chinese")
sent_idx_zh, v2i_zh = ut_zh.get_idx_sentence()







"""# model, lstm using pytorch"""

import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import spacy
import torchtext
import tqdm

class Encoder(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, n_layers, dropout):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # src = [src length, batch size]
        src = src.permute(1, 0)
        embedded = self.dropout(self.embedding(src))
        # embedded = [src length, batch size, embedding dim]
        outputs, (hidden, cell) = self.rnn(embedded)
        # outputs are always from the top hidden layer
        # print("hidden and cell shape from encoder is: ", hidden.shape, cell.shape)
        return outputs, hidden, cell

class Decoder(nn.Module):
    def __init__(self, output_dim, embedding_dim, hidden_dim, n_layers, dropout):
        super().__init__()
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(output_dim, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=dropout)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell):
        input = input.unsqueeze(0)
        # input = [1, batch size]
        embedded = self.dropout(self.embedding(input))
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        prediction = self.fc_out(output.squeeze(0))
        return prediction, hidden, cell





class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder # initialize here
        self.decoder = decoder # initialize here
        self.device = device
        assert (
            encoder.hidden_dim == decoder.hidden_dim
        ), "Hidden dimensions of encoder and decoder must be equal!"
        assert (
            encoder.n_layers == decoder.n_layers
        ), "Encoder and decoder must have equal number of layers!"

    def forward(self, src, trg, teacher_forcing_ratio):
        # src = [src length, batch size]
        # trg = [trg length, batch size]
        batch_size = trg.shape[0]
        trg_length = trg.shape[1] ## trg length should be in the 2nd position, before permute operation
        trg_vocab_size = self.decoder.output_dim
        # tensor to store decoder outputs
        outputs = torch.zeros(trg_length, batch_size, trg_vocab_size).to(self.device)
        # last hidden state of the encoder is used as the initial hidden state of the decoder
        encoder_outputs,hidden, cell = self.encoder(src)
        trg = trg.permute(1, 0)
        input = trg[0, :] #
        # input = [batch size]
        for t in range(1, trg_length):
            output, hidden, cell = self.decoder(input, hidden, cell)
            outputs[t,:,:] = output
            # decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio
            # get the highest predicted token from our predictions
            top1 = output.argmax(1)
            input = trg[t] if teacher_force else top1
            # input = [batch size]
        return outputs





def train_fn(model, data_en, data_zh, teacher_forcing_ratio, device, batch_size):
    # optimizer
    optimizer = optim.Adam(model.parameters())
    # loss function
    pad_index = 0
    criterion = nn.CrossEntropyLoss(ignore_index=pad_index)

    model.train()
    epoch_loss = 0
    # print(len(data_en))
    n= len(data_en)//batch_size
    for i in range(n):
        # print(i)
        src = data_en[i*batch_size:(i+1)*batch_size]
        src = torch.tensor(src).long().to(device)
        trg = data_zh[i*batch_size:(i+1)*batch_size]
        trg = torch.tensor(trg).long().to(device)
        optimizer.zero_grad()
        output = model(src, trg, teacher_forcing_ratio)
        #print("output from model shape is: ", output.shape)
        # output = [trg length, batch size, trg vocab size]
        output = output.permute(1, 2, 0)
        output = output[:,:,:-1]
        trg= trg.permute(0,1)
        trg= trg[:,1:]
        loss = criterion(output, trg)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        res = epoch_loss
    print(epoch_loss)



if __name__ == "__main__":
    teacher_forcing_ratio = 0.5
    device= torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model parameters
    input_dim = len(v2i_en)
    output_dim = len(v2i_zh)

    encoder_embedding_dim = 256
    decoder_embedding_dim = 256
    hidden_dim = 512
    n_layers = 1 # number of layer for lstm model
    encoder_dropout = 0.5
    decoder_dropout = 0.5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder = Encoder(
        input_dim,
        encoder_embedding_dim,
        hidden_dim,
        n_layers,
        encoder_dropout)


    decoder = Decoder(
        output_dim,
        decoder_embedding_dim,
        hidden_dim,
        n_layers,
        decoder_dropout)

    model = Seq2Seq(encoder, decoder, device).to(device)
    batch_size = 100

    train_fn(model, sent_idx_en, sent_idx_zh, teacher_forcing_ratio, device, batch_size)



