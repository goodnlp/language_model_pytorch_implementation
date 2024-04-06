"""
transformer implementation in pytorch
"""



from google.colab import drive
drive.mount('/content/drive')

"""# translation dataset
https://www.kaggle.com/datasets/concyclics/machine-translation-chinese-and-english/data
"""



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


#res= []
#for i in range(len(data_EN)):
#  if len(data_EN[i]) > 10 and len(data_ZH[i]) > 10:
 #   res.append(i)

#data_EN = data_EN[res].reset_index(drop=True)
#data_ZH = data_ZH[res].reset_index(drop=True)

data_EN





"""## data preprocessing"""



from transformers import BertTokenizer, BertModel
import numpy as np

class DataPreprocess(object):
  def __init__(self,path=None, language="english"):
    self.path=path
    if language=="english":
      self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    elif language=="chinese":
      self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    ##

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



"""# transformer component"""

from torch import Tensor
import torch.nn.functional as f

import torch
from torch import nn

class AttentionHead(nn.Module):
    def __init__(self, dim_in: int, dim_q: int, dim_k: int):
        super().__init__()
        self.q = nn.Linear(dim_in, dim_q)
        self.k = nn.Linear(dim_in, dim_k)
        self.v = nn.Linear(dim_in, dim_k)
        #self.embedding=nn.Embedding() ##

    def forward(self, query: Tensor, key: Tensor, value: Tensor, mask: Tensor) -> Tensor:  ## 传入mask

        query= self.q(query)
        key= self.k(key)
        value=self.v(value)

        temp = query.bmm(key.transpose(1, 2)) ##
        scale = query.size(-1) ** 0.5
        score=temp/scale
        if mask is not None:
            #print(query.shape)
            #print(key.shape)
            #print("mask shape", mask.shape)
            score=score+mask
        else:
            score=temp/scale

        softmax = f.softmax(score, dim=-1)
        return softmax.bmm(value)

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads: int, dim_in: int, dim_k: int, dim_v: int):
        super().__init__()
        self.heads = nn.ModuleList(
            [AttentionHead(dim_in, dim_k, dim_v) for _ in range(num_heads)]
        )
        self.linear = nn.Linear(num_heads * dim_v, dim_in)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, mask) -> Tensor: ## 传入mask
        return self.linear(
            torch.cat([h(query, key, value,mask) for h in self.heads], dim=-1)
        )

def position_encoding(
    seq_len: int, dim_model: int, device: torch.device = torch.device("cpu"),
) -> Tensor:
    pos = torch.arange(seq_len, dtype=torch.float, device=device).reshape(1, -1, 1)
    dim = torch.arange(dim_model, dtype=torch.float, device=device).reshape(1, 1, -1)
    phase = pos / 1e4 ** (dim // dim_model)

    return torch.where(dim.long() % 2 == 0, torch.sin(phase), torch.cos(phase))

def feed_forward(dim_input: int = 512, dim_feedforward: int = 2048) -> nn.Module:
    return nn.Sequential(
        nn.Linear(dim_input, dim_feedforward),
        nn.ReLU(),
        nn.Linear(dim_feedforward, dim_input),
    )

class AttentionResidual(nn.Module):
    def __init__(self, sublayer: nn.Module, dimension: int, dropout: float = 0.1): ##不是在这里传入mask
        super().__init__()
        self.sublayer = sublayer
        self.norm = nn.LayerNorm(dimension)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tensors0: Tensor, tensors1: Tensor, tensors2: Tensor, mask: Tensor) -> Tensor:
        # Assume that the "value" tensor is given last, so we can compute the
        # residual.  This matches the signature of 'MultiHeadAttention'.
        # self.mask=mask
        return self.norm(tensors0 + self.dropout(self.sublayer(tensors0, tensors1, tensors2,mask))) ## 传入mask

## feed fowward network,Residual需要传入mask，这里不用，所以要分别开
class FeedForwardNetwork(nn.Module):
    def __init__(self, sublayer: nn.Module, dimension: int, dropout: float = 0.1): ##不传入mask
        super().__init__()
        self.sublayer = sublayer
        self.norm = nn.LayerNorm(dimension)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tensors: Tensor) -> Tensor:
        # Assume that the "value" tensor is given last, so we can compute the
        # residual.  This matches the signature of 'MultiHeadAttention'.
        #self.mask=mask ##
        return self.norm(tensors + self.dropout(self.sublayer(tensors)))

class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        dim_model: int = 512,
        num_heads: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        dim_k = dim_v = dim_model // num_heads
        self.attention = AttentionResidual(
            MultiHeadAttention(num_heads, dim_model, dim_k, dim_v), ## 传入mask
            dimension=dim_model,
            dropout=dropout,
        )
        self.feed_forward = FeedForwardNetwork(
            feed_forward(dim_model, dim_feedforward),
            dimension=dim_model,
            dropout=dropout,
        )

    def forward(self, src: Tensor,mask: Tensor) -> Tensor: ##传入mask
        # change to accept multiple input
        src = self.attention(src, src, src,mask) ###传入mask
        #return src
        return self.feed_forward(src)

class TransformerEncoder(nn.Module):
    def __init__(
        self,
        num_layers: int = 6,
        dim_model: int = 512,
        num_heads: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        device: str = 'cpu'
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(dim_model, num_heads, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        self.device= device

    def forward(self, src: Tensor, mask: Tensor) -> Tensor: ##可以传入mask
        seq_len, dimension = src.size(1), src.size(2)
        pos=position_encoding(seq_len,dimension, torch.device(self.device) ) #
        # pos=pos.cuda(0) # load data to gpu
        src += pos
        for layer in self.layers:
            src = layer(src,mask)  ##传入mask

        return src

class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        dim_model: int = 512,
        num_heads: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        dim_q = dim_k = max(dim_model // num_heads, 1)
        # self attention
        self.attention_1 = AttentionResidual(
            MultiHeadAttention(num_heads, dim_model, dim_q, dim_k),
            dimension=dim_model,
            dropout=dropout,
        )
        # cross attention
        self.attention_2 = AttentionResidual(
            MultiHeadAttention(num_heads, dim_model, dim_q, dim_k),
            dimension=dim_model,
            dropout=dropout,
        )
        # FFN
        self.feed_forward = FeedForwardNetwork(
            feed_forward(dim_model, dim_feedforward),
            dimension=dim_model,
            dropout=dropout,
        )

    def forward(self, tgt: Tensor, memory: Tensor, mask: Tensor) -> Tensor:
        tgt = self.attention_1(tgt, tgt, tgt, mask= mask) # for decoder, self attention need mask
        tgt = self.attention_2(tgt, memory, memory, mask=None) # for decoder, cross attention no need mask
        return self.feed_forward(tgt)



class TransformerDecoder(nn.Module):
    def __init__(
        self,
        num_layers: int = 6,
        dim_model: int = 512,
        num_heads: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        device: str = 'cpu'
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(dim_model, num_heads, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        self.device= device

    def forward(self, tgt: Tensor, enc_out: Tensor ,  mask: Tensor) -> Tensor: ##可以传入mask
        # receive output from encoder and do cross-attention,
        seq_len, dimension = tgt.size(1), tgt.size(2)
        pos=position_encoding(seq_len,dimension, torch.device(self.device) ) #
        #pos=pos.cuda(0) # load data to gpu
        tgt += pos
        for layer in self.layers:
            tgt = layer(tgt, enc_out, mask)  ##传入mask

        return tgt





# start a trial of gpt model testing
class Transformer(nn.Module):
    def __init__(
        self,
        num_encoder_layers: int = 4,
        dim_model: int = 512,
        num_heads: int = 8,
        dim_feedforward: int = 2048//2,
        dropout: float = 0.1,
        activation: nn.Module = nn.ReLU(),
        n_vocab_src: int=4,
        n_vocab_tgt: int=4,
        device: str = 'cpu'
    ):
        super().__init__()

        self.embedding_src = nn.Embedding(n_vocab_src, dim_model)
        self.embedding_tgt = nn.Embedding(n_vocab_tgt, dim_model)

        self.encoder = TransformerEncoder(
            num_layers=num_encoder_layers,
            dim_model=dim_model,
            num_heads=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            device= device
        )

        self.decoder = TransformerDecoder(
            num_layers=num_encoder_layers,
            dim_model=dim_model,
            num_heads=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            device= device)

        self.out = nn.Linear(dim_model, n_vocab_tgt)



    def forward(self, src: Tensor, tgt: Tensor, dec_mask: Tensor) -> Tensor: ##传入mask
        src_emb =self.embedding_src(src)
        tgt_emb =self.embedding_tgt(tgt)
        enc_out =self.encoder(src_emb,mask=None) ##传入mask
        dec_out= self.decoder(tgt_emb, enc_out, dec_mask)
        out = self.out(dec_out)

        return out ##no need softmax, nn.cross_entropy take care of it



"""# train model"""

# Generates a square matrix where the each row allows one word more to be seen
def generate_masks(src):
    seq_len= src.size(1)

    pad_int= [int(seq_len-i) for i in src.count_nonzero(dim=1)]

    mask = torch.tril(torch.ones(seq_len, seq_len) == 1) # Lower triangular matrix
    mask = mask.float()
    mask = mask.masked_fill(mask == 0, -1e9) # Convert zeros to -1e9
    mask = mask.masked_fill(mask == 1, float(0.0)) # Convert ones to 0

    mask_arr=[]
    for i in pad_int:
      mask[:,-(i):]= -1e9
      mask_arr.append(mask)

    masks=torch.cat(tuple(mask_arr),dim=0)
    masks=masks.reshape(src.size(0),seq_len,seq_len)

    return masks

import time
######
def loss_masked(output, src, loss_fn):
    #print(src.shape)
    #print(output.shape)
    nonpad_int=src.count_nonzero(dim=1) #number of actual tokens
    # discard pad elements
    loss_res = 0
    for k,item in enumerate(nonpad_int):
        #res.append(src[k][:int(item)])
        loss_res+=loss_fn(output[k][:int(item), :], src[k][:int(item)])

    return loss_res/src.size(0)
######

def train(model,src, tgt, batch_size):
    torch.manual_seed(0)
    model.to(device)

    # define loss function (criterion) and optimizer
    loss_fn = nn.CrossEntropyLoss().to(device)  ## this is for classification
    opt = torch.optim.SGD(model.parameters(), 1e-4)


    n=len(src)// batch_size

    # prepare for next word prediction
    t0=time.time()
    for i in range(n):
        input_src= src[batch_size*i:batch_size*(i+1)]
        input_src=torch.tensor(input_src).long()
        input_tgt= tgt[batch_size*i:batch_size*(i+1)]
        input_tgt=torch.tensor(input_tgt).long()

        _seq= input_tgt[:,:-1]
        seq_= input_tgt[:,1:]
        #enc_masks=generate_masks(src)
        dec_masks=generate_masks(_seq)
        #put to gpu
        input_src= input_src.to(device)
        _seq=_seq.to(device)
        seq_=seq_.to(device)
        #enc_masks= enc_masks.to(device)
        dec_masks= dec_masks.to(device)

        # Forward pass
        outputs = model(input_src, _seq, dec_masks)
        # the part of padding loss should be removed before backprop
        loss=loss_masked(outputs,seq_, loss_fn) # next word prediction
        ## update weight
        opt.zero_grad()
        loss.backward()
        opt.step()

        print(loss.detach().item())
    print(time.time()-t0)

        #if i%10000==0: np.savetxt('./torch_save_model/gpt_loss_%d.csv'%(i), np.array([loss.detach().item()]))

    # save model parameters after finish training model
    torch.save(model, "./model.pkl")



# the entry to start training the model, with data and specified parameter
if __name__ == '__main__':
    # load cpu or gpu name
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # model parameter
    MODEL_DIM = 256
    N_LAYER = 4
    N_HEAD = 8
    # training argument
    batch_size = 128
    # load data
    #ut = utils(path='/content/drive/MyDrive/goodnlp_github_data/language_model_implementation/wiki_zh_10MB.txt')
    #d, v2i=ut.get_idx_sentence()
    src, tgt= sent_idx_en, sent_idx_zh
    n_vocab_en= len(v2i_en) # encoder and decoder both need one vocab
    n_vocab_zh= len(v2i_zh)
    # build model
    m= Transformer(num_encoder_layers= N_LAYER,dim_model= MODEL_DIM, num_heads= N_HEAD, n_vocab_src=n_vocab_en, n_vocab_tgt=n_vocab_zh, device=device)
    # start training
    train(m, src, tgt, batch_size)





