"""
gpt implementation in pytorch
"""

from transformers import BertTokenizer, BertModel
import numpy as np

class utils(object):
  def __init__(self,path=None):
    self.path=path
    self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

  def preprocess_wiki(self, path):
    f = open(self.path)
    res = f.read().splitlines()
    f.close()

    vocab=set()
    length=[]
    sentence_arr=[]
    for i in range(len(res)):
      text=res[i]
      tokenized_text = self.tokenizer.tokenize(text)
      # remove sentence which is too short, too long
      if 5<= len(tokenized_text)<=100:
        sentence_arr.append(tokenized_text)
        length.append(len(tokenized_text))
        vocab.update(tokenized_text)

    v2i={v: i for i, v in enumerate(sorted(vocab), start=1)}
    v2i['<PAD>']=0
    v2i["<SEP>"] = len(v2i) # <GO> as start of sequence ,<SEP> as end of sequence
    v2i["<GO>"] = len(v2i) # the total number of tokens should include these special tokens: len(v2i)

    i2v = {i: v for v, i in v2i.items()}
    return sentence_arr, v2i, i2v, max(length)

  def token_to_idx(self,sentence_arr, v2i):
    sentence_idx=[]
    for i in range(len(sentence_arr)):
      sentence_idx.append([v2i['<GO>']]+[v2i[item] for item in sentence_arr[i]]+[v2i['<SEP>']])
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

"""# GPT component"""

from torch import Tensor
import torch.nn.functional as f
import torch
from torch import nn

from pytorch_transformer import TransformerEncoder

# start a trial of gpt model testing
class GPT(nn.Module):
    def __init__(
        self,
        num_encoder_layers: int = 4,
        dim_model: int = 512,
        num_heads: int = 8,
        dim_feedforward: int = 2048//2,
        dropout: float = 0.1,
        activation: nn.Module = nn.ReLU(),
        n_vocab: int=4,
        device: str = 'cpu'
    ):
        super().__init__()

        self.embedding = nn.Embedding(n_vocab, dim_model)

        self.encoder = TransformerEncoder(
            num_layers=num_encoder_layers,
            dim_model=dim_model,
            num_heads=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            device= device
        )

        self.out = nn.Linear(dim_model, n_vocab)



    def forward(self, src: Tensor, mask: Tensor) -> Tensor: ##传入mask
        emb=self.embedding(src)
        enc=self.encoder(emb,mask) ##传入mask
        out=self.out(enc)

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
    nonpad_int=src.count_nonzero(dim=1) #number of actual tokens
    # discard pad elements
    loss_res = 0
    for k,item in enumerate(nonpad_int):
        loss_res+=loss_fn(output[k][:int(item), :], src[k][:int(item)])
    return loss_res/src.size(0)

def train(model, data, batch_size):
    torch.manual_seed(0)
    model.to(device)

    # define loss function (criterion) and optimizer
    loss_fn = nn.CrossEntropyLoss().to(device)  ## this is for classification
    opt = torch.optim.SGD(model.parameters(), 1e-4)


    n=len(data)// batch_size

    # prepare for next word prediction
    t0=time.time()

    for i in range(n):
        src=data[batch_size*i:batch_size*(i+1)]
        src=torch.tensor(src).long()

        _seq=src[:,:-1]
        seq_=src[:,1:]
        masks=generate_masks(_seq)


        #put to gpu
        _seq=_seq.to(device)
        seq_=seq_.to(device)
        masks=masks.to(device)

        # Forward pass
        outputs = model(_seq,masks)
        loss=loss_masked(outputs,seq_, loss_fn) # next word prediction

        # the part of padding loss should be removed before backprop
        opt.zero_grad()
        loss.backward()
        opt.step()

        print(loss.detach().item())
    print(time.time()-t0)

        #if i%10000==0: np.savetxt('./torch_save_model/gpt_loss_%d.csv'%(i), np.array([loss.detach().item()]))

    # save model parameters after finish training model
    torch.save(model, "./model.pkl")



# the entry to start training the model, with data, with specified parameter
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
    ut = utils(path='/content/drive/MyDrive/goodnlp_github_data/language_model_implementation/wiki_zh_10MB.txt')
    d, v2i=ut.get_idx_sentence()
    n_vocab= len(v2i)
    m= GPT(num_encoder_layers= N_LAYER,dim_model= MODEL_DIM, num_heads= N_HEAD, n_vocab=n_vocab, device=device)
    # start training
    train(m, d, batch_size)

