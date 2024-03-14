# 实现Pytorch版本的GPT模型

## 为什么要自己从头实现GPT?

通过逐步实现GPT模型，了解基于transformer的语言模型是如何工作的。同时，了解预训练语料是如何准备的，也是很有必要的。有了模型和数据，在家用电脑就可以上手训练当今最前沿的语言模型。在此基础上，可以对模型提出修改甚至创新，推动自然语言处理发展。

## 准备数据

使用wikipedia 文本数据 (file00包含wikipedia前80000个句子，已经清洗干净)


## 模型代码解读

模型代码在同目录的pytorch-gpt文件夹中，包含两个jupyter notebooks。其中pytorch_gpt_cpu_single_gpu.ipynb是运行于cpu或者单个gpu的计算环境中的；pytorch_gpt_multigpu.ipynb是运行于多个gpu计算环境中的。可以在terminal用nvidia-smi查看计算环境有多少块gpu。

下面对pytorch_gpt_cpu_single_gpu.ipynb的代码进行解释。

```python
from transformers import BertTokenizer, BertModel
import numpy as np

class utils(object):
  def __init__(self,path=None):
    self.path=path
    self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

  def preprocess_wiki(self, path):
    ...

  def token_to_idx(self,sentence_arr, v2i):
    ...

  # add a pad_zero function to align the sentences of various length
  def pad_zero(self, seqs, max_len):
    ...

  def get_idx_sentence(self):
    sentence_arr, v2i, i2v, max_len= self.preprocess_wiki(self.path) #input is part of wiki data, for demo usage
    sentence_idx = self.token_to_idx(sentence_arr, v2i)
    # define idx for padding
    PAD_ID= v2i['<PAD>']
    # there is <GO> and <SEP> at start and ending of sentence, so the full length should be 100+2=102
    sentence_idx_padded = self.pad_zero(sentence_idx,max_len+2)

    return sentence_idx_padded.tolist(), v2i
```
utils是一个数据预处理的class，输入文本文件的路径，文件中包含80000个英文维基百科的句子。处理步骤如下：

* 第一步，使用wordpiece 把每一个句子都化为单个的token
* 第二步，把每一个句子中的token逐一对应为数字索引
* 第三步，转化为数字索引后，需要对有些句子的索引补零，使得所有的句子索引都是一样长度的，方便后面输入GPT模型进行批量操作。

<br />
<br />

```python

class AttentionHead(nn.Module):
    def __init__(self, dim_in: int, dim_k: int, dim_v: int):
        super().__init__()
        self.q = nn.Linear(dim_in, dim_k)
        self.k = nn.Linear(dim_in, dim_k)
        self.v = nn.Linear(dim_in, dim_v)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, mask: Tensor) -> Tensor:  ## 传入mask 
        query= self.q(query)
        key= self.k(key)
        value=self.v(value)

        temp = query.bmm(key.transpose(1, 2))
        scale = query.size(-1) ** 0.5

        score=temp/scale
        score=score+mask

        softmax = f.softmax(score, dim=-1)
        return softmax.bmm(value)
 ```
Attentionhead 定义了标准注意力机制的操作过程，Attention(q,k,v) = softmax( ${q \times k^T}\over{\sqrt d_k}$ ) * v


<br />
<br />


```python
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

 ```

这里实现了包含多个注意力头的类， 是为了在embedding的多个不同的子空间分别做注意力操作，然后计算结果再简单地合并起来，合并起来的结果再输入一个全连接层。

<br />
<br />


```python

def position_encoding(
    seq_len: int, dim_model: int, device: torch.device = torch.device("cpu"),
) -> Tensor:
    pos = torch.arange(seq_len, dtype=torch.float, device=device).reshape(1, -1, 1)
    dim = torch.arange(dim_model, dtype=torch.float, device=device).reshape(1, 1, -1)
    phase = pos / 1e4 ** (dim // dim_model)

    return torch.where(dim.long() % 2 == 0, torch.sin(phase), torch.cos(phase))

```
因为transformer架构中，每一个词都是并行输入的，这样的话，就丢失了单词在句子中的位置信息，所以需要人为的加上位置信息，这里使用的是正弦余弦信号，是一种相对位置信息。


<br />
<br />


```python
def feed_forward(dim_input: int = 512, dim_feedforward: int = 2048) -> nn.Module:
    return nn.Sequential(
        nn.Linear(dim_input, dim_feedforward),
        nn.ReLU(),
        nn.Linear(dim_feedforward, dim_input),
    )
```

因为transformer架构中，attention也是有很多层的，层与层之间也需要一些连接层，也就是这杨的操作: linear + relu + linear 。


<br />
<br />


```python
class Residual(nn.Module):
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
```


这里为了实现在attention层的残差网络，当sublayer是attention机制的时候，我们可以看到，输入向量先经历了自注意力操作，然后dropout操作，再然后是与残差加起来，最后还有一个layernorm的操作。
我们可以看到，transformer是集前人的成果而大成的，每一个小的研究成果在最后都起了作用，这就是机器学习领域不断积累却在进步的原因。

<br />
<br />


```python
class Residual_ffn(nn.Module):
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
```


这里实现在上文提到的连接层的残差网络。

<br />
<br />


```
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
        self.attention = Residual(
            MultiHeadAttention(num_heads, dim_model, dim_k, dim_v), ## 传入mask
            dimension=dim_model,
            dropout=dropout, 
        )
        self.feed_forward = Residual_ffn(
            feed_forward(dim_model, dim_feedforward),
            dimension=dim_model,
            dropout=dropout,
        )

    def forward(self, src: Tensor,mask: Tensor) -> Tensor: ##传入mask
        src = self.attention(src, src, src,mask) ###传入mask
        #return src 
        return self.feed_forward(src)
```

TransformerEncoderLayer这个类，就把前面实现的模块就包含组装起来了。其包含了attention层，和连接层。

<br />
<br />


```
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
        #pos=pos.cuda(0) # load data to gpu
        src += pos
        for layer in self.layers:
            src = layer(src,mask)  ##传入mask

        return src
```
TransformerEncoder这个类，就把TransformerEncoderLayer这个类使用了好多次。


```
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
```
最终，主角登场，最初始化函数中就可以看出来，GPT总体来看，就包含了3个东西，embedding, encoder, out。其中embedding是嵌入层；encoder是主要部分，其完成自注意力操作；out是输出层，把经过计算的向量转换为一个具有n_vocab维的向量。




## 参考
* (1) https://towardsdatascience.com/pre-processing-a-wikipedia-dump-for-nlp-model-training-a-write-up-3b9176fdf67
* (2) https://medium.com/the-dl/transformers-from-scratch-in-pytorch-8777e346ca51
