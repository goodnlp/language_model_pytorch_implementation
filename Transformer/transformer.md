# 实现Pytorch版本的Transformer模型

## 如何学习Transformer模型?

经常有人问有没有通俗易懂地解释transformer。最高效直接地方法就是阅读源码，了解输入数据是如何预处理的，然后如何在模型中流动和计算处理，如何输出，输出结果如何用来计算损失函数，这也是模型训练过程的从数据视角做的一个宏观概述。我认为从数据视角更符合人类的直觉和习惯，因为数据是直接的可以感知的，模型结构是一种抽象的表示。因此我们可以从数据视角入手，然后再深入了解模型。

## transformer是干什么的？
transformer在原论文中的任务是机器翻译，比如输入一段英文，输出一段中文，因此这个模型是对序列与序列之间关系进行建模学习。

## Transformer模型的宏观结构
整体来讲，Transformer包含encoder和decoder。 下面我们以输入英文（记为src），输出中文(记为tgt)为例进行解释。

scr= "i eat melon"， 
tgt= "我吃瓜"，

src进入encoder，encoder按照数据流动的先后顺序，依次是embedding layer, N层 (attention layer + feedforward layer),注意这里的attention操作就是self attention,  然后得到encoder的输出，我们记为enc_out。假设enc_out为[vector_e1, vector_e2, vector_e3].

到了decoder，结构也是是embedding layer, N层 (attention layer + feedforward layer), 注意这里的attention操作也是self attention。最后一层是一个全连接层用于预测输出的token。但是情况稍微复杂一点，以上的enc_out 和tgt 都要进入decoder。

tgt经过 (attention layer + feedforward layer)处理后，tgt可以表征为[vector_d1, vector_d2, vector_d3]， 当要预测“吃”这个字的时候，vector_d1会和enc_out做attention_score的计算（这就是cross attention），假设attention_score = [0.1, 0.3, 0.6], 那么得到一个向量dec_out = 0.1*vector_e1 + 0.3*vector_e2 + 0.6*vector_e3 + vector_d1， dec_out进入全连接层，然后进行softmax计算得到一个针对当前词库的概率分布，得到概率分布就可以带入损失函数计算得到损失，这样前向传播的过程就完成。反向传播就是将损失依照导数链式法则传播到各个权重层，从而更新每一个权重参数。

以上就完成了一次transformer模型的参数更新过程，也就是模型训练的过程，看是高大上的模型也没有很复杂吧。


## 训练数据

使用一个kaggle上的英中翻译数据集。

地址： https://www.kaggle.com/datasets/concyclics/machine-translation-chinese-and-english/data


## 模型代码解读

模型代码在同目录的transformer文件夹中, 这个实现可以自动检测是cpu环境还是gpu环境，当然优先使用gpu环境进行加速计算。

下面对pytorch_transformer.py文件的代码进行解释, 其中代码块在上面，相应的解释在下面。

```python
import pandas as pd
def load_data(path):
    with open(path, 'r', encoding='utf-8') as f:
        # replace '\n' with ' '
        data = pd.Series(f.readlines())
        data = data.str.replace('\n', ' ')
    return data

data_ZH = load_data('./english_chinese_translation_dataset/chinese.zh')
data_EN = load_data('./english_chinese_translation_dataset/english.en')

data_EN = data_EN[:10000]
data_ZH = data_ZH[:10000]
```
这里是load两个数据文件，取出前10000个句子对。

<br />
<br />



```python
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

```


DataPreprocess是一个数据预处理的class，输入文本文件的路径和对应语言就可以。因为本文以英文翻译中文为例，需要对应的tokenizer,这里直接使用的是bert-base的subword tokenizer, 我们这里之翻译前10000个句子对。其中的处理步骤如下：

* 第一步，使用bert-base的subword tokenizer 把每一个句子都化为单个的token, 对应preprocess_wiki这个函数,
* 第二步，把每一个句子中的token逐一对应为数字索引, 对应token_to_idx这个函数，
* 第三步，转化为数字索引后，需要对有些句子的索引补零，使得所有的句子索引都是一样长度的，方便后面输入transformer模型进行批量操作，对应pad_zero这个函数。

<br />
<br />

```python

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
            score=score+mask
        else:
            score=temp/scale

        softmax = f.softmax(score, dim=-1)
        return softmax.bmm(value)

 ```
Attentionhead 定义了标准注意力机制的操作过程，计算公式是：Attention(q,k,v) = softmax( ${q \times k^T}\over{\sqrt d_k}$ ) * v.
这个类也是整个transformer的核心操作所在。我们以自注意力为例子解释，在EncoderLayer中，我们把src 分别作为query, key, value传入计算。所以到这个AttentionHead的时候，src只是乘了不同的权重矩阵得到相对应的q, k, v，然后接下来的自注意力操作。自注意力计算的过程就是src中的某个token融合整个src中其他token（计算的时候也包含token自己）的信息表征，如何融合呢，就是根据上面计算公式求的attention_score, 其实就是一些权重， 整个src的所有token信息表征按照这个权重加和起来，就是某个token的新的信息表征。所以有的说法是attention是基于图的一个信息传递机制，节点之间的边是学习到的，这也是一个角度的理解。


另外一点，因为有的时候需要对attention  score进行mask，因此需要给这个函数加一个mask的参数，mask不是None则会启用mask功能，下文会有使用到。
maks也是一个矩阵，只是对于需要mask的token，矩阵对应位置设置为-1e9，这是一个很大的负数值，在经过softmax激活函数后，会得到接近于零的值，从而attention score变为零，这样对应位置的token表征就不参与上文提到的按照权重加和的过程，这样达到了mask的效果。


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

因为transformer架构中，attention也是有很多层的，层与层之间也需要一些连接层, 这就是前馈神经网络（FFN），也就是这样的操作: linear + relu + linear 。这一层有什么作用呢？这可以做到对单个token的信息表征进行非线性变换（矩阵的运算可以理解为变换和平移），具体来说就是将多头注意力学到的东西进行了一波混合操作。


<br />
<br />


```python
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

```


这里为了实现在attention层的残差网络，当sublayer是attention机制的时候，我们可以看到，输入向量(tensor0)先经历了自注意力操作，然后dropout操作，再然后是与自己（tensor0）加起来，最后还有一个layernorm的操作。这里的残差网络，注意力操作，dropout, layernorm，都是一篇大名鼎鼎的论文的研究成果。
我们可以看到，transformer是集前人的成果而大成的，每一个小的研究成果在最后都起了作用，这就是机器学习领域不断积累却在进步的原因。

<br />
<br />


```python
## feed fowward network with residual
class FeedForwardNetwork(nn.Module):
    def __init__(self, sublayer: nn.Module, dimension: int, dropout: float = 0.1): ##不传入mask
        super().__init__()
        self.sublayer = sublayer
        self.norm = nn.LayerNorm(dimension)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tensors: Tensor) -> Tensor:
        # Assume that the "value" tensor is given last, so we can compute the
        # residual.  This matches the signature of 'MultiHeadAttention'.
        return self.norm(tensors + self.dropout(self.sublayer(tensors)))

```


这里实现在上文提到的前馈层的残差网络， 残差网络目前的深度学习的一个标准化组件了，这个组件使得层数非常深的深度学习训练过程中，训练效果不会退化。

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
        src = self.attention(src, src, src,mask) ###传入mask
        #return src 
        return self.feed_forward(src)
```

TransformerEncoderLayer这个类，就把前面实现的模块就包含组装起来了。其包含了attention层，和前馈神经网络（FFN）。

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
TransformerEncoder这个类，就把TransformerEncoderLayer这个类使用了好多次。这里的要注意的参数就是num_layers, 可以设置attention layer的层数，比如Llama2-7B有32层，因为层次很深，这也是深度学习这个名字的来源。


<br />
<br />



```python
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
```
decoder这里和encoder相比有点不一样，这里有两个attention。
attention_1对应的是decoder的自注意力，因为是下一词预测，所以需要传入一个mask。attention_2对应enc_out和tgt做的cross attention（tgt可以理解为query, enc_out可以理解为key, 和value），因为想利用src的所有信息，所以这里不需要mask(Mask=None)。 

<br />
<br />


```python
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
```
同encoder类似，多个decoder_layer串起来就组成decoder。

<br />
<br />



```python
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
```
最终，主角登场，最初始化函数中就可以看出来，transformer总体来看，Transformer包含encoder和decoder。以decoder为例，decoder有embedding, decoder_layer, out。其中embedding是嵌入层；decoder_layer是主要部分，其完成自注意力操作；out是输出层，把经过计算的向量转换为一个具有n_vocab维的向量。

<br />
<br />



```python
# Generates a square matrix where the each row allows one word more to be seen
def generate_masks(src):
    # current and next token mask
    seq_len= src.size(1)
    pad_int= [int(seq_len-i) for i in src.count_nonzero(dim=1)]
    mask = torch.tril(torch.ones(seq_len, seq_len) == 1) # Lower triangular matrix
    mask = mask.float()
    mask = mask.masked_fill(mask == 0, -1e9) # Convert zeros to -1e9
    mask = mask.masked_fill(mask == 1, float(0.0)) # Convert ones to 0
    # pad mask
    mask_arr=[]
    for i in pad_int:
      mask[:,-(i):]= -1e9
      mask_arr.append(mask)

    masks=torch.cat(tuple(mask_arr),dim=0)
    masks=masks.reshape(src.size(0),seq_len,seq_len)
    return masks

```
这个函数专门用来产生mask，这个mask作用于attention layer, mask两个方面地信息。第一种是预测当前token的时候，当前token以及其之后的token都要被mask掉，确保只使用当前token之前的token信息来预测当前的token， 这是为了防止信息泄露。因为attention layer是有很多层的，如果在第一层，当前token Xn融合了下一个token Xn+1的信息，在下一层attention layer计算时，token Xn+1会融合到Xn中包含的Xn+1， 这样在预测token Xn+1的时候，使用自己的信息预测自己，这显然是一种信息泄露。
第二种是要对 <PAD> 这些token进行mask， 因为这些token是我们为了统一所有句子的长度方便批量输入特意补上的，实际上没有什么意思。

<br />
<br />


```python
def loss_masked(output, src, loss_fn):
    nonpad_int=src.count_nonzero(dim=1) #number of actual tokens
    # discard pad elements
    loss_res = 0
    for k,item in enumerate(nonpad_int):
        loss_res+=loss_fn(output[k][:int(item), :], src[k][:int(item)])

    return loss_res/src.size(0)
######
```
如上文所说，<PAD>是我们为了统一所有句子的长度方便批量输入特意补上的，实际上没有什么意思。因此在计算损失函数的时候，这些token产生的结果就不要囊括进去了。
我们的做法就是，找到所有nonpad token，然后只计算这些nonpad token的损失函数。


<br />
<br />



```python
import time
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
    # save model parameters after finish training model
    torch.save(model, "./model.pkl")

```
数据可模型准备好了之后，就是要进入训练阶段了，这里先专门准备一个train的函数。

<br />
<br />


```python
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
    src, tgt= sent_idx_en, sent_idx_zh
    n_vocab_en= len(v2i_en) # encoder and decoder both need one vocab
    n_vocab_zh= len(v2i_zh)
    # build model
    m= Transformer(num_encoder_layers= N_LAYER,dim_model= MODEL_DIM, num_heads= N_HEAD, n_vocab_src=n_vocab_en, n_vocab_tgt=n_vocab_zh, device=device)
    # start training
    train(m, src, tgt, batch_size)

```
执行这个代码块之后，训练过程正式开始，模型和数据都加载到硬件（cpu或者gpu）, 马不停蹄地更新模型参数，这就是模型学习地过程。

<br />
<br />

## 总结

总体而言，在想法创意上，这个文章提出了self attention，multi-head attention，position encoding; 在工程实践上，也有很多的优秀之处，如resnet，layer normalisation, dropout各种技能的联合使用。这样的工作，便推出了机器翻译的最佳成绩。


## 拓展工作

通过学习transformer的源码实现，对这个模型的细枝末节可以说是非常了解了。可以看出，transformer实质上是对序列信息的建模，因此应用不仅仅在机器翻译，其他领域如图像，语音，甚至生物信息都可以应用。有了源码在手，可以修改模型的结构，或者注入先验知识使其在细分领域有用，参考DETR。





## 参考
* (1) https://medium.com/the-dl/transformers-from-scratch-in-pytorch-8777e346ca51
* (2) https://arxiv.org/abs/1706.03762
