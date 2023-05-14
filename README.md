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




## 参考
* (1) https://towardsdatascience.com/pre-processing-a-wikipedia-dump-for-nlp-model-training-a-write-up-3b9176fdf67
* (2) https://medium.com/the-dl/transformers-from-scratch-in-pytorch-8777e346ca51
