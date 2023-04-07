# 实现Pytorch版本的GPT模型

## 为什么要自己从头实现GPT?

通过逐步实现GPT模型，了解基于transformer的语言模型是如何工作的。同时，了解预训练语料是如何准备的，也是很有必要的。有了模型和数据，在家用电脑就可以上手训练当今最前沿的语言模型。在此基础上，可以对模型提出修改甚至创新，推动自然语言处理发展。

## 准备数据

使用wikipedia 文本数据 (file00包含wikipedia前80000个句子，已经清洗干净)


## 模型代码解读

模型代码在同目录的pytorch-gpt文件夹中，包含两个jupyter notebooks。下面对cpu_single_gpu版本的gpt模型代码进行解说。

```python
# 使用pytorch数据格式
from torch import Tensor
import torch.nn.functional as f
```


## 参考
* (1) https://towardsdatascience.com/pre-processing-a-wikipedia-dump-for-nlp-model-training-a-write-up-3b9176fdf67
* (2) https://medium.com/the-dl/transformers-from-scratch-in-pytorch-8777e346ca51
