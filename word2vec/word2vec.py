# -*- coding: utf-8 -*-
"""
# Implement word2vec using pytorch
"""

import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter
from torch.utils.data import DataLoader, Dataset

# Example corpus
corpus = [
    "the quick brown fox jumps over the lazy dog",
    "I love learning new things about machine learning"
]

# Tokenize
words = [word for sentence in corpus for word in sentence.split()]
vocab = set(words)

word_to_ix = {word: i for i, word in enumerate(vocab)}
ix_to_word = {i: word for word, i in word_to_ix.items()}

# Hyperparameters
context_size = 2  # Size of the window
embedding_dim = 10  # Size of the embedding vector

# Create context-target pairs

#cbow data
cbow_data = []
for sentence in corpus:
    tokens = sentence.split()
    for i in range(context_size, len(tokens) - context_size):
        context = [tokens[i - j - 1] for j in range(context_size)] + [tokens[i + j + 1] for j in range(context_size)]
        target = tokens[i]
        cbow_data.append((context, target))
# use target to predict context
# skipgram data
skipgram_data=[]
for item in cbow_data:
    skipgram_data = skipgram_data + [([item[1]],i) for i in item[0]]



class Word2Vec(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(Word2Vec, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs).mean(dim=0)  # Average the embeddings of context words
        out = self.linear(embeds)
        #log_probs = torch.log_softmax(out, dim=0)
        return out

# Custom dataset
class Word2VecDataset(Dataset):
    def __init__(self, data, word_to_ix):
        self.data = data
        self.word_to_ix = word_to_ix

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        context, target = self.data[idx]
        context_idxs = torch.tensor([self.word_to_ix[w] for w in context], dtype=torch.long)
        target_idx = torch.tensor([self.word_to_ix[target]], dtype=torch.long)
        return context_idxs, target_idx


# Model, Loss, and Optimizer
model = Word2Vec(len(vocab), embedding_dim)
loss_function =  nn.CrossEntropyLoss() ## use this loss

optimizer = optim.SGD(model.parameters(), lr=0.001)


def train(model, data, epochs=10, batch_size=1, lr=0.001):
    dataset = Word2VecDataset(data, word_to_ix)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    # Training loop
    for epoch in range(epochs):
        for context, target in dataloader:
            model.zero_grad()
            log_probs = model(context.squeeze(0))
            # print(context, target.squeeze(), '\n\n\n' )
            loss = loss_function(log_probs, target.squeeze() )
            loss.backward()
            optimizer.step()

            #total_loss += loss.item()
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')

if __name__ == '__main__':
    print("train model using skipgram data")
    train(model, skipgram_data, epochs=10, batch_size=1, lr=0.001)
    print("train model using cbow data")
    train(model, cbow_data, epochs=10, batch_size=1, lr=0.001)



"""# Implement word2vec using numpy"""

import numpy as np

# Example corpus
corpus = [
    "the quick brown fox jumps over the lazy dog",
    "I love learning new things about machine learning"
]

# Tokenize
words = [word for sentence in corpus for word in sentence.split()]
vocab = set(words)
vocab_size = len(vocab)

word_to_ix = {word: i for i, word in enumerate(vocab)}
ix_to_word = {i: word for word, i in word_to_ix.items()}

# Hyperparameters
context_size = 2  # Size of the window
embedding_dim = 10  # Size of the embedding vector

# Create context-target pairs

#skipgram
cbow_data = []
for sentence in corpus:
    tokens = sentence.split()
    for i in range(context_size, len(tokens) - context_size):
        context = [tokens[i - j - 1] for j in range(context_size)] + [tokens[i + j + 1] for j in range(context_size)]
        target = tokens[i]
        cbow_data.append((context, target))
# use target to predict context
skipgram_data=[]
for item in cbow_data:
    skipgram_data = skipgram_data + [([item[1]],i) for i in item[0]]



# Helper functions
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def to_one_hot(word_idx, vocab_size):
    x = np.zeros(vocab_size)
    x[word_idx] = 1
    return x

def cross_entropy_loss(y_pred, target_idx):
    return -np.log(y_pred[target_idx])


class Word2Vec_numpy():
    def __init__(self, vocab_size, embedding_dim, learning_rate):
        super(Word2Vec_numpy, self).__init__()
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.lr= learning_rate
        # Weight initialization
        self.W1 = np.random.rand(embedding_dim, vocab_size)  # Context to hidden
        self.W2 = np.random.rand(vocab_size, embedding_dim)  # Hidden to output


    def forward_pass(self, context_words):
        # Average context word vectors
        x = np.mean([to_one_hot(word_to_ix[w], self.vocab_size) for w in context_words], axis=0)
        h = np.dot(self.W1, x) # role of embedding layer
        u = np.dot(self.W2, h) # role of hidden layer
        y_pred = softmax(u)
        return x, h, y_pred

    def backward(self, error, h, x):
        dW2 = np.outer(error, h)
        dW1 = np.outer(np.dot(self.W2.T, error), x)

        # Update weights
        self.W1 -= self.lr * dW1
        self.W2 -= self.lr * dW2
        return self.W1, self.W2

def train(model, data, epochs=10, batch_size=1, lr=0.001):
    for epoch in range(epochs):
        loss = 0
        for context, target in data:
            x, h, y_pred = model.forward_pass(context)
            target_idx = word_to_ix[target]
            loss += cross_entropy_loss(y_pred, target_idx)

            # Error to propagate back
            error = y_pred.copy()
            error[target_idx] -= 1

            W1, W2 = model.backward(error, h, x)

        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {loss}')

if __name__ == '__main__':
    # Training
    epochs = 10
    lr = 0.001
    model = Word2Vec_numpy(vocab_size, embedding_dim, lr)
    print("train model using skipgram data")
    train(model, skipgram_data, epochs=10, batch_size=1, lr=0.001)
    print("train model using cbow data")
    train(model, cbow_data, epochs=10, batch_size=1, lr=0.001)

