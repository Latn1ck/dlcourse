import numpy as np
import nltk
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import FloatTensor, LongTensor
import matplotlib.pyplot as plt
from collections import Counter


np.random.seed(42)
nltk.download('brown')
nltk.download('universal_tagset')
data = nltk.corpus.brown.tagged_sents(tagset='universal')
train_data, test_data = train_test_split(data, test_size=0.25, random_state=42)
train_data, val_data = train_test_split(train_data, test_size=0.15, random_state=42)
words = {word for sample in train_data for word, tag in sample}
word2ind = {word: ind + 1 for ind, word in enumerate(words)}
word2ind['<pad>'] = 0
tags = {tag for sample in train_data for word, tag in sample}
tag2ind = {tag: ind + 1 for ind, tag in enumerate(tags)}
tag2ind['<pad>'] = 0

default_tagger = nltk.DefaultTagger('NN')
unigram_tagger = nltk.UnigramTagger(train_data, backoff=default_tagger)
bigram_tagger = nltk.BigramTagger(train_data, backoff=unigram_tagger)
trigram_tagger = nltk.TrigramTagger(train_data)
def convert_data(data, word2ind, tag2ind):
    X = [[word2ind.get(word, 0) for word, _ in sample] for sample in data]
    y = [[tag2ind[tag] for _, tag in sample] for sample in data]
    return X, y
X_train, y_train = convert_data(train_data, word2ind, tag2ind)
X_val, y_val = convert_data(val_data, word2ind, tag2ind)
X_test, y_test = convert_data(test_data, word2ind, tag2ind)
def iterate_batches(data, batch_size):
    X, y = data
    n_samples = len(X)

    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    
    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        batch_indices = indices[start:end]
        max_sent_len = max(len(X[ind]) for ind in batch_indices)
        X_batch = np.zeros((max_sent_len, len(batch_indices)))
        y_batch = np.zeros((max_sent_len, len(batch_indices)))
        for batch_ind, sample_ind in enumerate(batch_indices):
            X_batch[:len(X[sample_ind]), batch_ind] = X[sample_ind]
            y_batch[:len(y[sample_ind]), batch_ind] = y[sample_ind]
        yield X_batch, y_batch
X_batch, y_batch = next(iterate_batches((X_train, y_train), 4))
X_batch.shape, y_batch.shape
class LSTMTagger(nn.Module):
    def __init__(self, vocab_size, tagset_size, word_emb_dim=100, lstm_hidden_dim=128, lstm_layers_count=1):
        super().__init__()
        self.embedding=nn.Embedding(vocab_size, word_emb_dim)
        self.lstm = nn.LSTM(
        input_size=word_emb_dim,
        hidden_size=lstm_hidden_dim,
        num_layers=lstm_layers_count,
        batch_first=True
    )
        self.hidden2tag = nn.Linear(lstm_hidden_dim, tagset_size)
        self.hidden_dim = lstm_hidden_dim
    def forward(self, inputs):
    # Шаг 1: Преобразование входных индексов слов в эмбеддинги
        embeds = self.embedding(inputs)  # Размерность: (batch_size, seq_len, embedding_dim)
    
    # Шаг 2: Пропуск через LSTM
        lstm_out, _ = self.lstm(embeds)  # lstm_out: (batch_size, seq_len, hidden_dim)
    
    # Шаг 3: Преобразование выходов LSTM в теги через полносвязный слой
        tag_space = self.hidden2tag(lstm_out)  # Размерность: (batch_size, seq_len, tagset_size)
    
    # Шаг 4: Нормализация через LogSoftmax для получения вероятностей тегов
        tag_scores = torch.log_softmax(tag_space, dim=2)  # Нормализация по размерности тегов
    
        return tag_scores
model = LSTMTagger(vocab_size=len(word2ind), tagset_size=len(tag2ind))
X_batch, y_batch = torch.LongTensor(X_batch), torch.LongTensor(y_batch)
logits = model(X_batch)
predicted = torch.argmax(logits, dim=1)
predicted_tags = torch.argmax(logits, dim=2)  # Shape: (32, 4)

# Compare with ground truth
correct_predictions = (predicted_tags == y_batch).sum().item()

# Total number of elements
total_predictions = y_batch.numel()

# Accuracy calculation
accuracy = correct_predictions / total_predictions
print(f"Точность: {accuracy:.2%}")  # Результат, например, 100.00%
criterion = nn.CrossEntropyLoss()
#<calc loss>