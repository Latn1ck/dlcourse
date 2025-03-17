import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
import os
import random
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA 


class StanfordTreeBank:
    '''
    Wrapper for accessing Stanford Tree Bank Dataset
    https://nlp.stanford.edu/sentiment/treebank.html
    
    Parses dataset, gives each token and index and provides lookups
    from string token to index and back
    
    Allows to generate random context with sampling strategy described in
    word2vec paper:
    https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf
    '''
    def __init__(self):
        self.index_by_token = {}
        self.token_by_index = []

        self.sentences = []

        self.token_freq = {}
        
        self.token_reject_by_index = None

    def load_dataset(self, folder):
        filename = os.path.join(folder, "datasetSentences.txt")

        with open(filename, "r", encoding="latin1") as f:
            l = f.readline() # skip the first line
            
            for l in f:
                splitted_line = l.strip().split()
                words = [w.lower() for w in splitted_line[1:]] # First one is a number
                    
                self.sentences.append(words)
                for word in words:
                    if word in self.token_freq:
                        self.token_freq[word] +=1 
                    else:
                        index = len(self.token_by_index)
                        self.token_freq[word] = 1
                        self.index_by_token[word] = index
                        self.token_by_index.append(word)
        self.compute_token_prob()
                        
    def compute_token_prob(self):
        words_count = np.array([self.token_freq[token] for token in self.token_by_index])
        words_freq = words_count / np.sum(words_count)
        
        # Following sampling strategy from word2vec paper:
        # https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf
        self.token_reject_by_index = 1- np.sqrt(1e-5/words_freq)
    
    def check_reject(self, word):
        return np.random.rand() > self.token_reject_by_index[self.index_by_token[word]]
        
    def get_random_context(self, context_length=5):
        """
        Returns tuple of center word and list of context words
        """
        sentence_sampled = []
        while len(sentence_sampled) <= 2:
            sentence_index = np.random.randint(len(self.sentences)) 
            sentence = self.sentences[sentence_index]
            sentence_sampled = [word for word in sentence if self.check_reject(word)]
    
        center_word_index = np.random.randint(len(sentence_sampled))
        
        words_before = sentence_sampled[max(center_word_index - context_length//2,0):center_word_index]
        words_after = sentence_sampled[center_word_index+1: center_word_index+1+context_length//2]
        
        return sentence_sampled[center_word_index], words_before+words_after
    
    def num_tokens(self):
        return len(self.token_by_index)
        
data = StanfordTreeBank()
data.load_dataset("./stanfordSentimentTreebank/")

class Word2VecPlain(Dataset):
    '''
    PyTorch Dataset for plain Word2Vec.
    Accepts StanfordTreebank as data and is able to generate dataset based on
    a number of random contexts
    '''
    def __init__(self, data, num_contexts=30000):
        self.data=data
        self.num_contexts=num_contexts
        self.num_tokens = self.data.num_tokens()
        self.dataset=[]
    num_tokens=data.num_tokens
    def generate_dataset(self):
        '''
        Generates dataset samples from random contexts
        Note: there will be more samples than contexts because every context
        can generate more than one sample
        '''
        # TODO: Implement generating the dataset
        # You should sample num_contexts contexts from the data and turn them into samples
        # Note you will have several samples from one context
        for i in range(self.num_contexts):
            center_word, context = self.data.get_random_context()
            
            # Преобразуем центральное слово и контекстные слова в индексы
            center_word_idx = self.data.index_by_token[center_word]
            context_indices = [self.data.index_by_token[word] for word in context]
            
            # Генерируем пары (центральное слово, контекстное слово)
            for context_word_idx in context_indices:
                self.dataset.append((center_word_idx, context_word_idx))
        
    def __len__(self):
        '''
        Returns total number of samples
        '''
        # TODO: Return the number of samples
        return len(self.dataset)
    
    def __getitem__(self, index):
        '''
        Returns i-th sample
        
        Return values:
        input_vector - torch.Tensor with one-hot representation of the input vector
        output_index - index of the target word (not torch.Tensor!)
        '''
        # TODO: Generate tuple of 2 return arguments for i-th sample
        central_word_idx, context_word_idx = self.dataset[index]
        
        # Создаем one-hot представление для центрального слова
        input_vector = torch.zeros(self.num_tokens)
        input_vector[central_word_idx] = 1  # One-hot encoding

        # Контекстное слово возвращается как индекс
        output_index = context_word_idx
        return input_vector, output_index
dataset = Word2VecPlain(data, 30000)
dataset.generate_dataset()

# We'll be training very small word vectors!
wordvec_dim = 10

# We can use a standard sequential model for this
nn_model = nn.Sequential(
            nn.Linear(dataset.num_tokens, wordvec_dim, bias=False),
            nn.Linear(wordvec_dim, dataset.num_tokens, bias=False), 
         )
nn_model.type(torch.FloatTensor)
def extract_word_vectors(nn_model):
    '''
    Extracts word vectors from the model
    
    Returns:
    input_vectors: torch.Tensor with dimensions (num_tokens, num_dimensions)
    output_vectors: torch.Tensor with dimensions (num_tokens, num_dimensions)
    '''
    # Извлекаем веса первого и второго слоя
    input_vectors = nn_model[0].weight.data  # Входные веса, размеры: (num_dimensions, num_tokens)
    output_vectors = nn_model[1].weight.data  # Выходные веса, размеры: (num_tokens, num_dimensions)

    # Транспонируем input_vectors, чтобы получить размеры (num_tokens, num_dimensions)
    input_vectors = input_vectors.T  # Теперь: (num_tokens, num_dimensions)

    return input_vectors, output_vectors

untrained_input_vectors, untrained_output_vectors = extract_word_vectors(nn_model)
def train_model(model, dataset, train_loader, optimizer, scheduler, num_epochs):
    '''
    Trains plain word2vec using cross-entropy loss and regenerating dataset every epoch
    
    Returns:
    loss_history, train_history
    '''
    # Определяем функцию потерь
    criterion = nn.CrossEntropyLoss().type(torch.FloatTensor)
    
    loss_history = []
    train_history = []

    for epoch in range(num_epochs):
        model.train()  # Устанавливаем режим обучения
        dataset.generate_dataset()  # Обновляем датасет для каждой эпохи
        
        epoch_loss = 0
        correct_predictions = 0
        total_samples = 0

        for input_vector, target in train_loader:  # Итерируемся по DataLoader
            # Приводим данные к нужному типу
            input_vector = input_vector.type(torch.FloatTensor)
            target = target.type(torch.LongTensor)

            # Обнуляем градиенты оптимизатора
            optimizer.zero_grad()
            
            # Прямой проход
            outputs = model(input_vector)

            # Функция потерь
            loss = criterion(outputs, target)
            epoch_loss += loss.item()
            
            # Обратное распространение ошибки
            loss.backward()

            # Шаг оптимизатора
            optimizer.step()

            # Вычисляем точность (правильные предсказания)
            _, predictions = torch.max(outputs, 1)
            correct_predictions += (predictions == target).sum().item()
            total_samples += target.size(0)
        
        # Шаг изменения скорости обучения
        scheduler.step()

        # Считаем среднюю потерю и точность для эпохи
        ave_loss = epoch_loss / len(train_loader)
        train_accuracy = correct_predictions / total_samples

        loss_history.append(ave_loss)
        train_history.append(train_accuracy)
        
        print("Epoch %i, Average loss: %f, Train accuracy: %f" % (epoch, ave_loss, train_accuracy))
    
    return loss_history, train_history
optimizer = optim.SGD(nn_model.parameters(), lr=1, weight_decay=1e-3,momentum=0.8)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
train_loader = torch.utils.data.DataLoader(dataset, batch_size=20)

loss_history, train_history = train_model(nn_model, dataset, train_loader, optimizer, scheduler, 10)