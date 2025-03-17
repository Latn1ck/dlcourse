import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
import numpy as np
import os


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
        self.index_by_token = {} # map of string -> token index
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
        
        # Following sampling strategy from word2vec paper
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

print("Num tokens:", data.num_tokens())
for i in range(5):
    center_word, other_words = data.get_random_context(5)
    print(center_word, other_words)

num_negative_samples = 10

class Word2VecNegativeSampling(Dataset):
    '''
    PyTorch Dataset for Word2Vec with Negative Sampling.
    Accepts StanfordTreebank as data and is able to generate dataset based on
    a number of random contexts
    '''
    def __init__(self, data, num_negative_samples, num_contexts=30000):
        '''
        Initializes Word2VecNegativeSampling, but doesn't generate the samples yet
        (for that, use generate_dataset)
        Arguments:
        data - StanfordTreebank instace
        num_negative_samples - number of negative samples to generate in addition to a positive one
        num_contexts - number of random contexts to use when generating a dataset
        '''
        self.data=data
        self.num_negative_samples=num_negative_samples
        self.num_contexts=num_contexts
        self.dataset=[]
    
    def generate_dataset(self):
        '''
        Generates dataset samples from random contexts
        Note: there will be more samples than contexts because every context
        can generate more than one sample
        '''
        # TODO: Implement generating the dataset
        # You should sample num_contexts contexts from the data and turn them into samples
        # Note you will have several samples from one context
        center_word, context_words = self.data.get_random_context()
    
        for context_word in context_words:
            positive_sample = (center_word, context_word, 1)
            self.dataset.append(positive_sample)
            for i in range(self.num_negative_samples):
                while True:
                    negative_word = np.random.choice(self.data.token_by_index)
                    if negative_word != center_word and negative_word not in context_words:
                        break
                negative_sample = (center_word, negative_word, 0)
                self.dataset.append(negative_sample)
    def __len__(self):
        '''
        Returns total number of samples
        '''
        return len(self.dataset)

    
    def __getitem__(self, index):
        '''
        Returns i-th sample
        
        Return values:
        input_vector - index of the input word (not torch.Tensor!)
        output_indices - torch.Tensor of indices of the target words. Should be 1+num_negative_samples.
        output_target - torch.Tensor with float targets for the training. Should be the same size as output_indices
                        and have 1 for the context word and 0 everywhere else
        '''
        sample = self.dataset[index]
        center_word, context_word, label = sample
        input_vector = self.data.index_by_token[center_word]
        positive_index = self.data.index_by_token[context_word]
        negative_indices = []
        while len(negative_indices) < self.num_negative_samples:
            negative_word = np.random.choice(self.data.token_by_index)
            negative_index = self.data.index_by_token[negative_word]
            if negative_index != positive_index and negative_index not in negative_indices:
                negative_indices.append(negative_index)
        output_indices = torch.tensor([positive_index] + negative_indices, dtype=torch.long)
        output_target = torch.tensor([1.0] + [0.0] * self.num_negative_samples, dtype=torch.float32)
        return input_vector, output_indices, output_target
        

dataset = Word2VecNegativeSampling(data, num_negative_samples, 10)
dataset.generate_dataset()
input_vector, output_indices, output_target = dataset[0]
print("Sample - input: %s, output indices: %s, output target: %s" % (int(input_vector), output_indices, output_target)) # target should be able to convert to int
assert isinstance(output_indices, torch.Tensor)
assert output_indices.shape[0] == num_negative_samples+1
assert isinstance(output_target, torch.Tensor)
assert output_target.shape[0] == num_negative_samples+1
assert torch.sum(output_target) == 1.0
# Create the usual PyTorch structures
dataset = Word2VecNegativeSampling(data, num_negative_samples, 30000)
dataset.generate_dataset()

# As before, we'll be training very small word vectors!
wordvec_dim = 10

class Word2VecNegativeSamples(nn.Module):
    def __init__(self, num_tokens):
        super(Word2VecNegativeSamples, self).__init__()
        self.input = nn.Linear(num_tokens, 10, bias=False)
        self.output = nn.Linear(10, num_tokens, bias=False)
        
    def forward(self, input_index_batch, output_indices_batch):
        '''
    Implements forward pass with negative sampling.

    Arguments:
    input_index_batch - Tensor of ints, shape: (batch_size,), indices of input words in the batch.
    output_indices_batch - Tensor of ints, shape: (batch_size, num_negative_samples+1),
                           indices of the target words for every sample.

    Returns:
    predictions - Tensor of floats, shape: (batch_size, num_negative_samples+1).
    '''
        batch_predictions = []
        for input_index, output_indices in zip(input_index_batch, output_indices_batch):
            input_one_hot = torch.zeros(self.input.weight.shape[1])
            input_one_hot[input_index] = 1.0
            input_vector = self.input(input_one_hot)
            output_vectors = self.output.weight[output_indices]
            predictions = torch.matmul(output_vectors, input_vector)
            batch_predictions.append(predictions)
        return torch.stack(batch_predictions)
   
nn_model = Word2VecNegativeSamples(data.num_tokens())
nn_model.type(torch.FloatTensor)
def extract_word_vectors(nn_model):
    '''
    Extracts word vectors from the model
    
    Returns:
    input_vectors: torch.Tensor with dimensions (num_tokens, num_dimensions)
    output_vectors: torch.Tensor with dimensions (num_tokens, num_dimensions)
    '''
    input_vectors = nn_model.input.weight.data.T
    output_vectors = nn_model.output.weight.data
    return input_vectors, output_vectors

untrained_input_vectors, untrained_output_vectors = extract_word_vectors(nn_model)
assert untrained_input_vectors.shape == (data.num_tokens(), wordvec_dim)
assert untrained_output_vectors.shape == (data.num_tokens(), wordvec_dim)
def train_neg_sample(model, dataset, train_loader, optimizer, scheduler, num_epochs):    
    '''
    Trains word2vec with negative samples on and regenerating dataset every epoch
    
    Returns:
    loss_history, train_history
    '''
    loss_fn = nn.BCEWithLogitsLoss()
    loss_history = []
    train_history = []
    for epoch in range(num_epochs):
        model.train()
        dataset.generate_dataset()
        epoch_loss = 0
        correct_predictions = 0
        total_samples = 0
        for input_vector, output_indices, output_target in train_loader:
            optimizer.zero_grad()
            predictions = model(input_vector, output_indices)
            loss = loss_fn(predictions, output_target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * input_vector.size(0)
            predicted = (predictions >= 0).float()
            correct_predictions += torch.sum(predicted == output_target).item()
            total_samples += output_target.size(0)
        scheduler.step()
        ave_loss = epoch_loss / total_samples
        train_accuracy = correct_predictions / total_samples
        loss_history.append(ave_loss)
        train_history.append(train_accuracy)
        print("Epoch %d: Average loss: %.6f, Train accuracy: %.6f" % (epoch + 1, ave_loss, train_accuracy))

    return loss_history, train_history
# Finally, let's train the model!

# TODO: We use placeholder values for hyperparameters - you will need to find better values!
optimizer = optim.SGD(nn_model.parameters(), lr=1, weight_decay=0.1)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
train_loader = torch.utils.data.DataLoader(dataset, batch_size=20)

loss_history, train_history = train_neg_sample(nn_model, dataset, train_loader, optimizer, scheduler, 10)