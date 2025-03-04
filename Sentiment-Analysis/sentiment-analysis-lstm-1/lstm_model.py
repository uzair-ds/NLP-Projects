import torch
import torch.nn as nn

class SentimentClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, output_size):
        super(SentimentClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, bidirectional=True, batch_first=True)
        self.fc1 = nn.Linear(hidden_size * 2, 24)  # *2 because of bidirectional LSTM
        self.fc2 = nn.Linear(24, output_size)
        self.relu = nn.ReLU()
        

    def forward(self, x):
        embedded = self.embedding(x)
        output, _ = self.lstm(embedded)    
        lstm_out = output[:, -1, :]  
        fc1_out = self.relu(self.fc1(lstm_out))
        output = self.fc2(fc1_out)
        return output  