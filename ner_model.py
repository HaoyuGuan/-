import os
from torch.utils.data import Dataset, DataLoader
import torch
from torch import nn


class BiLstm(nn.Module):
    def __init__(self, corpus_num, embedding_num, hidden_num, class_num, bi=True):
        super().__init__()
        self.embedding = nn.Embedding(corpus_num, embedding_num)
        self.lstm = nn.LSTM(embedding_num, hidden_num, batch_first=True, bidirectional=bi)
        if bi:
            self.fc = nn.Linear(hidden_num * 2, class_num)
        else:
            self.fc = nn.Linear(hidden_num, class_num)

    def forward(self, batch_data):
        embedding = self.embedding(batch_data)
        out, _ = self.lstm(embedding)
        return self.fc(out)
