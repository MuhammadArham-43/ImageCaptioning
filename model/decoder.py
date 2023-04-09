import torch
import torch.nn as nn


class DecoderRNN(nn.Module):
    def __init__(self, embed_dim, hidden_dim, vocab_dim) -> None:
        super(DecoderRNN, self).__init__()

        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.vocab_dim = vocab_dim

        self.embedding = nn.Embedding(vocab_dim, embed_dim)
        self.linear = nn.Linear(hidden_dim, vocab_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, vocab_dim)
        self.dropout = nn.Dropout(0.5)

    def forward(self, features, captions):
        embeddings = self.dropout(self.embedding(captions))
        embeddings = torch.cat((features.unsqueeze(0), embeddings), dim=0)

        hiddens, _ = self.lstm(embeddings)
        outputs = self.linear(hiddens)
        return outputs
