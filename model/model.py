import torch
import torch.nn as nn

from .decoder import DecoderRNN
from .encoder import EncoderCNN


class CNN2RNN(nn.Module):
    def __init__(self, embed_dim, hidden_dim, vocab_dim) -> None:
        super(CNN2RNN, self).__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.vocab_dim = vocab_dim

        self.encoder = EncoderCNN(embed_dim)
        self.decoder = DecoderRNN(embed_dim, hidden_dim, vocab_dim)

    def forward(self, images, captions):
        features = self.encoder(images)
        outputs = self.decoder(features, captions)
        return outputs

    def predict_caption(self, image, vocabulary, max_len=50):
        caption = []

        with torch.no_grad():
            x = self.encoder(image).unsqueeze(0)
            states = None

            for _ in range(max_len):
                hiddens, states = self.decoder.lstm(x, states)
                output = self.decoder.linear(hiddens.squeeze(0))
                predicted = output.argmax(1)

                caption.append(predicted.item())
                x = self.decoder.embed(predicted).unsqueeze(0)

                if vocabulary.idxToStr[predicted.item()] == "<EOS>":
                    break

            return [vocabulary.idxToStr[idx] for idx in caption]
