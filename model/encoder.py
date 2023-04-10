import torch.nn as nn
from torchvision.models import inception_v3


class EncoderCNN(nn.Module):
    def __init__(self, embed_dim, train_CNN=False) -> None:
        super(EncoderCNN, self).__init__()
        self.embed_dim = embed_dim

        self.inception =inception_v3(pretrained=True, aux_logits=True)
        self.inception.fc = nn.Linear(self.inception.fc.in_features, embed_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

        for name, param in self.inception.named_parameters():
            if 'fc.weights' in name or 'fc.bias' in name:
                param.requires_grad = True
            else:
                param.requires_grad = train_CNN

    def forward(self, images):
        featues, _ = self.inception(images)
        return self.dropout(self.relu(featues))

    def predict(self, images):
        features = self.inception(images)
        return self.dropout(self.relu(features))