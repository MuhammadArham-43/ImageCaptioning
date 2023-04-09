import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms

from model.model import CNN2RNN
from Dataset.flickrDataset import getLoader

from tqdm import tqdm


if __name__ == "__main__":
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    EMBED_DIM = 256
    HIDDEN_DIM = 256
    NUM_LAYERS = 1

    NUM_EPOCHS = 100
    LR = 1e-4
    BATCH_SIZE = 4

    transform = transforms.Compose([
        transforms.Resize((356, 356)),
        transforms.RandomCrop((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_loader, dataset = getLoader(
        root_dir='data/Flickr8k',
        annotation_file='data/Flickr8k/captions.txt',
        transform=transform,
        batch_size=BATCH_SIZE
    )

    VOCAB_DIM = len(dataset.vocab)

    writer = SummaryWriter('runs/Flick8k')
    step = 0

    model = CNN2RNN(
        embed_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
        vocab_dim=VOCAB_DIM
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss(
        ignore_index=dataset.vocab.strToIdx["<PAD>"])
    optimizer = optim.Adam(model.parameters(), lr=LR)

    model.train()

    print("STARTING TRAINING...")
    for epoch in tqdm(range(NUM_EPOCHS)):
        for idx, (images, captions) in enumerate(train_loader):
            images = images.to(DEVICE)
            captions = captions.to(DEVICE)
            
            outputs = model(images, captions[:-1])
            loss = criterion(
                outputs.reshape(-1, outputs.shape[2]), captions.reshape(-1))

            writer.add_scalar('Traning Loss', loss.item())
            optimizer.zero_grad()
            loss.backward(loss)
            optimizer.step()
