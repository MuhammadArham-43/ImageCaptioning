import os
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torchvision.transforms import Compose, Resize, ToTensor

import spacy

START_TAG = "<SOS>"
END_TAG = "<EOS>"
PAD_TAG = "<PAD>"
UNKOWN_TAG = "<UNK>"


spacy_eng = spacy.load('en_core_web_sm')


class Vocabulary:
    def __init__(self, freq_threshold) -> None:
        self.freq_threshold = freq_threshold

        self.idxToStr = {0: PAD_TAG, 1: START_TAG, 2: END_TAG, 3: UNKOWN_TAG}
        self.strToIdx = {PAD_TAG: 0, START_TAG: 1, END_TAG: 2, UNKOWN_TAG: 3}

    def __len__(self):
        return len(self.idxToStr)

    @staticmethod
    def tokenizer_eng(text):
        return [tok.text.lower() for tok in spacy_eng.tokenizer(text)]

    def numericalize(self, text):
        tokenized_text = self.tokenizer_eng(text)
        return [
            self.strToIdx[token] if token in self.strToIdx else self.strToIdx[UNKOWN_TAG] for token in tokenized_text
        ]

    def build_vocabulary(self, sentences):
        frequencies = {}
        idx = len(self.idxToStr)

        for sentence in sentences:
            for word in self.tokenizer_eng(sentence):
                if word not in frequencies:
                    frequencies[word] = 1
                else:
                    frequencies[word] += 1

                if frequencies[word] == self.freq_threshold:
                    self.idxToStr[idx] = word
                    self.strToIdx[word] = idx
                    idx += 1


class FlickrDataset(Dataset):
    def __init__(self,
                 root_dir,
                 annotations_file,
                 freq_treshold=5,
                 transform=None,
                 ) -> None:
        super().__init__()
        self.root_dir = root_dir
        self.annotations_file = annotations_file
        self.freq_threshold = freq_treshold
        self.transform = transform

        self.df = pd.read_csv(annotations_file)
        self.images = self.df['image']
        self.captions = self.df['caption']

        self.vocab = Vocabulary(self.freq_threshold)
        self.vocab.build_vocabulary(self.captions.tolist())

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        img_id, caption = self.images[index], self.captions[index]

        img = Image.open(os.path.join(
            self.root_dir, 'images', img_id)).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        numericalized_caption = [self.vocab.strToIdx[START_TAG]]
        numericalized_caption += self.vocab.numericalize(caption)
        numericalized_caption.append(self.vocab.strToIdx[END_TAG])

        return img, torch.tensor(numericalized_caption)


class PadCollate:
    def __init__(self, pad_idx) -> None:
        self.pad_idx = pad_idx

    def __call__(self, batch):
        imgs = [item[0].unsqueeze(0) for item in batch]
        imgs = torch.cat(imgs, dim=0)

        targets = [item[1] for item in batch]
        targets = pad_sequence(targets, batch_first=False,
                               padding_value=self.pad_idx)

        return imgs, targets


def getLoader(
    root_dir,
    annotation_file,
    transform=None,
    batch_size=32,
    shuffle=True,
):
    dataset = FlickrDataset(
        root_dir, annotations_file=annotation_file, transform=transform)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=PadCollate(dataset.vocab.strToIdx[PAD_TAG])
    )

    return dataloader, dataset


if __name__ == "__main__":
    ROOT_DIR = 'data/Flickr8k'
    ANNOTATIONS_FILE = 'data/Flickr8k/captions.txt'
    transform = Compose([
        Resize((224, 224)),
        ToTensor()
    ])

    BATCH_SIZE = 4
    SHUFFLE = True

    dataloader = getLoader(
        ROOT_DIR,
        ANNOTATIONS_FILE,
        transform=transform,
        batch_size=BATCH_SIZE,
        shuffle=SHUFFLE
    )

    for idx, (imgs, captions) in enumerate(dataloader):
        print(imgs.shape, captions.shape)
