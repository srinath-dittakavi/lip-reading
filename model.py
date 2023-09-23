from argparse import ArgumentError
import math
import torch
import torch.nn as nn
from torch.autograd import Variable

from pytorch_i3d import InceptionI3d

from config import I3D_WEIGHTS_PATH


class LipReadingModel(nn.Module):
    def __init__(self, sequence_length, vocab_size, in_channels=3):
        super(LipReadingModel, self).__init__()
        # Vocab size is the length of the vocabulary
        # in_channels refers to RBG/greyscale etc
        # sequence_length is the length of input in frames
        #   frames will be given to the i3d model in segments of 16 frames, one word predicted per 16 frames

        # if (sequence_length % 16 != 0):
        #     raise ArgumentError('Sequence length must be divisable by 16')

        self.vocab_size = vocab_size
        self.in_channels = 3

        self.i3d_embedding_length = 400

        self.sequence_length = sequence_length
        self.segment_length = 16
        self.segments = int(sequence_length / self.segment_length)


        i3d = InceptionI3d(in_channels=self.in_channels, num_classes=400)
        i3d.load_state_dict(torch.load(I3D_WEIGHTS_PATH))

        self.i3d = i3d

        encoder_layer = nn.TransformerEncoderLayer(d_model=self.i3d_embedding_length, nhead=4)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)

        self.fc = nn.Linear(self.i3d_embedding_length, self.vocab_size)



    def forward(self, x):
        b_s, channels, frames, width, height = x.shape
        print('x shape: ', x.shape)
        features = self.i3d(x)
        features = features.transpose(1, 2)
        print('features shape: ', features.shape)
        encoded_features = self.transformer_encoder(features)
        print('encoded_features shape: ', encoded_features.shape)
        vocab_predictions = self.fc(encoded_features)
        print('vocab_predictions shape: ', vocab_predictions.shape)

        return vocab_predictions