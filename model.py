import torch
import torch.nn as nn


def model(type):
    if type == 'basic':
        D = nn.Sequential(
            nn.Linear(784, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid())

        # Generator
        G = nn.Sequential(
            nn.Linear(64, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 784),
            nn.Tanh())


    elif type == 'condition':
        D = nn.Sequential(
            nn.Linear(794, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid())

        # Generator
        G = nn.Sequential(
            nn.Linear(74, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 784),
            nn.Tanh())

    if torch.cuda.is_available():
        D.cuda()
        G.cuda()
    return D, G