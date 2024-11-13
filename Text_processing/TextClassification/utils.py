
from transformers import AutoModel, AutoTokenizer
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from peft import get_peft_model, LoraConfig
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optimizer

# Sample dataset for text classification
texts = [
    "The football team won the championship in a thrilling final game",  # Class 1: Sports
    "The government passed a new bill in the parliament today",  # Class 2: Politics
    "The latest smartphone model includes an improved camera and processor",  # Class 3: Technology
    "The tennis player broke several records this season",  # Class 1: Sports
    "New policies on climate change were discussed at the international summit",  # Class 2: Politics
    "A breakthrough in artificial intelligence was announced by tech companies",  # Class 3: Technology
]

# Corresponding labels (integer-coded)
labels = [0, 1, 2, 0, 1, 2]  # 0: Sports, 1: Politics, 2: Technology

def cluster_emb(embeddings, labels, method='PCA'):

    if method == 'Kmeans':
        kmeans = KMeans(n_clusters=5, random_state=0).fit(embeddings)
        labels = kmeans.labels_
    elif method == 'PCA':
        pca = PCA(n_components=2)
        reduced_embeddings = pca.fit_transform(embeddings)

        plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=labels, alpha=0.7)
        plt.show()
    elif method == 'tSNE':
        tsne = TSNE(n_components=2)
        reduced_embeddings = tsne.fit_transform(embeddings)

        plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=emotions, alpha=0.7)
        plt.show()
    else:
        print('Undefined Method')

class myNNHead(nn.Module):
    def __init__(self, n_hidden=30) -> None:
        super().__init__()
        self.fc = nn.Linear(in_features=768, out_features=n_hidden)
        self.out = nn.Linear(in_features=n_hidden, out_features=3)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc(x))

        return self.out(x)

class combine_nets(nn.Module):
    def __init__(self, basemodel, headmodel) -> None:
        super().__init__()
        self.basemodel = basemodel
        self.headmodel = headmodel

    def forward(self, x):
        x = self.basemodel(x)
        x = x.last_hidden_state.mean(dim=1)

        return self.headmodel(x)
