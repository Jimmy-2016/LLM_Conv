
import numpy as np
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def read_conversation(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    # Remove any newline characters
    conversation = [line.strip() for line in lines]
    return conversation


def cluster_emb(embeddings, emotions, method='Kmeans'):

    if method == 'Kmeans':
        kmeans = KMeans(n_clusters=5, random_state=0).fit(embeddings)
        labels = kmeans.labels_
    elif method == 'PCA':
        pca = PCA(n_components=2)
        reduced_embeddings = pca.fit_transform(embeddings)

        plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=emotions, alpha=0.7)
        plt.show()
    elif method == 'tSNE':
        tsne = TSNE(n_components=2)
        reduced_embeddings = tsne.fit_transform(embeddings)

        plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=emotions, alpha=0.7)
        plt.show()
    else:
        print('Undefined Method')



    return labels

def get_dialog(dataset, N):
    dialogs = dataset['train'].select(range(N))  # Select the first 10 dialogues for example

    # Extract sentences from the dialogues
    dialogues = []
    emotions = []
    for dialogue in dialogs:
        for i in range(len(dialogue['dialog'])):
            # if i + 1 < len(dialogue['dialog']):
            #     sentence_pair = (dialogue['dialog'][i], dialogue['dialog'][i + 1])
            dialogues.append(dialogue['dialog'][i])
            emotions.append(dialogue['emotion'][i])


    return dialogues, emotions


def get_embeddings(sentences, tokenizer, model):
    inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt', max_length=512)

    # Move tensors to the same device as the model
    inputs = {key: value.to(model.device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    # Extract embeddings (mean pooling over tokens)
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings