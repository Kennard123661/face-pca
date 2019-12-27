import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from model import TripletNetwork
from dataset import TrainDataset
from pca import get_transformed_data

N_OUTPUTS = 100
BATCH_SIZE = 1000
LEARNING_RATE = 1e-4
N_EPOCHS = 100


class CosineTripletLoss(nn.Module):
    def __init__(self, margin=0.1):
        super(CosineTripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchors, positives, negatives):
        pos_similarity = torch.cosine_similarity(anchors, positives)
        pos_distance = (torch.ones_like(pos_similarity) - pos_similarity).reshape(-1, 1)
        neg_similarity = torch.cosine_similarity(anchors, negatives)
        neg_distance = (torch.ones_like(neg_similarity) - neg_similarity).reshape(-1, 1)

        loss = torch.clamp_min(pos_distance - neg_distance + self.margin, min=0)
        loss = torch.mean(loss)
        return loss


def evaluate_epoch(model, train_data, train_labels, test_data, test_labels):
    model.eval()
    database_feats = model.predict(torch.from_numpy(train_data))
    query_feats = model.predict(torch.from_numpy(test_data))
    assert isinstance(model, nn.Module)
    predictions = list()

    for query_feat in query_feats:
        query_feat = query_feat.reshape(1, -1)
        database_sim = torch.cosine_similarity(query_feat, database_feats, dim=1).reshape(-1)
        nearest_idx = torch.argmax(database_sim).item()
        predictions.append(train_labels[nearest_idx])
    predictions = np.array(predictions)
    acc = np.mean((predictions - np.array(test_labels) == 0))
    return acc, predictions


def train(model, train_data, train_labels, test_data, test_labels):
    train_dataset = TrainDataset(train_data, train_labels)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                  collate_fn=train_dataset.collate_fn, drop_last=True)
    optimizer = optim.Adam(params=model.parameters(), lr=LEARNING_RATE)
    for i in range(N_EPOCHS):
        loss = train_epoch(model, train_dataloader, optimizer)
        print('iteration {0}: average loss: {1}', i, loss)
        if i != 0 and i % 5 == 0:
            acc, predictions = evaluate_epoch(model, train_data, train_labels, test_data, test_labels)
            print('iteration {0}: accuracy: {1}', i, acc)


def train_epoch(model, train_dataloader, optimizer):
    model.train()  # put model into training mode.
    triplet_loss = CosineTripletLoss()
    optimizer.zero_grad()
    average_loss = 0
    n_iter = 0
    for (anchors, positives, negatives) in train_dataloader:
        anchor_feats, pos_feats, neg_feats = model(anchors, positives, negatives)
        loss = triplet_loss(anchor_feats, pos_feats, neg_feats)
        n_iter += 1

        # compute gradinets for loss, then train
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        average_loss += loss.item()
    del triplet_loss
    return average_loss / n_iter


def main():
    train_data, train_labels, test_data, test_labels = get_transformed_data()
    n_inputs = train_data.shape[-1]
    model = TripletNetwork(n_inputs=n_inputs, n_outputs=100)
    train(model, train_data, train_labels, test_data, test_labels)


if __name__ == '__main__':
    main()
