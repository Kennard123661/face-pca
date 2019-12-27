import torch
from model import TripletNetwork
from dataset import TrainDataLoader
from pca import get_transformed_data



def main():
    train_data, train_labels, test_data, test_labels = get_transformed_data()
    print(train_data.shape)
    raise NotImplementedError
