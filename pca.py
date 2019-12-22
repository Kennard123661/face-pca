import matplotlib.pyplot as plt
import numpy as np

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_lfw_people
from sklearn.decomposition import PCA

N_PCA_COMPONENTS = 100


def get_data():
    """ returns the data and labels. labels are already vectorized """
    lfw_dataset = fetch_lfw_people(min_faces_per_person=20)
    return lfw_dataset.data, lfw_dataset.target


def get_fitted_pca(train_data):
    pca = PCA(n_components=N_PCA_COMPONENTS, whiten=True)
    pca.fit(train_data)
    return pca


def get_transformed_data():
    data, labels = get_data()
    data /= 255.0
    train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.2)

    # return train_data, train_labels, test_data, test_labels
    fitted_pca = get_fitted_pca(train_data)
    transformed_train_data = fitted_pca.transform(train_data)
    transformed_test_data = fitted_pca.transform(test_data)
    return transformed_train_data, train_labels, transformed_test_data, test_labels


def compute_accuracy(predictions, labels):
    return np.mean((predictions - labels) == 0)


train_data, train_labels, test_data, test_labels = get_transformed_data()
classifier = MLPClassifier(hidden_layer_sizes=(), batch_size=256, verbose=True, early_stopping=True)
classifier.fit(train_data, train_labels)
print(train_data.shape)
predictions = classifier.predict(test_data)
accuracy = compute_accuracy(predictions, test_labels)
print(accuracy)



