import numpy as np
from torch.utils.data._utils.collate import default_collate
from torch.utils.data import Dataset


class TrainDataset(Dataset):
    def __init__(self, data, labels):
        super(TrainDataset, self).__init__()
        self.x = data
        self.y = labels
        assert len(self.x) == len(self.y), 'this should have the same number of samples'
        assert len(self.x.shape) == 2, 'this should be vectorized.'
        self.positive_idxs, self.negative_idxs = self.get_positive_negative_idxs()

    def get_positive_negative_idxs(self):
        positive_idxs, negative_idxs = list(), list()
        for i, label in enumerate(self.y):
            # setup the positive idxs first
            other_label = self.y[:i]
            positive_idx = list(np.argwhere(other_label == label).reshape(-1))
            other_label = self.y[i+1:]
            positive_idx += list(np.argwhere(other_label == label).reshape(-1) + i + 1)
            negative_idx = np.setdiff1d(np.arange(len(self.y)), positive_idx + [i])
            assert len(negative_idx) + len(positive_idx) == len(self.y) - 1

            positive_idxs.append(positive_idx)
            negative_idxs.append(negative_idx)
        return positive_idxs, negative_idxs

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        idx = idx % len(self.x)
        anchor = self.x[idx]
        positive_idxs = self.positive_idxs[idx]
        negative_idxs = self.negative_idxs[idx]
        # print(np.array(positive_idxs).shape)
        positive_idx = np.random.choice(positive_idxs, 1).item()
        negative_idx = np.random.choice(negative_idxs, 1).item()

        positive = self.x[positive_idx]
        negative = self.x[negative_idx]
        return anchor, positive, negative

    @staticmethod
    def collate_fn(batch):
        anchors, positives, negatives = zip(*list(batch))
        anchors = default_collate(anchors)
        positives = default_collate(positives)
        negatives = default_collate(negatives)
        return anchors, positives, negatives
