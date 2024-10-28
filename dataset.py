from torch.utils.data import Dataset
import os
import pickle
import cv2
import numpy as np

class MyCifarDataset(Dataset):
    def __init__(self, data_path, is_train = True):
        if is_train:
            data_files = [os.path.join(data_path, "data_batch_{}".format(i)) for i in range(1, 6)]
        else:
            data_files = [os.path.join(data_path, "test_batch")]

        self.images = []
        self.labels = []

        for data_file in data_files:
            with open(data_file, "rb") as fo:
                data = pickle.load(fo, encoding="bytes")
                images = data[b"data"]
                labels = data[b"labels"]
                self.images.extend(images)
                self.labels.extend(labels)

    def __len__(self):
        return len(self.labels)
    def __getitem__(self, item):
        image = self.images[item]
        label = self.labels[item]
        return image, label

if __name__ == '__main__':
    dataset = MyCifarDataset(data_path="data/my_cifar10_datasets/cifar-10-batches-py", is_train=True)
    image, label = dataset[5500]
    image = np.reshape(image, (3, 32, 32))
    image = np.transpose(image, (1, 2, 0))
    image = cv2.resize(image, (320, 320))
    cv2.imshow("image", image)
    cv2.waitKey(0)

