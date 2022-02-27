# -*- coding: utf-8 -*-
import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset
import numpy as np


class MnistRandomPlacement(Dataset):
    def __init__(self, crop_size, digits, mode, download=True):

        self.datasets = []
        self.cropsize = crop_size
        self.download = download
        self.digits = digits
        self.mode = mode

        # False (test) or True (train,val)
        trainingset = self.mode in ["train", "val"]

        self.datasets.append(
            datasets.MNIST(
                "src/data/MNIST/",
                transform=transforms.Compose([transforms.ToTensor()]),
                train=trainingset,
                download=self.download,
            )
        )

        # self.datasets.append(datasets.KMNIST(opt.dataroot,
        #                     transform=transforms.Compose([
        #                     transforms.ToTensor()]),
        #                     train=trainingset,
        #                     download=opt.download))

        self.num_images = self.digits

    def __len__(self):
        return min([self.datasets[0].__len__() for i in range(self.num_images)])

    def __getitem__(self, idx):

        im = torch.zeros((1, 96, 96), dtype=torch.float)

        used_positions, target = [], ""
        for i in range(self.num_images):
            while True:
                x = np.random.randint(0, 96 - 32)
                if len(used_positions) == 0 or abs(used_positions[0][0] - x) > 32:
                    break
            while True:
                y = np.random.randint(0, 96 - 32)
                if len(used_positions) == 0 or abs(used_positions[0][1] - y) > 32:
                    break

            im1, target1 = self.datasets[i].__getitem__(
                (idx) * (i + 1) % self.datasets[i].__len__()
            )

            c, w, h = im1.shape

            im[:, y : y + h, x : x + w] = im1.type(torch.float)
            # print('created image', im.shape, 'x:', x, 'y:', y)

            target += str(target1)

            transform = transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.Resize(self.cropsize),
                    transforms.ToTensor(),  # , transforms.RandomRotation(degrees=(0,180))
                    transforms.Normalize((0.1307,), (0.3081,)),
                ]
            )
            im = transform(im)

            return im, int(target)


def data(batch_size, crop_size, split=0.7, misplacement=True):

    if misplacement:

        train_set = MnistRandomPlacement(crop_size, 10, "train", True)
        test_set = MnistRandomPlacement(crop_size, 10, "test", True)

    else:
        # Training dataset

        train_set = datasets.MNIST(
            root="src/data/MNIST/",
            train=True,
            download=True,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            ),
        )

        # test set
        test_set = datasets.MNIST(
            root="src/data/MNIST/",
            train=False,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            ),
        )

    # split train,val,test

    train_n = int(split * len(train_set))
    val_n = len(train_set) - train_n
    train_set, val_set = torch.utils.data.random_split(train_set, [train_n, val_n])

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=2
    )

    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=batch_size, shuffle=True, num_workers=2
    )

    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=True, num_workers=2
    )

    return train_loader, val_loader, test_loader
