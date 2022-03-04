# -*- coding: utf-8 -*-
import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset
import numpy as np
import pickle
import os


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


def split(train_set):
    train_n = int(0.7 * len(train_set))
    val_n = len(train_set) - train_n
    train_set, val_set = torch.utils.data.random_split(train_set, [train_n, val_n])
    return train_set,val_set
    


def data(batch_size, crop_size, misplacement=True,load=True,save=True):
    

    if misplacement:
        
        if load:
            train_set = torch.load(r"/Users/georgioszefkilis/Bayesian_Deep_Learning/src/data/misMNIST/processed/train_misMNIST.pt")
            val_set = torch.load(r"/Users/georgioszefkilis/Bayesian_Deep_Learning/src/data/misMNIST/processed/val_misMNIST.pt")
            test_set = torch.load(r"/Users/georgioszefkilis/Bayesian_Deep_Learning/src/data/misMNIST/processed/test_misMNIST.pt")
            
            
        else:

            train_set = MnistRandomPlacement(crop_size, 10, "train", True)
            test_set = MnistRandomPlacement(crop_size, 10, "test", True)
            
            train_set,val_set = split(train_set)
            
        
            if not os.path.exists(r"/Users/georgioszefkilis/Bayesian_Deep_Learning/src/data/misMNIST/processed"):
                os.makedirs(r"/Users/georgioszefkilis/Bayesian_Deep_Learning/src/data/misMNIST/processed")
                
            if save:
                torch.save(train_set,r"//Users/georgioszefkilis/Bayesian_Deep_Learning/src/data/misMNIST/processed/train_misMNIST.pt")
                torch.save(val_set,r"/Users/georgioszefkilis/Bayesian_Deep_Learning/src/data/misMNIST/processed/val_misMNIST.pt")
                torch.save(test_set,r"/Users/georgioszefkilis/Bayesian_Deep_Learning/src/data/misMNIST/processed/test_misMNIST.pt")
            

    else:
        # Training dataset
        print("Getting MNIST")

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
        
        train_set,val_set=split(train_set)

    

    
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=2
    )
    
    print("Train_data", len(train_loader.dataset))
    
    assert len(train_loader.dataset)>0,"Train loader is empty"

    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=batch_size, shuffle=True, num_workers=2
    )
    print("Validation_data", len(val_loader.dataset))

    assert len(val_loader.dataset)>0,"Validation loader is empty"


    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=True, num_workers=2
    )
    print("Test_data", len(test_loader.dataset))

    
    assert len(test_loader.dataset)>0,"Test loader is empty"


    return train_loader,val_loader, test_loader
