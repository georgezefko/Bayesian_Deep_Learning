# -*- coding: utf-8 -*-
import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, Subset
import numpy as np
import os
from sklearn.model_selection import train_test_split
import h5py
import pandas as pd
from PIL import Image
import random
from functools import partial


import torch.nn.functional as F

from src.utils import AffineTransform
import math



class MnistRandomPlacement(Dataset):
    def __init__(self, crop_size, digits, mode, download=True,dataset="MNIST"):

        self.datasets = []
        self.cropsize = crop_size
        self.download = download
        self.digits = digits
        self.mode = mode

        # False (test) or True (train,val)
        trainingset = self.mode in ["train", "val"]

        if dataset == 'MNIST':
            self.datasets.append(
                datasets.MNIST(
                    "src/data/MNIST/",
                    transform=transforms.Compose([transforms.ToTensor()]),
                    train=trainingset,
                    download=self.download,
                )
            )

        else:
            self.datasets.append(datasets.KMNIST("src/data/KMNIST/",
                                transform=transforms.Compose([
                                transforms.ToTensor()]),
                                train=trainingset,
                                download=self.download))

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

class MnistRandomRotation(Dataset):
    # hardcode num_images=1 for now 
    def __init__(self, mode):
       
        
        self.mode = mode

        trainingset = self.mode in ["train", "val"]
        self.dataset = datasets.MNIST(
                    "src/data/MNIST/",
                    transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]),
                    train=trainingset,
                    download=True,
                )



    def __len__(self):
        return self.dataset.__len__()

    def __getitem__(self, idx):
        im, target = self.dataset.__getitem__(idx)

        # add rotation 'noise' in 20% of training cases and every second val/test image:
        if self.mode == 'train':
            add_noise = True # (torch.rand(size=[1]) > 0.8)
        else: 
            add_noise = True # (idx % 2 == 0)
        
        angle = torch.tensor([0.])
        if add_noise:
            angle = - torch.tensor(math.pi) + 2 * torch.tensor(math.pi) * torch.rand(size=[1])

            def transform_image_affine(x):
                random_params = torch.tensor([angle, 1., 1., 0, 0])
                theta = AffineTransform.make_affine_matrix(*random_params)
                x = x.unsqueeze(0)
                grid = F.affine_grid(theta, x.size(),align_corners=True)  # makes the flow field on a grid
                x_transformed = F.grid_sample(x, grid)  # interpolates x on the grid
                return x_transformed.squeeze(0)

            im = transform_image_affine(im)
        
        
        trafo = transforms.Normalize((0.1307,), (0.3081,))
        im = trafo(im)

        return im, int(target)#, angle # also return ground truth angle

class PatchCamelyon(Dataset):

    def __init__(self, path, mode='train', n_iters=None, augment=False):
        super().__init__()
        
        self.n_iters = n_iters

        assert mode in ['train', 'valid', 'test']
        base_name = "camelyonpatch_level_2_split_{}_{}.h5"

        print('\n')
        print("# " * 50)
        print('Loading {} dataset...'.format(mode))
        print('path is',os.path.join(path, base_name.format(mode, 'x')))
        # Open the files
        h5X = h5py.File(os.path.join(path, base_name.format(mode, 'x')), "r")
        h5y = h5py.File(os.path.join(path, base_name.format(mode, 'y')), "r")
        

        # Read into numpy array
        self.X = np.array(h5X.get('x'))
        self.y = np.array(h5y.get('y')).reshape(-1, 1).astype(np.float32)


        print('Loaded {} dataset with {} samples'.format(mode, len(self.X)))
        print("# " * 50)

        if augment:
            self.transform = transforms.Compose([transforms.ToPILImage(),
                                                 transforms.ColorJitter(brightness=.5, saturation=.25,
                                                                        hue=.1, contrast=.5),
                                                 transforms.RandomAffine(10, (0.05, 0.05), fillcolor=255),
                                                 transforms.RandomHorizontalFlip(.5),
                                                 transforms.RandomVerticalFlip(.5),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize([0.6716241, 0.48636872, 0.60884315],
                                                                      [0.27210504, 0.31001145, 0.2918652])])
        else:
            self.transform = transforms.Compose([transforms.ToPILImage(),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize([0.6716241, 0.48636872, 0.60884315],
                                                                      [0.27210504, 0.31001145, 0.2918652])])

    def __getitem__(self, item):
        images = self.X[item]
        images = self.transform(images)
        labels = self.y[item]
        return images, labels

    def _transform(self, images):
        tensors = []
        for image in images:
            tensors.append(self.transform(image))
        return torch.stack(tensors)

    def __len__(self):
        return len(self.y)


class GTSRB(Dataset):
    base_folder = r'/zhome/fc/5/104708/Desktop/Thesis/src/data/'

    def __init__(self, root_dir, train=False, transform=False):
        """
        Args:
            train (bool): Load trainingset or test set.
            root_dir (string): Directory containing GTSRB folder.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir

        self.sub_directory = 'GTSRB'# if train else 'GTSRB' #you can change subdrectories
        self.csv_file_name = 'Train_GTSRB_speed_limit_signs.csv' if train else 'Test_GTSRB_speed_limit_signs.csv'

        csv_file_path = os.path.join(
            root_dir, self.base_folder, self.sub_directory, self.csv_file_name)

        self.csv_data = pd.read_csv(csv_file_path)

        self.transform =  transforms.Compose([
                                                transforms.Resize((32, 32)),
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.3403, 0.3121, 0.3214),
                                                                    (0.2724, 0.2608, 0.2669))
                                            ])


    def __len__(self):
        return len(self.csv_data)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.base_folder, self.sub_directory,
                                self.csv_data.iloc[idx, 8]) #change the column of paths in csv file
        img = Image.open(img_path)

        classId = self.csv_data.iloc[idx, 7] #change the column of classes in csv
       
        if self.transform:
            img = self.transform(img)

        return img, classId

class GTSDB(Dataset):
    base_folder = r'/zhome/fc/5/104708/Desktop/Thesis/src/data/'

    def __init__(self, root_dir, train=False, transform=False):
        """
        Args:
            train (bool): Load trainingset or test set.
            root_dir (string): Directory containing GTSRB folder.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.directory = 'GTSDB'
        self.sub_directory = 'TrainIJCNN2013/TrainIJCNN2013' if train else 'TrainIJCNN2013/TrainIJCNN2013' #remember to change to test path
        self.csv_file_name = 'data_complete.csv' if train else 'data_complete.csv' #change to test path

        csv_file_path = os.path.join(
            root_dir, self.base_folder, self.directory,self.sub_directory, self.csv_file_name)

        self.csv_data = pd.read_csv(csv_file_path)

        self.transform =  transforms.Compose([
                                                transforms.Resize((128, 128)),
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.3403, 0.3121, 0.3214),
                                                                    (0.2724, 0.2608, 0.2669))
                                            ])


    def __len__(self):
        return len(self.csv_data)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.base_folder, self.directory,self.sub_directory,
                                self.csv_data.iloc[idx, 0]) #change the column of paths in csv file
        image = Image.open(img_path)

        image = np.array(image)

        x1, y1, x2, y2 = self.csv_data.iloc[idx, 1:5]
        #xmin, xmax, ymin, ymax = int(xmin), int(xmax), int(ymin), int(ymax)

        y_range = random.randrange(0,150)
        x_range = random.randrange(0,46)

        
        x1crop = x1 - x_range
        y1crop = y1 - y_range
        x2crop = x2 + y_range
        y2crop = y2 + y_range
  
  
        roi = image[max(0, y1crop):y2crop, 
                      max(0, x1crop):x2crop, :]


        image = Image.fromarray(roi)

        classId = self.csv_data.iloc[idx, 5] #change the column of classes in csv

        if self.transform:
            img = self.transform(image)

        return img, classId

class MTSD(torch.utils.data.Dataset):
    def __init__(self,dataroot, bbox_size,max_dataset_size, mode,data_augmentation=False):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        self.bbox_size = bbox_size

        if mode in ['train']:
            transform = [transforms.Resize((128, 128))]
            if data_augmentation:
                transform.append(transforms.RandomHorizontalFlip(0.5))

            transform.extend([transforms.ToTensor(), normalize])

            self.transform = transforms.Compose(transform)
            
        elif mode in ['test','val']:
            self.transform = transforms.Compose(
                [transforms.Resize((128, 128)), transforms.ToTensor(), normalize])

        self.transform_high_res = transforms.Compose([transforms.Resize((1024, 1024)), transforms.ToTensor(), normalize])

        self.root = dataroot
        self.base_folder = 'trafic'

        self.fn = partial(os.path.join, self.root, self.base_folder)

        if mode == "train":
            self.data = pd.read_csv(self.fn("train_trafic_data.csv"))
        else:
            self.data = pd.read_csv(self.fn("test_trafic_data.csv"))
        self.data = self.data.values

        # make sure we only look at the first max_dataset_size images.
        if mode == 'train':
            size = min(max_dataset_size, len(self.data))
            self.data = self.data[:size]

        self.mapper = {
            'information--pedestrians-crossing--g1': 0,
            'regulatory--no-entry--g1': 1,
            'complementary--chevron-left--g1': 2,
            'complementary--chevron-right--g1': 3,
            'regulatory--yield--g1': 4,
            'regulatory--stop--g1': 5,
            'regulatory--keep-right--g1': 6,
            'information--parking--g1': 7,
            'warning--pedestrians-crossing--g4': 8,
            'regulatory--no-overtaking--g5': 9
        }


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        
        image = Image.open(self.fn("images", self.data[idx, 1].replace(".json", ".jpg")))
        image = np.array(image)

        xmin, xmax, ymin, ymax = self.data[idx, 3:]
        xmin, xmax, ymin, ymax = int(xmin), int(xmax), int(ymin), int(ymax)

        
        y_range = int(self.bbox_size * 150)
        x_range = int(self.bbox_size * 150)
        
        
        image = image[max(0, ymin - y_range):(ymax + y_range), 
                      max(0, xmin - x_range):(xmax + x_range), :]

        
        image = Image.fromarray(image)

        target = self.mapper[self.data[idx, 2]]

        image_high_res = self.transform_high_res(image)
        image = self.transform(image)

        return image, target

def split(train_set):
    
    train_n = int(0.7 * len(train_set))
    val_n = len(train_set) - train_n
    train_set, val_set = torch.utils.data.random_split(train_set, [train_n, val_n])
    return train_set,val_set

def loaders(train_set,val_set,test_set,batch_size):

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=2
    )
    
    print("Train_data", len(train_loader.dataset))
    
    assert len(train_loader.dataset)>0,"Train loader is empty"

    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=batch_size,shuffle=False, num_workers=2
    )
    print("Validation_data", len(val_loader.dataset))

    assert len(val_loader.dataset)>0,"Validation loader is empty"


    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=2
    )
    print("Test_data", len(test_loader.dataset))

    
    assert len(test_loader.dataset)>0,"Test loader is empty"


    return train_loader,val_loader, test_loader


def get_subset(train_set,train_subset):

    train_n = int(10 * train_subset)
    val_n = len(train_set) - train_n
    train_set, val_set = torch.utils.data.random_split(train_set, [train_n, val_n])
    

    indices = np.arange(len(val_set))
    val_indices,_ = train_test_split(indices, train_size=18000) #stratify=torch.tensor([train_set[i][1] for i,j in enumerate(train_set)]))

    # Warp into Subsets and DataLoaders
    #train_set = Subset(train_set, train_indices)
    val_set = Subset(val_set,val_indices)
    #train_set,val_set = split(train_set)

    return train_set,val_set


def data(batch_size, crop_size,train_subset,dataset, misplacement=False,rotation=False,load=True,save=False,subset=False):
    
    if dataset=='PCAM':

        # Training dataset
        print("Getting PCAM")
        path = r"/zhome/fc/5/104708/Desktop/Thesis/src/data/PCAM/"
        dataset_train = PatchCamelyon(path, mode='train', augment=False)
        dataset_valid = PatchCamelyon(path, mode='valid', )
        dataset_test = PatchCamelyon(path, mode='test', )
        
        train_loader,val_loader,test_loader = loaders(dataset_train,dataset_valid,dataset_test,batch_size)
    
    elif dataset=='GTSRB':
        print("Getting GTSRB")

        # Create Datasets
        trainset = GTSRB(
            root_dir='./data', train=True,  transform=True)

        dataset_test = GTSRB(
            root_dir='./data', train=False,  transform=True)


        # Divide data into training and validation (0.7 and 0.3)
        dataset_train,dataset_valid = split(trainset)
        train_loader,val_loader,test_loader = loaders(dataset_train,dataset_valid,dataset_test,batch_size)
    
    elif dataset =='GTSDB':
        print('Getting GTSDB')

         # Create Datasets
         
        trainset = GTSDB(
            root_dir='./data', train=True,  transform=True)

        dataset_test = GTSDB(
            root_dir='./data', train=False,  transform=True)


        # Divide data into training and validation (0.7 and 0.3)
        dataset_train,dataset_valid = split(trainset)
        train_loader,val_loader,test_loader = loaders(dataset_train,dataset_valid,dataset_test,batch_size)
    
    elif dataset == 'Mapiliary':


        print('Getting MApiliary')

         # Create Datasets
         
        trainset = MTSD(dataroot='/zhome/fc/5/104708/Desktop/Thesis/src/data/Mapiliary/',bbox_size=4,max_dataset_size=5000,mode='train')

        dataset_test = MTSD(dataroot='/zhome/fc/5/104708/Desktop/Thesis/src/data/Mapiliary/',bbox_size=4,max_dataset_size=500,mode='test')


        # Divide data into training and validation (0.7 and 0.3)
        dataset_train,dataset_valid = split(trainset)
        train_loader,val_loader,test_loader = loaders(dataset_train,dataset_valid,dataset_test,batch_size)

    else:
        if misplacement:
            # Training dataset
            print("Getting misMNIST")
            if load:
                if dataset =='MNIST':
                    train_set = torch.load(r"/zhome/fc/5/104708/Desktop/Thesis/src/data/misMNIST/processed/train_misMNIST.pt")
                    val_set = torch.load(r"/zhome/fc/5/104708/Desktop/Thesis/src/data/misMNIST/processed/val_misMNIST.pt")
                    test_set = torch.load(r"/zhome/fc/5/104708/Desktop/Thesis/src/data/misMNIST/processed/test_misMNIST.pt")
                else:
                    train_set = torch.load(r"/zhome/fc/5/104708/Desktop/Thesis/src/data/misKMNIST/processed/train_misKMNIST.pt")
                    val_set = torch.load(r"/zhome/fc/5/104708/Desktop/Thesis/src/data/misKMNIST/processed/val_misKMNIST.pt")
                    test_set = torch.load(r"/zhome/fc/5/104708/Desktop/Thesis/src/data/misKMNIST/processed/test_misKMNIST.pt")

                
                if subset:
                    train_set,val_set= get_subset(train_set,train_subset)

                train_loader,val_loader,test_loader = loaders(train_set,val_set,test_set,batch_size)
                
            else:

                train_set = MnistRandomPlacement(crop_size, 10, "train", True, dataset = dataset)
                test_set = MnistRandomPlacement(crop_size, 10, "test", True, dataset = dataset)
                
                train_set,val_set = split(train_set)
                
                if dataset == 'MNIST':
                    
                    if not os.path.exists(r"/zhome/fc/5/104708/Desktop/Thesis/src/data/misMNIST/processed"):
                        os.makedirs(r"/zhome/fc/5/104708/Desktop/Thesis/src/data/misMNIST/processed")
                        
                    if save:
                        torch.save(train_set,r"/zhome/fc/5/104708/Desktop/Thesis/src/data/misMNIST/processed/train_misMNIST.pt")
                        torch.save(val_set,r"/zhome/fc/5/104708/Desktop/Thesis/src/data/misMNIST/processed/val_misMNIST.pt")
                        torch.save(test_set,r"/zhome/fc/5/104708/Desktop/Thesis/src/data/misMNIST/processed/test_misMNIST.pt")
                else:
                    
                    if not os.path.exists(r"/zhome/fc/5/104708/Desktop/Thesis/src/data/misKMNIST/processed"):
                        os.makedirs(r"/zhome/fc/5/104708/Desktop/Thesis/src/data/misKMNIST/processed")
                        
                    if save:
                        torch.save(train_set,r"/zhome/fc/5/104708/Desktop/Thesis/src/data/misKMNIST/processed/train_misKMNIST.pt")
                        torch.save(val_set,r"/zhome/fc/5/104708/Desktop/Thesis/src/data/misKMNIST/processed/val_misKMNIST.pt")
                        torch.save(test_set,r"/zhome/fc/5/104708/Desktop/Thesis/src/data/misKMNIST/processed/test_misKMNIST.pt")

                train_loader,val_loader,test_loader = loaders(train_set,val_set,test_set,batch_size)

        elif rotation:
            # Training dataset
            print("Getting rotMNIST")
            if load:
                if dataset =='MNIST':
                    train_set = torch.load(r"/zhome/fc/5/104708/Desktop/Thesis/src/data/rotMNIST/processed/train_rotMNIST.pt")
                    val_set = torch.load(r"/zhome/fc/5/104708/Desktop/Thesis/src/data/rotMNIST/processed/val_rotMNIST.pt")
                    test_set = torch.load(r"/zhome/fc/5/104708/Desktop/Thesis/src/data/rotMNIST/processed/test_rotMNIST.pt")
                else:
                    train_set = torch.load(r"/zhome/fc/5/104708/Desktop/Thesis/src/data/misKMNIST/processed/train_misKMNIST.pt")
                    val_set = torch.load(r"/zhome/fc/5/104708/Desktop/Thesis/src/data/misKMNIST/processed/val_misKMNIST.pt")
                    test_set = torch.load(r"/zhome/fc/5/104708/Desktop/Thesis/src/data/misKMNIST/processed/test_misKMNIST.pt")

                
                if subset:
                    train_set,val_set= get_subset(train_set,train_subset)

                train_loader,val_loader,test_loader = loaders(train_set,val_set,test_set,batch_size)
                
            else:
                print('pass save')
                train_set = MnistRandomRotation("train")
                test_set =  MnistRandomRotation("test")
                
                train_set,val_set = split(train_set)
                
                if dataset == 'MNIST':
                    
                    if not os.path.exists(r"/zhome/fc/5/104708/Desktop/Thesis/src/data/rotMNIST/processed"):
                        os.makedirs(r"/zhome/fc/5/104708/Desktop/Thesis/src/data/rotMNIST/processed")
                        
                    if save:
                        torch.save(train_set,r"/zhome/fc/5/104708/Desktop/Thesis/src/data/rotMNIST/processed/train_rotMNIST.pt")
                        torch.save(val_set,r"/zhome/fc/5/104708/Desktop/Thesis/src/data/rotMNIST/processed/val_rotMNIST.pt")
                        torch.save(test_set,r"/zhome/fc/5/104708/Desktop/Thesis/src/data/rotMNIST/processed/test_rotMNIST.pt")
                else:
                    
                    if not os.path.exists(r"/zhome/fc/5/104708/Desktop/Thesis/src/data/misKMNIST/processed"):
                        os.makedirs(r"/zhome/fc/5/104708/Desktop/Thesis/src/data/misKMNIST/processed")
                        
                    if save:
                        torch.save(train_set,r"/zhome/fc/5/104708/Desktop/Thesis/src/data/misKMNIST/processed/train_misKMNIST.pt")
                        torch.save(val_set,r"/zhome/fc/5/104708/Desktop/Thesis/src/data/misKMNIST/processed/val_misKMNIST.pt")
                        torch.save(test_set,r"/zhome/fc/5/104708/Desktop/Thesis/src/data/misKMNIST/processed/test_misKMNIST.pt")

                train_loader,val_loader,test_loader = loaders(train_set,val_set,test_set,batch_size)


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
                    [transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))]
                ),
            )


            if subset:
                train_set,val_set= get_subset(train_set,train_subset)
                train_loader,val_loader,test_loader = loaders(train_set,val_set,test_set,batch_size)
            else:
                train_set,val_set=split(train_set)
                train_loader,val_loader,test_loader = loaders(train_set,val_set,test_set,batch_size)
    
    return train_loader,val_loader,test_loader

    
    

    
    
