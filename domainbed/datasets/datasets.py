# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import os
import torch
from PIL import Image, ImageFile
from torchvision import transforms as T
from torch.utils.data import TensorDataset, Subset
from torchvision.datasets import MNIST, CIFAR10, ImageFolder
from torchvision.transforms.functional import rotate
import random

ImageFile.LOAD_TRUNCATED_IMAGES = True

DATASETS = [
    # Debug
    "Debug28",
    "Debug224",
    # Small images
    "ColoredMNIST",
    "RotatedMNIST",
    # Big images
    "VLCS",
    "PACS",
    
    "PACS_on_Mario"
    "PACS_on_VLCS"
    "PACS_on_DomainNet"

    "OfficeHome",
    "OfficeHome_on_Mario",
    "OfficeHome_on_VLCS",
    "OfficeHome_on_DomainNet",
    
    "TerraIncognita",
    
    "DomainNet",
    "DomainNet_on_Mario",
    "DomainNet_on_PACS",
    "DomainNet_on_OfficeHome",
    
    # Mine
    "SMD",

    
]


def get_dataset_class(dataset_name):
    """Return the dataset class with the given name."""
    if dataset_name not in globals():
        raise NotImplementedError("Dataset not found: {}".format(dataset_name))
    return globals()[dataset_name]

def get_class_indexes(imgs, size_per_class=-1):
    ret = dict()
    for idx in range(len(imgs)):
        _, class_idx = imgs[idx]
        if class_idx not in ret:
            ret[class_idx] = [idx]
        else:
            ret[class_idx].append(idx)
            
    return_idx = list()
    if size_per_class > -1:
        for class_idx in ret:
            if size_per_class < len(ret[class_idx]):
                ret[class_idx] = random.sample(ret[class_idx], size_per_class)
    
    for class_idx in ret:
        return_idx = return_idx + ret[class_idx] 
          
    return sorted(return_idx)        
            
    
        
        
        


def num_environments(dataset_name):
    return len(get_dataset_class(dataset_name).ENVIRONMENTS)


class MultipleDomainDataset:
    N_STEPS = 5001  # Default, subclasses may override
    CHECKPOINT_FREQ = 100  # Default, subclasses may override
    N_WORKERS = 4  # Default, subclasses may override
    ENVIRONMENTS = None  # Subclasses should override
    INPUT_SHAPE = None  # Subclasses should override

    def __getitem__(self, index):
        """
        Return: sub-dataset for specific domain
        """
        return self.datasets[index]

    def __len__(self):
        """
        Return: # of sub-datasets
        """
        return len(self.datasets)


class Debug(MultipleDomainDataset):
    def __init__(self, root):
        super().__init__()
        self.input_shape = self.INPUT_SHAPE
        self.num_classes = 2
        self.datasets = []
        for _ in [0, 1, 2]:
            self.datasets.append(
                TensorDataset(
                    torch.randn(16, *self.INPUT_SHAPE),
                    torch.randint(0, self.num_classes, (16,)),
                )
            )


class Debug28(Debug):
    INPUT_SHAPE = (3, 28, 28)
    ENVIRONMENTS = ["0", "1", "2"]


class Debug224(Debug):
    INPUT_SHAPE = (3, 224, 224)
    ENVIRONMENTS = ["0", "1", "2"]


class MultipleEnvironmentMNIST(MultipleDomainDataset):
    def __init__(self, root, environments, dataset_transform, input_shape, num_classes):
        """
        Args:
            root: root dir for saving MNIST dataset
            environments: env properties for each dataset
            dataset_transform: dataset generator function
        """
        super().__init__()
        if root is None:
            raise ValueError("Data directory not specified!")

        original_dataset_tr = MNIST(root, train=True, download=True)
        original_dataset_te = MNIST(root, train=False, download=True)

        original_images = torch.cat((original_dataset_tr.data, original_dataset_te.data))

        original_labels = torch.cat((original_dataset_tr.targets, original_dataset_te.targets))

        shuffle = torch.randperm(len(original_images))

        original_images = original_images[shuffle]
        original_labels = original_labels[shuffle]

        self.datasets = []
        self.environments = environments

        for i in range(len(environments)):
            images = original_images[i :: len(environments)]
            labels = original_labels[i :: len(environments)]
            self.datasets.append(dataset_transform(images, labels, environments[i]))

        self.input_shape = input_shape
        self.num_classes = num_classes
        

        

class ColoredMNIST(MultipleEnvironmentMNIST):
    ENVIRONMENTS = ["+90%", "+80%", "-90%"]

    def __init__(self, root):
        super(ColoredMNIST, self).__init__(
            root,
            [0.1, 0.2, 0.9],
            self.color_dataset,
            (2, 28, 28),
            2,
        )

    def color_dataset(self, images, labels, environment):
        # # Subsample 2x for computational convenience
        # images = images.reshape((-1, 28, 28))[:, ::2, ::2]
        # Assign a binary label based on the digit
        labels = (labels < 5).float()
        # Flip label with probability 0.25
        labels = self.torch_xor_(labels, self.torch_bernoulli_(0.25, len(labels)))

        # Assign a color based on the label; flip the color with probability e
        colors = self.torch_xor_(labels, self.torch_bernoulli_(environment, len(labels)))
        images = torch.stack([images, images], dim=1)
        # Apply the color to the image by zeroing out the other color channel
        images[torch.tensor(range(len(images))), (1 - colors).long(), :, :] *= 0

        x = images.float().div_(255.0)
        y = labels.view(-1).long()

        return TensorDataset(x, y)

    def torch_bernoulli_(self, p, size):
        return (torch.rand(size) < p).float()

    def torch_xor_(self, a, b):
        return (a - b).abs()


class RotatedMNIST(MultipleEnvironmentMNIST):
    ENVIRONMENTS = ["0", "15", "30", "45", "60", "75"]

    def __init__(self, root):
        super(RotatedMNIST, self).__init__(
            root,
            [0, 15, 30, 45, 60, 75],
            self.rotate_dataset,
            (1, 28, 28),
            10,
        )

    def rotate_dataset(self, images, labels, angle):
        rotation = T.Compose(
            [
                T.ToPILImage(),
                T.Lambda(lambda x: rotate(x, angle, fill=(0,), resample=Image.BICUBIC)),
                T.ToTensor(),
            ]
        )

        x = torch.zeros(len(images), 1, 28, 28)
        for i in range(len(images)):
            x[i] = rotation(images[i])

        y = labels.view(-1)

        return TensorDataset(x, y)


class MultipleEnvironmentImageFolder(MultipleDomainDataset):
    def __init__(self, root, downsample_size=None):
        super().__init__()
        environments = [f.name for f in os.scandir(root) if f.is_dir()]
        if len(environments) != self.ENVIRONMENTS:
            print('Using user specified environments instead of all subfolders:', self.ENVIRONMENTS)
            environments = self.ENVIRONMENTS
        
        environments = sorted(environments)
        self.environments = environments

        self.datasets = []
        for environment in environments:
            path = os.path.join(root, environment)
            env_dataset = ImageFolder(path)
            self.num_classes = len(env_dataset.classes)
            
            if downsample_size is None:
                self.datasets.append(env_dataset)
            else:
                env_downsize = downsample_size // len(environments) 
                size_per_class = env_downsize // len(env_dataset.classes)
                subset_indexes = get_class_indexes(env_dataset.imgs, size_per_class)
                env_subset = torch.utils.data.Subset(env_dataset, subset_indexes)
                self.datasets.append(env_subset)
                

        self.input_shape = (3, 224, 224)
        # self.num_classes = len(self.datasets[-1].classes)


class VLCS(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 200
    ENVIRONMENTS = ["Caltech101", "LabelMe", "SUN09", "VOC2007"]

    def __init__(self, root, downsample_size=None):
        self.dir = os.path.join(root, "VLCS_png/")
        super().__init__(self.dir, downsample_size)


class PACS(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 200
    ENVIRONMENTS = ['photo', 'art_painting', 'cartoon', 'sketch']

    def __init__(self, root, downsample_size=None):
        self.dir = os.path.join(root, 'PACS/')
        super().__init__(self.dir, downsample_size)


class PACS_on_Mario(PACS):
    N_STEPS = 10001
    def __init__(self, root, downsample_size=None):
        super().__init__(root, downsample_size)
        
class PACS_on_PACS(PACS):
    def __init__(self, root, downsample_size=None):
        super().__init__(root, downsample_size)
        
class PACS_on_Mario_SSDG(PACS):
    def __init__(self, root, downsample_size=None):
        super().__init__(root, downsample_size)
        
class PACS_on_VLCS(PACS):
    def __init__(self, root, downsample_size=None):
        super().__init__(root, downsample_size)
        
class PACS_on_VLCS_SSDG(PACS):
    def __init__(self, root, downsample_size=None):
        super().__init__(root, downsample_size)
        
class PACS_on_DomainNet(PACS):
    def __init__(self, root, downsample_size=None):
        super().__init__(root, downsample_size)
        
class PACS_on_DomainNet_SSDG(PACS):
    def __init__(self, root, downsample_size=None):
        super().__init__(root, downsample_size)

class PACSDithered(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 200
    ENVIRONMENTS = ['photo', 'art_painting', 'cartoon', 'sketch']

    def __init__(self, root):
        self.dir = os.path.join('/home/yluo97/scratch/PACS/PACS_dithered_1/')
        super().__init__(self.dir)


class DomainNet(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 500
    N_STEPS = 15001
    N_WORKERS = 1
    ENVIRONMENTS = ["clipart", "painting", "quickdraw", "real", "sketch", "infograph"]

    def __init__(self, root, downsample_size=None):
        self.dir = os.path.join(root, "domain_net/")
        super().__init__(self.dir, downsample_size)
        
class DomainNet_on_DomainNet(DomainNet):
    def __init__(self, root, downsample_size=None):
        super().__init__(root, downsample_size)
        
class DomainNet_on_Mario(DomainNet):
    def __init__(self, root, downsample_size=None):
        super().__init__(root, downsample_size)

class DomainNet_on_PACS(DomainNet):
    def __init__(self, root, downsample_size=None):
        super().__init__(root, downsample_size)

class DomainNet_on_OfficeHome(DomainNet):
    def __init__(self, root, downsample_size=None):
        super().__init__(root, downsample_size)
        
class DomainNet_on_Places365(DomainNet):
    def __init__(self, root, downsample_size=None):
        super().__init__(root, downsample_size)
        
class DomainNet_on_Cityscapes(DomainNet):
    def __init__(self, root, downsample_size=None):
        super().__init__(root, downsample_size)


class DomainNet_CPQR(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 1000
    N_STEPS = 12001
    ENVIRONMENTS = ["clipart", "painting", "quickdraw", "real"]

    def __init__(self, root, downsample_size=None):
        self.dir = os.path.join(root, "domain_net/")
        super().__init__(self.dir, downsample_size)


class OfficeHome(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 200
    ENVIRONMENTS = ["Art", "Clipart", "Product", "RealWorld"]

    def __init__(self, root, downsample_size=None):
        self.dir = os.path.join(root, "office_home/")
        super().__init__(self.dir)
        
class OfficeHome_on_Mario(OfficeHome):
    N_STEPS = 6001
    def __init__(self, root, downsample_size=None):
        super().__init__(root, downsample_size)
        
class OfficeHome_on_VLCS(OfficeHome):
    N_STEPS = 6001
    def __init__(self, root, downsample_size=None):
        super().__init__(root, downsample_size)
        
class OfficeHome_on_OfficeHome(OfficeHome):
    N_STEPS = 6001
    def __init__(self, root, downsample_size=None):
        super().__init__(root, downsample_size)
        
class OfficeHome_on_DomainNet(OfficeHome_on_Mario):
    N_STEPS = 6001
    def __init__(self, root, downsample_size=None):
        super().__init__(root, downsample_size)


class TerraIncognita(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 200
    ENVIRONMENTS = ["L100", "L38", "L43", "L46"]

    def __init__(self, root):
        self.dir = os.path.join(root, "terra_incognita/")
        super().__init__(self.dir)

class SMD(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = ["mario_nes", "mario_snes", "mario_n64", "mario_wii"]
    def __init__(self, root, downsample_size=None):
        self.dir = os.path.join(root, "smd/")
        super().__init__(self.dir, downsample_size)

class Mario(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = ["mario_nes", "mario_snes", "mario_n64", "mario_wii"]
    def __init__(self, root, downsample_size=None):
        self.dir = os.path.join(root, "smd/")
        super().__init__(self.dir, downsample_size)
        
class PACS_on_Places365(PACS):
    def __init__(self, root, downsample_size=None):
        super().__init__(root, downsample_size)
        
class PACS_on_Cityscapes(PACS):
    def __init__(self, root, downsample_size=None):
        super().__init__(root, downsample_size)
        
class OfficeHome_on_Places365(OfficeHome):
    def __init__(self, root, downsample_size=None):
        super().__init__(root, downsample_size)
        
class OfficeHome_on_Cityscapes(OfficeHome):
    def __init__(self, root, downsample_size=None):
        super().__init__(root, downsample_size)
        
class TerraIncognita_on_Cityscapes(TerraIncognita):
    def __init__(self, root, downsample_size=None):
        super().__init__(root, downsample_size)

class TerraIncognita_on_Places365(TerraIncognita):
    def __init__(self, root, downsample_size=None):
        super().__init__(root, downsample_size)
        
class TerraIncognita_on_Mario(TerraIncognita):
    def __init__(self, root, downsample_size=None):
        super().__init__(root, downsample_size)

class VLCS_on_Cityscapes(VLCS):
    def __init__(self, root, downsample_size=None):
        super().__init__(root, downsample_size)

class VLCS_on_Places365(VLCS):
    def __init__(self, root, downsample_size=None):
        super().__init__(root, downsample_size)
        
class VLCS_on_Mario(VLCS):
    def __init__(self, root, downsample_size=None):
        super().__init__(root, downsample_size)