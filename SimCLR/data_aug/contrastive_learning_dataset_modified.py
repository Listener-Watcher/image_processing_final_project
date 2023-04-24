from torchvision.transforms import transforms
from data_aug.gaussian_blur import GaussianBlur
from torchvision import transforms, datasets
from data_aug.view_generator import ContrastiveLearningViewGenerator
from exceptions.exceptions import InvalidDatasetSelection
import medmnist
from medmnist import INFO, Evaluator
import torch.utils.data as data
class ContrastiveLearningDataset:
    def __init__(self, root_folder):
        self.root_folder = root_folder

    @staticmethod
    def get_simclr_pipeline_transform(size, s=1):
        """Return a set of data augmentation transformations as described in the SimCLR paper."""
        color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=size),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.RandomApply([color_jitter], p=0.8),
                                              transforms.RandomGrayscale(p=0.2),
                                              GaussianBlur(kernel_size=int(0.1 * size)),
                                              transforms.ToTensor()])
        return data_transforms

    def get_dataset(self, name, n_views,as_rgb=True):
        info = INFO[name]
        task = info['task']
        n_channels = 3 if as_rgb else info['n_channels']
        n_classes = len(info['label'])
        DataClass = getattr(medmnist, info['python_class'])
        data_transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize(mean=[.5], std=[.5])])
        train_dataset = DataClass(split='train', transform=ContrastiveLearningViewGenerator(self.get_simclr_pipeline_transform(28),n_views), download=True, as_rgb=as_rgb)
        # dataset = [train_dataset,val_dataset,test_dataset]
        return train_dataset,n_channels
