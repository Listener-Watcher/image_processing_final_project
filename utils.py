from torchvision.transforms import transforms
from SimCLR.data_aug.gaussian_blur import GaussianBlur
from torchvision import transforms, datasets
#from exceptions.exceptions import InvalidDatasetSelection


def inject_jitter(prob):
    s = 1
    color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    data_transforms = transforms.Compose([  transforms.RandomApply([color_jitter], p=prob),
                                            # transforms.RandomGrayscale(p=0.2),
                                            # GaussianBlur(kernel_size=int(0.1 * size)),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=[.5], std=[.5])])
    return data_transforms

def inject_graynoise(prob):
    data_transforms = transforms.Compose([  #transforms.RandomGrayscale(p=prob),
                                            GaussianBlur(kernel_size=int(0.1 * 28)),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=[.5], std=[.5])])
    return data_transforms

def inject_randomflip(prob):
    data_transforms = transforms.Compose([  transforms.RandomHorizontalFlip(p=prob),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=[.5], std=[.5])])
    return data_transforms
