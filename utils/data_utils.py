from typing import Any, Callable, Dict, Optional, Tuple
import os

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as datasets

from robustbench.model_zoo.enums import BenchmarkDataset
from robustbench.loaders import CustomImageFolder
from robustbench.data import PREPROCESSINGS


def _extract_cifar_data(
    dataset: Dataset, num_classes: int = 10, n_examples: Optional[int] = None
):
    assert num_classes in [10, 100], "num_classes must be 10 or 100."
    assert n_examples > 0, "n_examples must be positive."

    x_test_full = torch.tensor(dataset.data.transpose((0, 3, 1, 2)))
    y_test_full = torch.tensor(dataset.targets)
    x_test = x_test_full[:n_examples].float() / 255.
    y_test = y_test_full[:n_examples]

    return x_test, y_test


def load_cifar10(
    n_examples: Optional[int] = None,
    data_dir: str = './data',
    transforms_test: Callable = PREPROCESSINGS[None]
) -> Tuple[torch.Tensor, torch.Tensor]:

    dataset = datasets.CIFAR10(
        root=data_dir, train=False, transform=transforms_test, download=True
    )
    return _extract_cifar_data(dataset, n_examples=n_examples, num_classes=10)


def load_cifar100(
    n_examples: Optional[int] = None,
    data_dir: str = './data',
    transforms_test: Callable = PREPROCESSINGS[None]
) -> Tuple[torch.Tensor, torch.Tensor]:

    dataset = datasets.CIFAR100(
        root=data_dir, train=False, transform=transforms_test, download=True
    )
    return _extract_cifar_data(dataset, n_examples=n_examples, num_classes=100)


def load_imagenet(
    n_examples: Optional[int] = 5000,
    data_dir: str = './data',
    transforms_test: Callable = PREPROCESSINGS['Res256Crop224']
) -> Tuple[torch.Tensor, torch.Tensor]:

    assert n_examples <= 5000, \
        "The evaluation uses at most 5000 points for ImageNet."

    imagenet = CustomImageFolder(data_dir + '/val', transforms_test)
    test_loader = DataLoader(
        imagenet, batch_size=n_examples, shuffle=False, num_workers=4
    )

    x_test, y_test, paths = next(iter(test_loader))
    return x_test, y_test


def load_clean_dataset(
    dataset: BenchmarkDataset,
    n_examples: Optional[int],
    data_dir: str,
    prepr: Callable
) -> Tuple[torch.Tensor, torch.Tensor]:

    CleanDatasetLoader = Callable[
        [Optional[int], str, Callable], Tuple[torch.Tensor, torch.Tensor]
    ]
    _clean_dataset_loaders: Dict[BenchmarkDataset, CleanDatasetLoader] = {
        BenchmarkDataset.cifar_10: load_cifar10,
        BenchmarkDataset.cifar_100: load_cifar100,
        BenchmarkDataset.imagenet: load_imagenet,
    }
    return _clean_dataset_loaders[dataset](n_examples, data_dir, prepr)


class CIFAR10_float(datasets.CIFAR10):
    """ Based on the TorchVision CIFAR-10 dataset.
        This version allows the data to be float in addition to uint8.
        By doing so, this dataset can be compatible with attacked data.
    """
    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        super().__init__(root, train, transform, target_transform, download)
        self.data = self.data.astype(float) / 255.

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img, target = self.data[index], self.targets[index]

        if self.transform is not None:
            img = self.transform(img).float()
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class CIFAR100_float(datasets.CIFAR100):
    """ Based on the TorchVision CIFAR-100 dataset.
        This version allows the data to be float in addition to uint8.
        By doing so, this dataset can be compatible with attacked data.
    """
    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        super().__init__(root, train, transform, target_transform, download)
        self.data = self.data.astype(float) / 255.

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img, target = self.data[index], self.targets[index]

        if self.transform is not None:
            img = self.transform(img).float()
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class ImageNetValSubset(Dataset):
    def __init__(self, root, images_per_class=50, transform=None):
        self.image_paths = []
        self.labels = []
        self.transform = transform

        # Assuming root_dir is the path to the ImageNet validation folder
        for class_id, class_name in enumerate(
            sorted(os.listdir(os.path.join(root, 'val')))
        ):
            class_dir = os.path.join(root, 'val', class_name)
            if os.path.isdir(class_dir):
                images = sorted(os.listdir(class_dir))[:images_per_class]
                for img_name in images:
                    self.image_paths.append(os.path.join(class_dir, img_name))
                    self.labels.append(class_id)

        print(f"Loaded {self.__len__()} images from ImageNet validation set.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = datasets.folder.pil_loader(self.image_paths[idx])
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        return image, label


class ImageNet_float(Dataset):
    """ Float type ImageNet dataset used to load attacked images. Unlike
        the TorchVision implementation, this class pre-loads all images.
    """
    def __init__(
        self, images: torch.FloatTensor, targets: torch.LongTensor
    ) -> None:
        super().__init__()
        
        self.images, self.targets = images.cpu(), targets.cpu()
        # Check the dimensions of the images and targets
        (n, c, h, w), (_n,) = self.images.shape, self.targets.shape
        assert n == _n, "Number of images and targets must match."
        assert c == 3, "Number of channels must be 3."
        assert h == w == 224, "Image dimensions must be 224x224."
        assert targets.max() <= 1000, "Number of classes must be 1000."

    def __len__(self) -> int:
        return self.images.shape[0]

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        return self.images[index], self.targets[index]
