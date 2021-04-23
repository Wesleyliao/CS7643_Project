import logging

from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

log = logging.getLogger(__name__)


def get_dataloader(
    path: str, batch_size: int = 32, random_flip: bool = False, image_size: int = None
) -> DataLoader:

    # Define transformations
    transforms_list = []

    if random_flip:
        transforms_list.append(transforms.RandomHorizontalFlip())

    if image_size:
        transforms_list.append(transforms.Resize(image_size))

    transforms_list.append(transforms.ToTensor())
    transforms_list.append(
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    )

    # Transform train data with augmentation
    transform = transforms.Compose(transforms_list)

    # Get dataset
    dataset = datasets.ImageFolder(path, transform=transform)

    # Train loader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    log.info(f'Created dataloader from {path}')

    return dataloader
