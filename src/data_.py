import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms

class LensingDataset(Dataset):
    def __init__(self, npz_path, transform=None):
        """
        Arguments:
            npz_path : Path to the .npz file containing image data.
            transform : Optional transformations
        """
        data = np.load(npz_path)
        self.images = torch.tensor(data['images'], dtype=torch.float32)  # Shape: (N, 1, 150, 150)
        self.labels = torch.tensor(data['labels'], dtype=torch.long)  # Shape: (N,)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


def get_dataloaders(train_npz, val_npz, batch_size=200, test_split=0.2, num_workers=4):
    """
    Arguments:
        train_npz : Path to the training data .npz file.
        val_npz : Path to the validation data .npz file.
        batch_size : Batch size for DataLoaders.
        test_split : Fraction of validation set to use for testing.
    Returns:
        dictionary: {"train": train_loader, "val": val_loader, "test": test_loader}
    """

    # Training set transforms
    train_transform = transforms.Compose([
    transforms.RandomAffine(degrees=90, translate=(0.1, 0.1)),  # Rotation up to 90° + small translation
])
    #Turning Transformations off due to poor performance:
    # train_transform = transforms.Compose([
    #     transforms.RandomHorizontalFlip(),
    #     transforms.RandomVerticalFlip(),
    #     transforms.RandomRotation(20),
    # ])
    #transforms.Normalize(mean=[0.5], std=[0.5])
    # Augmentations for validation/test
    # test_transform = transforms.Normalize(mean=[0.5], std=[0.5])
    # val_transform = transforms.Normalize(mean=[0.5], std=[0.5])

    # Load datasets
    train_dataset = LensingDataset(train_npz, transform=train_transform)
    val_dataset = LensingDataset(val_npz)

    # Split validation dataset to create test set
    val_size = int(len(val_dataset) * (1 - test_split))
    test_size = len(val_dataset) - val_size
    val_dataset, test_dataset = random_split(val_dataset, [val_size, test_size])

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return {"train": train_loader, "val": val_loader, "test": test_loader}


