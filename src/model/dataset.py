import torch
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import os

class TrafficDataset(Dataset):
    """
    Custom PyTorch Dataset for loading the 28x28 traffic "images".
    
    This class handles:
    1. Loading the .npy files.
    2. Normalizing pixel values from [0, 255] to [0, 1].
    3. Adding the required "channel" dimension.
    """
    def __init__(self, data_path, labels_path):
        # Load the entire dataset into memory.
        # This is fine for datasets up to a few GB.
        self.images = np.load(data_path)
        self.labels = np.load(labels_path)
        
        print(f"Dataset loaded. Found {len(self.labels)} samples.")

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.labels)

    def __getitem__(self, idx):
        """
        Fetches one sample (image and label) at the given index.
        This is where the transformation happens.
        """
        # 1. Get the data for one sample
        image = self.images[idx] # Shape: (28, 28), dtype: uint8
        label = self.labels[idx] # Shape: (), dtype: int
        
        # 2. Convert from NumPy array to PyTorch tensor
        # .float() converts from uint8 (0-255) to float32
        tensor = torch.from_numpy(image).float()
        
        # 3. Normalize the data to [0, 1] as specified in the paper
        #
        tensor = tensor / 255.0
        
        # 4. Add the channel dimension
        # The CNN (nn.Conv2d) expects [Channels, Height, Width]
        # This changes shape from (28, 28) -> (1, 28, 28)
        #
        tensor = tensor.unsqueeze(0) 
        
        # 5. Convert the label to a tensor
        # CrossEntropyLoss expects labels as Long (int64)
        label_tensor = torch.tensor(label).long()

        return tensor, label_tensor

def get_dataloaders(data_path, labels_path, batch_size=64, test_split=0.2):
    """
    A helper function to create and return train and test dataloaders.
    """
    # 1. Create the full dataset
    full_dataset = TrafficDataset(data_path, labels_path)
    
    # 2. Calculate split sizes
    total_size = len(full_dataset)
    test_size = int(total_size * test_split)
    train_size = total_size - test_size
    
    print(f"Splitting data: {train_size} train, {test_size} test")
    
    # 3. Split the dataset
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])
    
    # 4. Create the DataLoaders
    # The train loader shuffles data for every epoch
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4  # Speeds up loading (set to 0 if it causes errors)
    )
    
    # The test loader doesn't need to shuffle
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )
    
    return train_loader, test_loader

# --- This is how you would use it in your main training script ---
if __name__ == '__main__':
    # This block is for testing. You would put this logic
    # in your main `train.py` script.
    
    # Find the project root (assuming this script is in src/preprocessing)
    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)
    PROJECT_ROOT = os.path.dirname(os.path.dirname(script_dir))
    
    DATA_PATH = os.path.join(PROJECT_ROOT, "processed_test/idx/data.npy")
    LABELS_PATH = os.path.join(PROJECT_ROOT, "processed_test/idx/labels.npy")
    
    # 1. Get the data loaders
    train_loader, test_loader = get_dataloaders(
        DATA_PATH, 
        LABELS_PATH, 
        batch_size=64
    )
    
    # 2. Test the train loader by getting one batch
    print("\nTesting the train_loader:")
    try:
        images, labels = next(iter(train_loader))
        
        print(f"  Batch of images shape: {images.shape}")
        print(f"  Batch of labels shape: {labels.shape}")
        print(f"  Image tensor data type: {images.dtype}")
        print(f"  Label tensor data type: {labels.dtype}")
        print(f"  Image min value: {images.min()}")
        print(f"  Image max value: {images.max()}")
    
    except Exception as e:
        print(f"Error testing data loader: {e}")