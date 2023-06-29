from torch.utils.data import Dataset, DataLoader
import lightning as L 
import torch

class FeatureDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class FeatureDataModule(L.LightningDataModule):
    def __init__(self, root_dir = './data/', dataset='swag' , batch_size=32, num_workers=1, pin_memory=False, **kwargs):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.root_dir = "data"
        self.dataset = dataset

    def setup(self, stage=None):
        
        # load training data
        self.train_features = torch.load(f"data/{self.dataset}/train_features.pt")
        self.train_labels = torch.load(f"data/{self.dataset}/train_labels.pt")

        # load validation data
        self.val_features = torch.load(f"data/{self.dataset}/valid_features.pt")
        self.val_labels = torch.load(f"data/{self.dataset}/valid_labels.pt")


    def _pin_memory(self):
        self.features = self.features.pin_memory()
        self.labels = self.labels.pin_memory()

    def train_dataloader(self):
        return DataLoader(
            FeatureDataset(self.train_features, self.train_labels),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
        )
    
    def val_dataloader(self):
        return DataLoader(
            FeatureDataset(self.val_features, self.val_labels),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )
    
    def test_dataloader(self):
        return DataLoader(
            FeatureDataset(self.features, self.labels),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )