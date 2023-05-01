from torch.utils.data import Dataset
class SubsetDataset(Dataset):
    def __init__(self, subset, transform=None):
        # call me in debugging
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)
