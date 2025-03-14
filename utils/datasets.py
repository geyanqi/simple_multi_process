from torch.utils.data import Dataset
import torch


class InferenceDataset(Dataset):

    def __init__(self, input_paths):
        self.input_paths = input_paths
        self.input_tensors = torch.randn(len(input_paths), 3, 224, 224)

    def __len__(self):
        return len(self.input_paths)

    def __getitem__(self, idx):
        """Return the list of tasks you need to execute.
        """

        return self.input_tensors[idx], self.input_paths[idx]
