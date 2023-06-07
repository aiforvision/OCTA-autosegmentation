from torch.utils.data import Dataset
from monai.transforms import Compose
import random

class UnalignedZipDataset(Dataset):
    """
    Manages the dateset for the gan_ves_seg Task.
    It pairs synethic samples (real_A) with its corresponding label (real_A_seg),
    a random real sample (real_B) and a background noise image (background). 
    """
    def __init__(self, data: dict, transform: Compose, phase = "train", inference="S") -> None:
        super().__init__()
        A_paths = data.get("real_A") if phase == "train" or inference == "G" else None
        A_seg_paths = data.get("real_A_seg") if phase == "train" else None
        B_paths = data.get("real_B")  if phase == "train" or inference == "S" else None
        background = data.get("background") if phase == "train" or inference == "G" else None
        self.A_paths = A_paths
        self.B_paths = B_paths
        self.A_seg_paths = A_seg_paths
        self.transform = transform
        self.A_size = 0 if A_paths is None else len(A_paths)
        self.B_size = 0 if B_paths is None else len(B_paths)
        self.A_seg_size = 0 if A_seg_paths is None else len(A_seg_paths)
        self.background = background
        self.background_size = 0 if background is None else len(background)
        self.phase = phase

    def __len__(self) -> int:
        """
        Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of both sets
        """
        return max(self.A_size, self.B_size)

    def __getitem__(self, index) -> dict:
        data = dict()
        if self.A_paths is not None:
            A_path = self.A_paths[index % self.A_size]
            data["real_A_path"] = A_path
            data["real_A"] = A_path
        if self.B_paths is not None:
            if "real_A" in data:
                index_B = random.randint(0, self.B_size - 1)
            else:
                index_B = index
            B_path = self.B_paths[index_B]
            data["real_B_path"] = B_path
            data["real_B"] = B_path
        if self.A_seg_paths is not None:
            A_seg_path = self.A_seg_paths[index % self.A_size]
            data["real_A_seg_path"] = A_seg_path
            data["real_A_seg"] = A_seg_path
        if self.background is not None:
            data["background"] = self.background[random.randint(0, self.background_size - 1)]
        data_transformed = self.transform(data)
        return data_transformed

