import os
from PIL import Image

from torch.utils.data import Dataset
import torchvision.transforms as transforms

class ERKATAGIR(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.transform = transforms.Compose([
            # transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5)
        ])
        self.samples = []

        for fname in sorted(os.listdir(root_dir)):
            if not fname.lower().endswith('.png'):
                continue

            label_str = fname.split('_', 1)[0]
            try:
                label = label_str
            except ValueError:
                raise ValueError(f"Cannot parse label from filename '{fname}'")

            img_path = os.path.join(root_dir, fname)
            self.samples.append((img_path, label))

        if not self.samples:
            raise RuntimeError(f"No .png files found in '{root_dir}'")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('L')  # Grayscale

        if self.transform:
            image = self.transform(image)

        return {
            'target_image': image,
            'label': label
        }