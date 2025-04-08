# utils/dataset_loader.py
from torch.utils.data import Dataset
from PIL import Image, UnidentifiedImageError
import os
import pandas as pd
from collections import defaultdict
import random

class ImageCaptionDataset(Dataset):
    def __init__(self, root_dir, transform=None, use_caption=True):
        self.root_dir = root_dir
        self.transform = transform
        self.use_caption = use_caption
        self.samples = []
        self.image_to_captions = defaultdict(set)

        for subfolder in os.listdir(root_dir):
            subfolder_path = os.path.join(root_dir, subfolder)
            if os.path.isdir(subfolder_path):
                csv_files = [f for f in os.listdir(subfolder_path) if f.endswith('.csv')]
                for csv_file in csv_files:
                    csv_path = os.path.join(subfolder_path, csv_file)
                    try:
                        df = pd.read_csv(csv_path)
                        for _, row in df.iterrows():
                            image_rel_path = row.get('image')
                            caption = row.get('caption_groq', "")

                            if not image_rel_path:
                                continue

                            image_path = os.path.join(subfolder_path, image_rel_path)

                            if os.path.exists(image_path):
                                try:
                                    with Image.open(image_path) as img:
                                        img.verify()
                                    if not isinstance(caption, str):
                                        caption = str(caption) if caption is not None else ""
                                    self.image_to_captions[image_path].add(caption)
                                except (UnidentifiedImageError, Exception):
                                    pass
                    except Exception:
                        continue

        for image_path, captions in self.image_to_captions.items():
            selected_caption = random.choice(list(captions)) if captions else ""
            self.samples.append((image_path, selected_caption))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, caption = self.samples[idx]
        try:
            image = Image.open(image_path).convert("RGB")
        except:
            image = Image.new("RGB", (224, 224), (0, 0, 0))
            caption = ""

        if self.transform:
            image = self.transform(image)

        return image, caption
