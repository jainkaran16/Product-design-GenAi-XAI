import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class ImageCaptionDataset(Dataset):
    def __init__(self, root_dir, image_size=256):
        self.samples = []
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
        ])

        for subfolder in os.listdir(root_dir):
            if subfolder.startswith('.') or subfolder == '__MACOSX':
                continue

            subfolder_path = os.path.join(root_dir, subfolder)
            if not os.path.isdir(subfolder_path):
                continue

            # Find any CSV file in the subfolder
            csv_files = [f for f in os.listdir(subfolder_path) if f.endswith(".csv")]
            if not csv_files:
                continue

            caption_file = os.path.join(subfolder_path, csv_files[0])

            try:
                df = pd.read_csv(caption_file)
                if 'image' not in df.columns or 'caption_groq' not in df.columns:
                    print(f"⚠️ Skipping {caption_file} — Required columns not found.")
                    continue
            except Exception as e:
                print(f"❌ Error reading {caption_file}: {e}")
                continue

            for _, row in df.iterrows():
                image_name = row.get("image")
                caption = row.get("caption_groq")
                image_path = os.path.join(subfolder_path, str(image_name))

                if image_name and caption and os.path.isfile(image_path):
                    self.samples.append((image_path, caption))

        print(f"✅ Total samples loaded: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, caption = self.samples[idx]
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        return image, caption
