from datasets import Dataset
from PIL import Image
import pandas as pd
import os

# Load existing train CSV
train_csv = pd.read_csv("data/gujarati_train.csv")

# Convert to Hugging Face Dataset
dataset = Dataset.from_pandas(train_csv)

# Split into validation and test (50-50% of 10%)
split = dataset.train_test_split(test_size=0.1, seed=42)
val_test_split = split["test"].train_test_split(test_size=0.5, seed=42)

# Combine all splits
dataset_dict = {
    "train": split["train"],
    "validation": val_test_split["train"],
    "test": val_test_split["test"]
}

# Save new CSVs
for split in ["validation", "test"]:
    csv_path = f"data/gujarati_{split}.csv"
    dataset_dict[split].to_csv(csv_path)
    print(f"✅ Saved: {csv_path}")

    # Save corresponding images
    img_dir = f"data/gujarati_{split}_images"
    os.makedirs(img_dir, exist_ok=True)

    for i, example in enumerate(dataset[split]):
        image = example["image"]               # ✅ Use image object, not "image_path"
        img = image.convert("RGB")             # ✅ Ensure JPEG-compatible
        img_path = os.path.join(img_dir, f"{split}_{i}.jpg")
        img.save(img_path)

