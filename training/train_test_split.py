from datasets import load_dataset
from PIL import Image
import os

# Load the original Marathi dataset (it has only train + validation)
dataset = load_dataset("processvenue/Marathi_Handwritten")

# Split train set: 90% train, 10% test
split_dataset = dataset["train"].train_test_split(test_size=0.1, seed=42)

# Create new dataset dict including validation
dataset = {
    "train": split_dataset["train"],
    "test": split_dataset["test"],
    "validation": dataset["validation"]
}

# Create output folders
os.makedirs("data", exist_ok=True)

for split in ["train", "validation", "test"]:
    csv_path = f"data/marathi_{split}.csv"
    dataset[split].to_csv(csv_path)
    print(f"✅ CSV saved: {csv_path}")

    img_dir = f"data/marathi_{split}_images"
    os.makedirs(img_dir, exist_ok=True)

    for i, example in enumerate(dataset[split]):
        image = example["image"]
        img = image.convert("RGB")  # Ensure compatibility with JPEG
        img_path = os.path.join(img_dir, f"{split}_{i}.jpg")
        img.save(img_path)

print("\n✅ Marathi dataset split and images saved successfully.")
