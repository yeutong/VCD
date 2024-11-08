from datasets import load_dataset
import os
from PIL import Image
import io

# Load the dataset
dataset = load_dataset("lmms-lab/POPE")

# Create a directory to save the images
image_folder = "../experiments/data/coco/val2014"
os.makedirs(image_folder, exist_ok=True)

# Function to save an image
def save_image(example):
    image = example['image']
    image_source = example['image_source']
    
    # Full path for saving the image
    save_path = os.path.join(image_folder, f'{image_source}.jpg')
    image.save(save_path)

# Iterate through each split and save images
for split_name in dataset.keys():
    print(f"Processing {split_name} split...")
    dataset[split_name].map(lambda example: save_image(example), num_proc=4)

print("All images have been saved.")