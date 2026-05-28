import os
import requests
import pandas as pd
from pathlib import Path

print("Downloading HAM10000 metadata...")

metadata_url = "https://raw.githubusercontent.com/ieee8023/covid-chestxray-dataset/master/metadata.csv"

try:
    url = "https://raw.githubusercontent.com/ieee8023/covid-chestxray-dataset/master/metadata.csv"
    response = requests.get(url, timeout=10)
    print(f"Status: {response.status_code}")
except Exception as e:
    print(f"Could not download: {e}")

data_dir = Path("C:/Users/ACER/OneDrive/Desktop/Skin Disease Detection")

image_files = []
for folder in ["HAM10000_images_part_1", "HAM10000_images_part_2"]:
    folder_path = data_dir / folder
    if folder_path.exists():
        for f in folder_path.glob("*.jpg"):
            image_files.append(f.name)

print(f"Found {len(image_files)} images")

print("\nPlease download HAM10000_metadata.csv from:")
print("https://www.kaggle.com/kmader/skin-cancer-mnist-ham10000")
print("\nOr search for: HAM10000_metadata.csv online")
print("\nPlace it in: C:/Users/ACER/OneDrive/Desktop/Skin Disease Detection/")