import requests
import os
from tqdm import tqdm

# Tutachukua miezi 3 ya 2023 — inatosha (~ 1.5GB)
DATASETS = [
    "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-01.parquet",
    "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-02.parquet",
    "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-03.parquet",
]

RAW_DIR = "data/raw"

def download_file(url: str, dest_folder: str):
    filename = url.split("/")[-1]
    filepath = os.path.join(dest_folder, filename)

    if os.path.exists(filepath):
        print(f" Already exists: {filename}")
        return

    print(f"⬇  Downloading {filename}...")
    response = requests.get(url, stream=True)
    total = int(response.headers.get("content-length", 0))

    with open(filepath, "wb") as f, tqdm(
        total=total, unit="B", unit_scale=True, desc=filename
    ) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))

    print(f" Done: {filename}")

if __name__ == "__main__":
    os.makedirs(RAW_DIR, exist_ok=True)
    for url in DATASETS:
        download_file(url, RAW_DIR)
    print("\n All files downloaded to data/raw/")