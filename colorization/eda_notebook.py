import marimo

__generated_with = "0.19.6"
app = marimo.App()


@app.cell
def _():
    import numpy as np
    import cv2
    import matplotlib.pyplot as plt
    return cv2, np


@app.cell
def _():
    import os
    import glob

    def load_images(path):
        # Find all JPGs recursively
        files = glob.glob(os.path.join(path, "**/*.jpg"), recursive=True)
        if not files:
            print("No images found! Check your path.")
            return []
        print(f"Found {len(files)} images.")
        return files

    dataset_path = "../landscape-pictures/"
    files = load_images(dataset_path)
    files
    return (files,)


@app.cell
def _(cv2, files, np):
    from tqdm import tqdm

    def create_pairs(file):
        img = cv2.imread(file)
        if img is None:
            print("Problem with ", file)
            return None
    
        # 1. Resize
        img = cv2.resize(img, (64, 64))
    
        # 2. Convert to LAB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    
        # 3. Normalize entire image to [-1, 1] range first
        # This handles L, a, and b all at once.
        # (0 -> -1.0, 128 -> 0.0, 255 -> 1.0)
        img_norm = (img.astype(np.float32) / 127.5) - 1.0
    
        # 4. Split and Reshape
        # X (Input): L channel only
        # Extract channel 0 and keep the dimension -> (64, 64, 1)
        X = img_norm[:, :, 0:1] 
    
        # y (Target): a, b channels
        # Extract channels 1 and 2 -> (64, 64, 2)
        y = img_norm[:, :, 1:3]
    
        return X, y

    X_list = []
    y_list = []

    for i in tqdm(range(0, len(files), 16), desc='Processing'):
        batch_files = files[i : i+16]
    
        # Process batch
        results = [create_pairs(f) for f in batch_files]
    
        # Filter Nones (failed reads)
        results = [r for r in results if r is not None]
    
        if not results: 
            print("Problem!!")
            continue

        # Unzip into separate lists
        batch_X, batch_y = zip(*results)
    
        X_list.extend(batch_X)
        y_list.extend(batch_y)

    # Final conversion to big numpy arrays
    X = np.array(X_list) # (N, 64, 64, 1)
    y = np.array(y_list) # (N, 64, 64, 2)

    np.savez("preprocess_data", X=X, y=y)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
