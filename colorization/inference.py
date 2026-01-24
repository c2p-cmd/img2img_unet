import marimo

__generated_with = "0.19.6"
app = marimo.App()


@app.cell
def _():
    import mlx.core as mx
    from matplotlib import pyplot as plt
    import numpy as np
    from model import UNet

    unet = UNet()
    unet = unet.load_weights("colorizer_step_5000.safetensors")
    unet
    return mx, np, plt, unet


@app.cell
def _(mx, np, plt, unet):
    import cv2

    def load_and_prep_image(image_path, target_size=(64, 64)):
        """
        Loads an image, converts to Lab, and extracts/normalizes the L channel.
        """
        # 1. Load and Resize
        img = cv2.imread(image_path)
        if img is None:
            print("Problem")
            return None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_arr = cv2.resize(img, target_size)
    
        # 2. Convert to Lab
        lab_img = cv2.cvtColor(img_arr, cv2.COLOR_RGB2LAB)
    
        # 3. Extract L channel and Normalize to [-1, 1]
        # L is 0..100
        L = lab_img[:, :, 0]
        L_norm = (L - 50.0) / 50.0
    
        # 4. Add Batch and Channel dimensions -> (1, 64, 64, 1)
        L_input = L_norm[..., np.newaxis]
        L_input = np.expand_dims(L_input, axis=0)
    
        return mx.array(L_input), L, img_arr

    def post_process_image(L_channel, ab_pred):
        """
        Combines L channel with predicted ab channels and converts back to RGB.
        """
        # 1. Convert MLX array to Numpy
        ab = np.array(ab_pred[0]) # Remove batch dim -> (64, 64, 2)
    
        # 2. Denormalize ab from [-1, 1] to [-128, 128]
        a = ab[:, :, 0] * 127
        b = ab[:, :, 1] * 127
    
        # 3. Stack L and ab -> (64, 64, 3)
        # L_channel is (64, 64), need to stack depth-wise
        shape = L_channel.shape
        lab_out = np.dstack((L_channel, a, b))
    
        # 4. Convert Lab to RGB
        # Note: color.lab2rgb expects L in 0..100, a,b in -128..128
        rgb_out = cv2.cvtColor(lab_out, cv2.COLOR_LAB2RGB)
    
        return rgb_out


    # 2. Prep Data
    L_input, L_raw, orig = load_and_prep_image("../landscape-pictures/00000002_(6).jpg")

    # 3. Predict
    # Result is (1, 64, 64, 2) with values approx -1 to 1
    ab_pred = unet(L_input)

    # 4. Reconstruct
    color_img = post_process_image(L_raw, ab_pred)

    # 5. Save/Plot
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(orig)
    plt.title("Input (RGB)")

    plt.subplot(1, 3, 2)
    plt.imshow(L_raw, cmap='gray')
    plt.title("Input (Grayscale)")

    plt.subplot(1, 3, 3)
    plt.imshow(color_img)
    plt.title("Predicted Color")

    plt.gcf()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
