import marimo

__generated_with = "0.19.6"
app = marimo.App()


@app.cell
def _():
    from model import UNet

    unet = UNet()
    unet = unet.load_weights("denoiser_epoch_2.safetensors")
    unet
    return (unet,)


@app.cell
def _(unet):
    import mlx.core as mx
    from PIL import Image
    import numpy as np
    import matplotlib.pyplot as plt

    img = Image.open("../landscape-pictures/00000002_(6).jpg").resize((64, 64))
    img_np = np.array(img).astype(np.float32) / 255.0
    arr = mx.array(img_np)
    arr = mx.expand_dims(arr, 0) # Add batch dim -> (1, 64, 64, 3)

    noisy_arr = arr + 0.3 * mx.random.normal(arr.shape)
    noisy_arr = mx.clip(noisy_arr, 0.0, 1.0)
    noisy_arr_np = np.array(noisy_arr)

    clean_img = unet(noisy_arr)
    mx.eval(clean_img)

    clean_img_np = np.array(clean_img)

    plt.figure(figsize=(16, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(img_np)
    plt.title("Original Clean")

    plt.subplot(1, 3, 2)
    plt.imshow(noisy_arr_np[0])
    plt.title("With Noise")

    plt.subplot(1, 3, 3)
    plt.imshow(clean_img_np[0])
    plt.title("Predicted by Model")

    plt.gcf()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
