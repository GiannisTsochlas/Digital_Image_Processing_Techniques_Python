#Tsochlas Ioannis undergraduate student of cs department of ioannina university
import os

import numpy as np
import scipy.ndimage

from common import read_img, save_img
from PIL import Image
import matplotlib.pyplot as plt


def image_patches(image, patch_size=(16, 16)):
    """
    --- Zitima 1.a ---
    Given an input image and patch_size,
    return the corresponding image patches made
    by dividing up the image into patch_size sections.

    Input- image: H x W
           patch_size: a scalar tuple M, N
    Output- results: a list of images of size M x N
    """

    output = []
    img_height, img_width = image.shape
    patch_height, patch_width = patch_size

    # Iterate image to extract patches
    for i in range(0, img_height, patch_height):
        for j in range(0, img_width, patch_width):
            patch = image[i:i + patch_height, j:j + patch_width]  # TODO: Use slicing to complete the function
            # patch of the correct size
            if patch.shape == (patch_height, patch_width):
                # Normalize the patch
                patch_mean = np.mean(patch)
                patch_std = np.std(patch)
                normalized_patch = (patch - patch_mean) / patch_std
                output.append(normalized_patch)
    return output


def convolve(image, kernel):
    """
    --- Zitima 2.b ---
    Return the convolution result: image * kernel.
    Reminder to implement convolution and not cross-correlation!
    Caution: Please use zero-padding.

    Input- image: H x W
           kernel: h x w
    Output- convolve: H x W
    """

    # Check if kernel is 1D
    if kernel.ndim == 1:
        kernel = kernel[:, np.newaxis]

    # revert kenel
    kernel = np.flip(kernel)

    # Get image dimensions
    img_height, img_width = image.shape
    kernel_height, kernel_width = kernel.shape

    # Pad the image with zeros
    pad_height = kernel_height // 2
    pad_width = kernel_width // 2
    padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant', constant_values=0)

    # Prepare the output array
    output = np.zeros_like(image)

    # Perform convolution
    for i in range(img_height):
        for j in range(img_width):
            output[i, j] = np.sum(padded_image[i:i + kernel_height, j:j + kernel_width] * kernel)

    return output


# Set the Gaussian filter
def gaussian_kernel(size, sigma=1):
    """Generates a 2D Gaussian kernel."""
    kernel_1D = np.linspace(-(size // 2), size // 2, size)

    for i in range(size):
        kernel_1D[i] = np.exp(-(kernel_1D[i] ** 2) / (2 * sigma ** 2))

    kernel_2D = np.outer(kernel_1D.T, kernel_1D.T)

    kernel_2D *= 1.0 / kernel_2D.max()

    return kernel_2D


# Μη κανονικοποιημένο Gaussian φίλτρο
def unnormalized_gaussian_kernel(size, sigma):
    """Generates an unnormalized 2D Gaussian kernel."""
    kernel_1D = np.linspace(-(size // 2), size // 2, size)
    for i in range(size):
        kernel_1D[i] = np.exp(-(kernel_1D[i] ** 2) / (2 * sigma ** 2))
    kernel_2D = np.outer(kernel_1D, kernel_1D)
    return kernel_2D


def edge_detection(image):
    """
    --- Zitima 2.f ---
    Return Ix, Iy and the gradient magnitude of the input image

    Input- image: H x W
    Output- Ix, Iy, grad_magnitude: H x W
    """
    # TODO: Fix kx, ky
    kx = np.array([-1, 0, 1]).reshape(1, 3)  # 1x3
    ky = np.array([-1, 0, 1]).reshape(3, 1)  # 3x1

    Ix = convolve(image, kx)
    Iy = convolve(image, ky)

    # TODO: Use Ix, Iy to calculate grad_magnitude
    grad_magnitude = np.sqrt(Ix ** 2 + Iy ** 2)

    return Ix, Iy, grad_magnitude


def sobel_operator(image):
    """
    --- Zitima 3.b ---
    Return Gx, Gy, and the gradient magnitude.

    Input- image: H x W
    Output- Gx, Gy, grad_magnitude: H x W
    """
    # TODO: Use convolve() to complete the function
    # Ορισμός των πυρήνων Sobel
    S_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    S_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    # Υπολογισμός των παραγώγων της εικόνας
    Gx = convolve(image, S_x)
    Gy = convolve(image, S_y)

    grad_magnitude = np.sqrt(Gx**2 + Gy**2)

    return Gx, Gy, grad_magnitude


def main():
    # The main function
    img = read_img('./grace_hopper.png')
    """ Image Patches """
    if not os.path.exists("./image_patches"):
        os.makedirs("./image_patches")

    # -- TODO Zitima 1: Image Patches --
    # (a)
    # First complete image_patches()
    patches = image_patches(img)
    # Now choose any three patches and save them
    # chosen_patches should have those patches stacked vertically/horizontally
    chosen_patches = np.vstack(patches[:3])
    save_img(chosen_patches, "./image_patches/q1_patch.png")

    # (b), (c): No code

    """ Convolution and Gaussian Filter """
    if not os.path.exists("./gaussian_filter"):
        os.makedirs("./gaussian_filter")

    # -- TODO Zitima 2: Convolution and Gaussian Filter --
    # (a): No code

    # (b): Complete convolve()

    # (c)
    # Calculate the Gaussian kernel described in the question 2.(c).
    # There is tolerance for the kernel.

    kernel_size = 3
    sigma = 0.572
    kernel_gaussian = gaussian_kernel(kernel_size, sigma)
    #test the convolve function with filtered_gaussian = scipy.ndimage.convolve(img, kernel_gaussian)
    filtered_gaussian = convolve(img, kernel_gaussian)
    save_img(filtered_gaussian, "./gaussian_filter/q2_gaussian.png")

    # plot the original and filtered images
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Αρχική Εικόνα")
    plt.imshow(img, cmap='gray')
    plt.subplot(1, 2, 2)
    plt.title("Φιλτραρισμένη Εικόνα")
    plt.imshow(filtered_gaussian, cmap='gray')
    plt.show()

    # (d)

    # Υπολογισμός των εντάσεων της φιλτραρισμένης εικόνας
    min_intensity = filtered_gaussian.min()
    max_intensity = filtered_gaussian.max()
    intensity_range = max_intensity - min_intensity

    print(f"Min Intensity: {min_intensity}")
    print(f"Max Intensity: {max_intensity}")
    print(f"Intensity Range: {intensity_range}")

    unnormalized_gaussian_filter = unnormalized_gaussian_kernel(3, 2)

    # Υπολογισμός αθροίσματος των συντελεστών του μη κανονικοποιημένου φίλτρου
    sum_of_weights = unnormalized_gaussian_filter.sum()
    print(f"Sum of Gaussian filter weights (Unnormalized): {sum_of_weights}")

    # Εφαρμογή του μη κανονικοποιημένου φίλτρου
    filtered_image_unnormalized = convolve(img, unnormalized_gaussian_filter)

    # Υπολογισμός των εντάσεων της φιλτραρισμένης εικόνας
    min_intensity_unnormalized = filtered_image_unnormalized.min()
    max_intensity_unnormalized = filtered_image_unnormalized.max()
    intensity_range_unnormalized = max_intensity_unnormalized - min_intensity_unnormalized

    print(f"Min Intensity (Unnormalized): {min_intensity_unnormalized}")
    print(f"Max Intensity (Unnormalized): {max_intensity_unnormalized}")
    print(f"Intensity Range (Unnormalized): {intensity_range_unnormalized}")

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 3, 1)
    plt.title("Αρχική Εικόνα")
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    plt.subplot(1, 3, 2)
    plt.title("Κανονικοποιημένο Φίλτρο")
    plt.imshow(filtered_gaussian, cmap='gray', vmin=0, vmax=255)
    plt.subplot(1, 3, 3)
    plt.title("Μη Κανονικοποιημένο Φίλτρο")
    plt.imshow(filtered_image_unnormalized, cmap='gray', vmin=0, vmax=255)
    plt.tight_layout()
    plt.show()

    # (f): Complete edge_detection()

    # (g)
    # Use edge_detection() to detect edges
    # for the orignal and gaussian filtered images.
    _, _, edge_detect = edge_detection(img)
    save_img(edge_detect, "./gaussian_filter/q3_edge.png")

    _, _, edge_with_gaussian = edge_detection(filtered_gaussian)
    save_img(edge_with_gaussian, "./gaussian_filter/q3_edge_gaussian.png")

    # Plot the gradient magnitude of the original and filtered images
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.title("Gradient Magnitude - Original Image")
    plt.imshow(edge_detect, cmap='gray')
    plt.colorbar()

    plt.subplot(1, 2, 2)
    plt.title("Gradient Magnitude - Gaussian Filtered Image")
    plt.imshow(edge_with_gaussian, cmap='gray')
    plt.colorbar()

    plt.tight_layout()
    plt.show()

    print("Gaussian Filter is done. ")

    # -- TODO Zitima 3: Sobel Operator --
    if not os.path.exists("./sobel_operator"):
        os.makedirs("./sobel_operator")

    # (a): No code

    # (b): Complete sobel_operator()

    # (c)
    Gx, Gy, edge_sobel = sobel_operator(img)
    save_img(Gx, "./sobel_operator/q2_Gx.png")
    save_img(Gy, "./sobel_operator/q2_Gy.png")
    save_img(edge_sobel, "./sobel_operator/q2_edge_sobel.png")

    plt.figure(figsize=(18, 6))

    plt.subplot(1, 3, 1)
    plt.title("(Gradient in x direction)")
    plt.imshow(Gx, cmap='gray')
    plt.colorbar()

    plt.subplot(1, 3, 2)
    plt.title("(Gradient in y direction)")
    plt.imshow(Gy, cmap='gray')
    plt.colorbar()

    plt.subplot(1, 3, 3)
    plt.title("Gradient Magnitude - Sobel Operator")
    plt.imshow(edge_sobel, cmap='gray')
    plt.colorbar()

    plt.tight_layout()
    plt.show()

    print("Sobel Operator is done. ")

    # -- TODO Zitima 4: LoG Filter --
    if not os.path.exists("./log_filter"):
        os.makedirs("./log_filter")

    # (a)
    kernel_LoG1 = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    kernel_LoG2 = np.array([[0, 0, 3, 2, 2, 2, 3, 0, 0],
                            [0, 2, 3, 5, 5, 5, 3, 2, 0],
                            [3, 3, 5, 3, 0, 3, 5, 3, 3],
                            [2, 5, 3, -12, -23, -12, 3, 5, 2],
                            [2, 5, 0, -23, -40, -23, 0, 5, 2],
                            [2, 5, 3, -12, -23, -12, 3, 5, 2],
                            [3, 3, 5, 3, 0, 3, 5, 3, 3],
                            [0, 2, 3, 5, 5, 5, 3, 2, 0],
                            [0, 0, 3, 2, 2, 2, 3, 0, 0]])
    filtered_LoG1 = convolve(img, kernel_LoG1)
    filtered_LoG2 = convolve(img, kernel_LoG2)

    # Use convolve() to convolve img with kernel_LOG1 and kernel_LOG2
    save_img(filtered_LoG1, "./log_filter/q1_LoG1.png")
    save_img(filtered_LoG2, "./log_filter/q1_LoG2.png")

    plt.figure(figsize=(18, 6))

    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(img, cmap='gray')
    plt.colorbar()

    plt.subplot(1, 3, 2)
    plt.title("LoG Filter 1 Output")
    plt.imshow(filtered_LoG1, cmap='gray')
    plt.colorbar()

    plt.subplot(1, 3, 3)
    plt.title("LoG Filter 2 Output")
    plt.imshow(filtered_LoG2, cmap='gray')
    plt.colorbar()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
