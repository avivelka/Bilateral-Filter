import numpy as np
import cv2


def gaussian(x_square, sigma):
    return np.exp(-0.5*x_square/sigma**2)


def bilateral_filter(image, sigma_space, sigma_intensity):
    kernel_size = int(2*sigma_space+1)
    half_kernel_size = int(kernel_size/2)
    result = np.zeros(image.shape)
    W = np.ones(image.shape)

    for Px in range(half_kernel_size, image.shape[0] - half_kernel_size):
        for Py in range(half_kernel_size, image.shape[1] - half_kernel_size):

            for x in range(-half_kernel_size, half_kernel_size+1):
                for y in range(-half_kernel_size, half_kernel_size+1):
                    Gspace = gaussian(x ** 2 + y ** 2, sigma_space)

                    pixel_intensity_difference = image[Px,
                                                       Py] - image[Px+x, Py+y]
                    Gintensity = gaussian(
                        pixel_intensity_difference ** 2, sigma_intensity)

                    result[Px, Py] += Gspace * Gintensity * \
                        image[Px+x, Py+y]

                    W[Px, Py] += Gspace * Gintensity

        print("Processing: ", int(Px / image.shape[0] * 100), "%")

    return result / W


input_image = cv2.imread('lena.png',
                         cv2.IMREAD_UNCHANGED).astype(np.float32)/255.0

# Grayscale Bilateral Filter
bf = bilateral_filter(input_image[:, :, 0], 5.0, 0.1)
output_image = np.stack([bf, bf, bf], axis=2)

cv2.imwrite('output_naive_lena.png', output_image*255.0)
