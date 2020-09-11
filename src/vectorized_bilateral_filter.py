import numpy as np
import cv2


def gaussian(x_square, sigma):
    return np.exp(-0.5*x_square/sigma**2)


def bilateral_filter(image, sigma_space, sigma_intensity):
    # kernel_size should be twice the sigma space to avoid calculating negligible values
    kernel_size = int(2*sigma_space+1)
    half_kernel_size = int(kernel_size / 2)
    result = np.zeros(image.shape)
    W = 0

    # Iterating over the kernel
    for x in range(-half_kernel_size, half_kernel_size+1):
        for y in range(-half_kernel_size, half_kernel_size+1):
            Gspace = gaussian(x ** 2 + y ** 2, sigma_space)
            shifted_image = np.roll(image, [x, y], [1, 0])
            intensity_difference_image = image - shifted_image
            Gintenisity = gaussian(
                intensity_difference_image ** 2, sigma_intensity)
            result += Gspace*Gintenisity*shifted_image
            W += Gspace*Gintenisity

    return result / W


input_image = cv2.imread('lena.png',
                         cv2.IMREAD_UNCHANGED).astype(np.float32)/255.0

# bilateral filter the image

bf = bilateral_filter(input_image[:, :, 0], 5.0, 0.1)
output_image = np.stack([bf, bf, bf], axis=2)

# write out the image
cv2.imwrite('output_lena.png', output_image*255.0)
