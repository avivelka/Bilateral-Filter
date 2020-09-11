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


input_image = cv2.imread('img1.jpeg',
                         cv2.IMREAD_UNCHANGED).astype(np.float32)/255.0

# bilateral filter the image

bf = bilateral_filter(input_image[:, :, 0], 2, 0.1)
output_image_00 = np.stack([bf, bf, bf], axis=2)

bf = bilateral_filter(input_image[:, :, 0], 2, 0.2)
output_image_01 = np.stack([bf, bf, bf], axis=2)

bf = bilateral_filter(input_image[:, :, 0], 2, 0.4)
output_image_02 = np.stack([bf, bf, bf], axis=2)

bf = bilateral_filter(input_image[:, :, 0], 2, 0.8)
output_image_03 = np.stack([bf, bf, bf], axis=2)

Row1 = np.hstack([output_image_00, output_image_01,
                  output_image_02, output_image_03])

# write out the image
cv2.imwrite('row1.png', Row1*255.0)


bf = bilateral_filter(input_image[:, :, 0], 4, 0.1)
output_image_00 = np.stack([bf, bf, bf], axis=2)

bf = bilateral_filter(input_image[:, :, 0], 4, 0.2)
output_image_01 = np.stack([bf, bf, bf], axis=2)

bf = bilateral_filter(input_image[:, :, 0], 4, 0.4)
output_image_02 = np.stack([bf, bf, bf], axis=2)

bf = bilateral_filter(input_image[:, :, 0], 4, 0.8)
output_image_03 = np.stack([bf, bf, bf], axis=2)

Row2 = np.hstack([output_image_00, output_image_01,
                  output_image_02, output_image_03])

# write out the image
cv2.imwrite('row2.png', Row2*255.0)


bf = bilateral_filter(input_image[:, :, 0], 8, 0.1)
output_image_00 = np.stack([bf, bf, bf], axis=2)

bf = bilateral_filter(input_image[:, :, 0], 8, 0.2)
output_image_01 = np.stack([bf, bf, bf], axis=2)

bf = bilateral_filter(input_image[:, :, 0], 8, 0.4)
output_image_02 = np.stack([bf, bf, bf], axis=2)

bf = bilateral_filter(input_image[:, :, 0], 8, 0.8)
output_image_03 = np.stack([bf, bf, bf], axis=2)


Row3 = np.hstack([output_image_00, output_image_01,
                  output_image_02, output_image_03])

# write out the image
cv2.imwrite('row3.png', Row3*255.0)


# # write out the image
# cv2.imwrite('vectorized_result.png', Row1*255.0)
