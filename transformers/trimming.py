import numpy as np


def trim_transparent_background(image):
    if image.shape[2] != 4:
        print("Image does not have an alpha channel (4th channel).")
        return None

    # Find the coordinates of the non-transparent pixels
    non_transparent_coords = np.argwhere(image[:, :, 3] > 0)

    if non_transparent_coords.size == 0:
        print("No non-transparent pixels found.")
        return None

    # Get the coordinates of the bounding box
    (top, left) = non_transparent_coords.min(axis=0)
    (bottom, right) = non_transparent_coords.max(axis=0)

    # Crop the image to the bounding box
    trimmed_image = image[top : bottom + 1, left : right + 1]

    return trimmed_image
