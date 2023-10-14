import numpy as np
import cv2
import math


def image_to_circle(img, gain=1.5):
    # resize image and grayscale image to a square
    height, width = img.shape[:2]
    dimension = min(width, height)
    img = cv2.resize(img, (dimension, dimension))

    xcent = ycent = dimension / 2

    h, w = dimension, dimension
    # set up the x and y maps as float32
    map_x = np.zeros((h, w), np.float32)
    map_y = np.zeros((h, w), np.float32)
    mask = np.zeros((h, w), np.uint8)

    # create map with the spherize distortion formula --- arcsin(r)
    # xcomp = arcsin(r)*x/r; ycomp = arsin(r)*y/r
    for y in range(h):
        Y = (y - ycent) / ycent
        for x in range(w):
            X = (x - xcent) / xcent
            R = math.hypot(X, Y)
            if R == 0:
                map_x[y, x] = x
                map_y[y, x] = y
                mask[y, x] = 255
            elif R > 1:
                map_x[y, x] = x
                map_y[y, x] = y
                mask[y, x] = 0
            elif gain >= 0:
                map_x[y, x] = (
                    xcent * X * math.pow((2 / math.pi) * (math.asin(R) / R), gain)
                    + xcent
                )
                map_y[y, x] = (
                    ycent * Y * math.pow((2 / math.pi) * (math.asin(R) / R), gain)
                    + ycent
                )
                mask[y, x] = 255
            elif gain < 0:
                gain2 = -gain
                map_x[y, x] = (
                    xcent * X * math.pow((math.sin(math.pi * R / 2) / R), gain2) + xcent
                )
                map_y[y, x] = (
                    ycent * Y * math.pow((math.sin(math.pi * R / 2) / R), gain2) + ycent
                )
                mask[y, x] = 255

    # do the remap  this is where the magic happens
    result = cv2.remap(
        img,
        map_x,
        map_y,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
        borderValue=(0, 0, 0),
    )

    # process with mask
    result = cv2.cvtColor(result, cv2.COLOR_BGR2BGRA)
    result[:, :, 3] = mask

    return result
