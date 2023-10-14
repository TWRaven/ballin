import cv2
import numpy as np


def extract_face_with_dnn(image, prototxt_path, model_path):
    net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

    # Resize the image to a standard size for face detection
    target_size = (300, 300)
    blob = cv2.dnn.blobFromImage(image, 1.0, target_size, (104, 177, 123))

    # Set the input to the network and perform a forward pass
    net.setInput(blob)
    detections = net.forward()

    # Initialize variables to keep track of the highest confidence face
    max_confidence = 0
    best_box = None

    # Iterate through all detected faces
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        # If this face has higher confidence, update the best box
        if confidence > max_confidence:
            max_confidence = confidence
            box = detections[0, 0, i, 3:7] * np.array(
                (image.shape[1], image.shape[0], image.shape[1], image.shape[0])
            )
            best_box = box.astype(int)

    if best_box is None:
        print("No faces detected in the image.")
        return None

    # Define the face bounding box
    (x, y, x2, y2) = best_box

    # Create a mask of the same size as the image
    mask = np.zeros(image.shape[:2], np.uint8)

    # Define the background and foreground models
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    # Apply GrabCut algorithm
    cv2.grabCut(
        image, mask, (x, y, x2, y2), bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT
    )

    # Modify the mask to create a binary mask for the foreground
    mask2 = np.where((mask == 2) | (mask == 0), 0, 255).astype("uint8")

    # Multiply the image with the binary mask to get the extracted face
    result = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    result[:, :, 3] = mask2

    return result
