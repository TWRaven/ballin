import argparse
import cv2
from os.path import splitext

from transformers import (
    extract_face_with_dnn,
    get_proto_and_model,
    image_to_circle,
    trim_transparent_background,
)


parser = argparse.ArgumentParser(
    description="Extract a face from an image and project it onto a sphere.",
)

parser.add_argument(
    "image_path",
    type=str,
    help="Path to the image file.",
)

parser.add_argument(
    "-o",
    "--output",
    type=str,
    nargs="?",
    help="Path to the output file.",
)

parser.add_argument(
    "-m",
    "--model",
    type=str,
    nargs="?",
    help="Name of the model to use.",
    default="two",
)

args = parser.parse_args()

if not args.output:
    args.output = f"{splitext(args.image_path)[0]}-output.png"


image = cv2.imread(args.image_path)

extracted_face = extract_face_with_dnn(image, *get_proto_and_model(args.model))

trimmed_face = trim_transparent_background(extracted_face)

projected_face = image_to_circle(trimmed_face)

cv2.imwrite(args.output, projected_face)
