import os


def get_proto_and_model(modelname: str):
    txt_file = None
    caffe_model_file = None

    path = f"models/{modelname}"

    # List files in the directory
    for filename in os.listdir(path):
        if filename.endswith(".txt") and txt_file is None:
            txt_file = f"{path}/{filename}"
        elif filename.lower().endswith(".caffemodel") and caffe_model_file is None:
            caffe_model_file = f"{path}/{filename}"

        if txt_file is not None and caffe_model_file is not None:
            # Found both files, break the loop
            break

    return txt_file, caffe_model_file
