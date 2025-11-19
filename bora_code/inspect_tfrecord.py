"""
Small script to inspect the contents of tfrecord files such as the ones used by the DROID 100 dataset.
I will read from these, they are located in: /

Reference(s): https://www.geeksforgeeks.org/deep-learning/how-to-inspect-a-tensorflow-tfrecord-file/
"""


import tensorflow as tf
import numpy as np
import glob
import os
import sys
import io
from PIL import Image


def parse_tfrecord_dataset(dataset):
    """Function for parsing a tfrecord file
    """
    return 0


if __name__ == "__main__":
    """Main
    """
    # Set the path to the tfrecord files, ignoring non-tfrecord files in the directory
    tfrecord_path = "../datasets/droid_100/1.0.0/*.tfrecord*"

    # Glob patterns cannot inherently map to the desired files, so import and use glob to explicitly store them as a list
    print("************************ Printing file names ************************")
    tfrecord_files = sorted(glob.glob(tfrecord_path))
    for i in range(0, len(tfrecord_files)):
        print(f"[{i}] {tfrecord_files[i]}")

    # Instantiate the raw tfrecords as a dataset
    dataset = tf.data.TFRecordDataset(tfrecord_files)
    print(f"dataset: {dataset}")

    all_image_triples = []
    
    # Per-episode loop starts here
    for i, raw_record in enumerate(dataset):
        example = tf.train.Example()
        example.ParseFromString(raw_record.numpy())
        # Outputs are massive and confusing, so let's only print the feature keys
        # print(example)

        image_triple = {}

        print(f"Episode {i} Feature Keys:")
        # print(list(example.features.feature.keys()))
        for key in sorted(example.features.feature.keys()):
            feature = example.features.feature[key]
            # There are only three possible kinds of feature keys: bytes_list, float_list, and int64_list
            kind = feature.WhichOneof('kind')
            if kind == 'bytes_list':
                print(f"{key:40} {kind:>15}             len = {len(feature.bytes_list.value)}")
            elif kind == 'float_list':
                print(f"{key:40} {kind:>15}             len = {len(feature.float_list.value)}")
            elif kind == 'int64_list':
                print(f"{key:40} {kind:>15}             len = {len(feature.int64_list.value)}")
            else:
                print("FEATURE TYPE NOT MATCHED")
        
        # Store images
        temp_dict = {}
        for key in example.features.feature.keys():
            feature = example.features.feature[key]
            if 'image' in key:
                temp_dict.update({key: feature.bytes_list.value})

        for key in temp_dict.keys():
            camera_name = key.split('/')[-1]
            os.makedirs(f"episodes/episode_{i:06d}/{camera_name}/images", exist_ok=True)

        for key, image_bytes_list in temp_dict.items():
            camera_name = key.split('/')[-1]
            for step_idx, image_bytes in enumerate(image_bytes_list):
                img = Image.open(io.BytesIO(image_bytes))
                img.save(f"episodes/episode_{i:06d}/{camera_name}/images/{camera_name}_step{step_idx:04d}.png")

        all_image_triples.append(temp_dict)

        print("==============================================================================================")

    # print(f"All image triples ({len(all_image_triples)}): {all_image_triples}")
    # Now we have all the image triples for all 100 episodes. Each triple is a mini dict of camera_name: image_data pairs
    # Since we appended to all_image_triples, [0] will be the first episode's camera



    

