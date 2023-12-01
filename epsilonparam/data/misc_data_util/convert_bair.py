import os
import numpy as np
from PIL import Image
import tensorflow.compat.v1 as tf
from tensorflow.python.platform import gfile
from imageio import imwrite as imsave

# Convert BAIR robot pushing data to numpy to use with PyTorch
# Based on Emily Denton's script: https://github.com/edenton/svg/blob/master/data/convert_bair.py


def convert(data_path):
    # iterate through the data splits
    for data_split in ['train', 'test']:
        os.makedirs(os.path.join(data_path, data_split))
        data_split_path = os.path.join(data_path, 'softmotion30_44k', data_split)
        data_split_files = gfile.Glob(os.path.join(data_split_path, '*'))
        # iterate through the TF records
        for f in data_split_files:
            print('Current file: ' + f)
            ind = int(f.split('/')[-1].split('_')[1])  # starting video index
            # iterate through the sequences in this TF record
            for serialized_example in tf.python_io.tf_record_iterator(f):
                os.makedirs(os.path.join(data_path, data_split, str(ind)))
                example = tf.train.Example()
                example.ParseFromString(serialized_example)
                # iterate through the sequence
                for i in range(30):
                    image_name = str(i) + '/image_aux1/encoded'
                    byte_str = example.features.feature[image_name].bytes_list.value[0]
                    img = Image.frombytes('RGB', (64, 64), byte_str)
                    img = np.array(img.getdata()).reshape(img.size[1], img.size[0], 3) / 255.
                    imsave(os.path.join(data_path, data_split, str(ind), str(i) + '.png'), img)
                print('     Finished processing sequence ' + str(ind))
                ind += 1
