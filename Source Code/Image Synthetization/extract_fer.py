import csv
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil

data = {
    'train': {'images': [], 'labels': []},
    'test': {'images': [], 'labels': []},
    'val': {'images': [], 'labels': []}
}

def split_name(s):
    traducere = {
        'Training': 'train',
        'PublicTest': 'val',
        'PrivateTest': 'test'
    }
    return traducere[s]

with open('./dataset/fer2013.csv', newline='') as csvfile:
    file = csv.reader(csvfile, delimiter=',')
    next(file)
    for line in file:
        usage = split_name(line[-1])
        data[usage]['images'].append(
            np.reshape(
                np.array([int(x) for x in line[1].split()]),
                [48, 48]
            )
        )
        data[usage]['labels'].append(line[0])

output_dir = './dataset/fer/'
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)

for usage in data.keys():
    output_folder = os.path.join(output_dir, f'{usage}/')

    for index, (image, label) in enumerate(
        zip(data[usage]['images'], data[usage]['labels'])
    ):
        class_folder = os.path.join(output_folder, label)
        if not os.path.exists(class_folder):
            os.makedirs(class_folder)
        cv2.imwrite(
            os.path.join(class_folder, f'{label}_{index}.png'),
            image
        )

