from model import config
import torch
from torch.utils.data import Dataset
import cv2
import os


class ImageDataset(Dataset):
    # initialize the constructor
    def __init__(self, data, transforms=None):
        self.transforms = transforms
        self.data = data

    def __getitem__(self, index):
        # retrieve annotations from stored list
        # Part3-3: retrieve bounding box labels
        filename, box_x1, box_y1, box_x2, box_y2, label = self.data[index]
        box_x1, box_y1, box_x2, box_y2 = int(box_x1), int(box_y1), int(box_x2), int(box_y2)

        # get full path of filename
        image_path = os.path.join(config.IMAGES_PATH, label, filename)

        # load the image (in OpenCV format), and grab its dimensions
        image = cv2.imread(image_path)
        h, w = image.shape[:2]
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # scale bounding box coordinates relative to dimensions of input image
        # Part3-3: normalize bounding box coordinates in (0, 1)
        box_x1 /= w
        box_y1 /= h
        box_x2 /= w
        box_y2 /= h
        bbox = torch.tensor([box_x1, box_y1, box_x2, box_y2])


        # normalize label in (0, 1, 2) and convert to tensor
        label = torch.tensor(config.LABELS.index(label))

        # apply image transformations if any
        if self.transforms:
            image = self.transforms(image)

        # return a tuple of the images, labels, and bounding box coordinates
        # Part3-3: add to tuple: normalized bounding box annotations (as tensor)
        return image, label, bbox

    def __len__(self):
        # return the size of the dataset
        return len(self.data)
