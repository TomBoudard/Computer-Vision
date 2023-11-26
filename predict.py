#! /usr/bin/env python

from model import config
import sys
import os
import torch
import cv2

# load our object detector, set it evaluation mode, and label
# encoder from disk
print("**** loading object detector...")
model = torch.load(config.LAST_MODEL_PATH).to(config.DEVICE)
model.eval()

def is_image_file(filename):
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp']
    return any(filename.lower().endswith(ext) for ext in image_extensions)

arg_list = sys.argv[1:]

directories = False
showAll = True
i = 0
while i < len(arg_list):
    mot = arg_list[i]
    if mot == "-d":
        directories = True
        arg_list.pop(i)
    elif mot == "-f":
        showAll = False
        arg_list.pop(i)
    else:
        i += 1

data = []
if directories:
    object_list = []
    for path in arg_list:
        dir_content = os.listdir(path)
        for elt in dir_content:
            if (is_image_file(elt)):
                object_list.append(os.path.join(path, elt))
else:
    object_list = arg_list

for path in object_list:
    if path.endswith('.csv'):
        # loop over CSV file rows (filename, startX, startY, endX, endY, label)
        for row in open(path).read().strip().split("\n"):
            # Part3-6: read bounding box annotations
            filename, box_x1, box_y1, box_x2, box_y2, label = row.split(',')
            box_x1, box_y1, box_x2, box_y2 = int(box_x1), int(box_y1), int(box_x2), int(box_y2)
            filename = os.path.join(config.IMAGES_PATH, label, filename)
            # Part3-6: add bounding box annotations here
            data.append((filename, box_x1, box_y1, box_x2, box_y2, label))
    else:
        data.append((path, None, None, None, None, None))

# loop over images to be tested with our model, with ground truth if available
# Part3-6: must read bounding box annotations once added
i = 0
while i < len(data):
    filename, gt_start_x, gt_start_y, gt_end_x, gt_end_y, gt_label = data[i]
    # load the image, copy it, swap its colors channels, resize it, and
    # bring its channel dimension forward
    image = cv2.imread(filename)
    display = image.copy()
    h, w = display.shape[:2]

    # convert image to PyTorch tensor, normalize it, upload it to the
    # current device, and add a batch dimension
    image = config.TRANSFORMS(image).to(config.DEVICE)
    image = image.unsqueeze(0)

    # predict the bounding box of the object along with the class label
    # Part3-2: need to retrieve label AND bbox predictions once added in network
    predict = model(image)
    predict_label = predict[0]
    predict_bbox = predict[1]

    # determine the class label with the largest predicted probability
    predict_label = torch.nn.Softmax(dim=-1)(predict_label)
    most_likely_label = predict_label.argmax(dim=-1).cpu()
    label = config.LABELS[most_likely_label]

    if (showAll or gt_label != label):
        # Part3-6:denormalize bounding box from (0,1)x(0,1) to (0,w)x(0,h)
        start_x, start_y, end_x, end_y = predict_bbox[0].tolist()
        start_x *= w
        start_y *= h
        end_x *= w  
        end_y *= h

        # draw the ground truth box and class label on the image, if any
        if gt_label is not None:
            cv2.putText(display, 'gt ' + gt_label, (0, h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 0,  0), 2)
        # Part3-6: display ground truth bounding box in blue
        if gt_start_x is not None:
            cv2.rectangle(display, (gt_start_x, gt_start_y), (gt_end_x, gt_end_y), (255, 0,  0))

        # draw the predicted bounding box and class label on the image
        cv2.putText(display, label, (0, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
        # Part3-6: display predicted bounding box, don't forget tp denormalize it!
        cv2.rectangle(display, (int(start_x), int(start_y)), (int(end_x), int(end_y)), (0, 255, 0))

        # show the output image
        cv2.imshow("Output", display)
        # name = filename.split("/")[-1]
        # folder = filename.split("/")[-2]
        # print("datasGuess/" + name)
        # cv2.imwrite("datasGuess/" + folder + "/" + name, display)
        # i += 1

        # exit on escape key or window close
        key = -1
        while key == -1:
            key = cv2.waitKey(100)
            closed = cv2.getWindowProperty('Output', cv2.WND_PROP_VISIBLE) < 1
            if key == 27 or closed:
                exit(0)
            elif key in [81, 82]:
                i -= 1
                i = max(0, i)
            elif key in [13, 32, 83, 84]:
                i += 1
            else:
                key = -1
    else:
        i += 1
