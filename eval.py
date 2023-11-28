#! /usr/bin/env python

from model import config
import sys
import os
import torch
import cv2
import numpy

# load our object detector, set it evaluation mode, and label

if len(sys.argv) < 2:
    print("Please enter the path to the model to be evaluated")
    sys.exit(1)

model_path = sys.argv[1]

print(f"**** loading object detector at {model_path}...")
model = torch.load(model_path).to(config.DEVICE)
model.eval()
print(f"**** object detector loaded")

results_labels = dict()
results_bbox = dict()

for mode, csv_file in [['train', config.TRAIN_PATH],
                       ['validation', config.VAL_PATH],
                       ['test', config.TEST_PATH],]:
    data = []
    assert(csv_file.endswith('.csv'))

    print(f"Evaluating {mode} set...")
    # loop over CSV file rows (filename, startX, startY, endX, endY, label)
    for row in open(csv_file).read().strip().split("\n"):
        # Part3-2: read bounding box annotations
        filename, box_x1, box_y1, box_x2, box_y2, label = row.split(',')
        filename = os.path.join(config.IMAGES_PATH, label, filename)
        # Part3-2: add bounding box annotations here
        data.append((filename, int(box_x1), int(box_y1), int(box_x2), int(box_y2), label))

    print(f"Evaluating {len(data)} samples...")

    # Store all results as well as per class results
    results_labels[mode] = dict()
    results_bbox[mode] = dict()
    results_labels[mode]['all'] = []
    results_bbox[mode]['all'] = []
    results_bbox[mode]['allcorrect'] = []
    results_bbox[mode]['allwrong'] = []
    for label_str in config.LABELS:
        results_labels[mode][label_str] = []
        results_bbox[mode][label_str] = []
        results_bbox[mode][label_str+'correct'] = []
        results_bbox[mode][label_str+'wrong'] = []

    # loop over the images that we'll be testing using our bounding box
    # regression model
    for filename, gt_start_x, gt_start_y, gt_end_x, gt_end_y, gt_label in data:
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
        predict = model(image)
        predict_label = predict[0]
        predict_bbox = predict[1]

        # determine the class label with the largest predicted probability
        predict_label = torch.nn.Softmax(dim=-1)(predict_label)
        most_likely_label = predict_label.argmax(dim=-1).cpu()
        label = config.LABELS[most_likely_label]

        # Part3-2: denormalize bounding box from (0,1)x(0,1) to (0,w)x(0,h)
        start_x, start_y, end_x, end_y = predict_bbox[0].tolist()
        start_x *= w
        start_y *= h
        end_x *= w  
        end_y *= h

        # Compare to gt data
        results_labels[mode]['all'].append(label == gt_label)
        results_labels[mode][gt_label].append(label == gt_label)

        # Part3-2: compute cumulated bounding box metrics
        intersect_start_x, intersect_start_y, intersect_end_x, intersect_end_y = max(start_x, gt_start_x), max(start_y, gt_start_y), min(end_x, gt_end_x), min(end_y, gt_end_y)
        gt_area = (gt_end_x-gt_start_x) * (gt_end_y - gt_start_y)
        area = (end_x-start_x) * (end_y - start_y)
        intersect_area = max(0, (intersect_end_x-intersect_start_x)) * max(0, (intersect_end_y - intersect_start_y))
        overlapPercent = intersect_area / (gt_area + area - intersect_area)
        results_bbox[mode]['all'].append(overlapPercent)
        if label == gt_label:
            results_bbox[mode]['all' + "correct"].append(overlapPercent)
        else:
            results_bbox[mode]['all' + "wrong"].append(overlapPercent)
        results_bbox[mode][gt_label].append(overlapPercent)
        if label == gt_label:
            results_bbox[mode][gt_label + "correct"].append(overlapPercent)
        else:
            results_bbox[mode][gt_label + "wrong"].append(overlapPercent)

        if label != gt_label:
            print(f"\tFailure at {filename}")


# Compute per dataset accuracy
for mode in ['train', 'validation', 'test']:
    print(f'\n*** {mode} set accuracy')
    print(f"\tMean accuracy for all labels: "
          f"{numpy.mean(numpy.array(results_labels[mode]['all']))}")
    # Part3-2: display bounding box metrics
    print(f"\tMean bbox accuracy for all labels: "
          f"{numpy.mean(numpy.array(results_bbox[mode]['all']))}")
    print(f'\n\tMean bbox accuracy for all labels correclty guessed: '
            f'{numpy.mean(numpy.array(results_bbox[mode]["all" + "correct"]))}')
    print(f'\n\tMean bbox accuracy for all labels wrongly guessed: '
            f'{numpy.mean(numpy.array(results_bbox[mode]["all" + "wrong"]))}')

    for label_str in config.LABELS:
        print(f'\n\tMean accuracy for label {label_str}: '
              f'{numpy.mean(numpy.array(results_labels[mode][label_str]))}')
        print(f'\t\t {numpy.sum(results_labels[mode][label_str])} over '
              f'{len(results_labels[mode][label_str])} samples')
        # Part3-2: display bounding box metrics
        print(f'\n\tMean bbox accuracy for label {label_str}: '
              f'{numpy.mean(numpy.array(results_bbox[mode][label_str]))}')
        print(f'\n\tMean bbox accuracy for label {label_str} correclty guessed: '
              f'{numpy.mean(numpy.array(results_bbox[mode][label_str + "correct"]))}')
        print(f'\n\tMean bbox accuracy for label {label_str} wrongly guessed: '
              f'{numpy.mean(numpy.array(results_bbox[mode][label_str + "wrong"]))}')

