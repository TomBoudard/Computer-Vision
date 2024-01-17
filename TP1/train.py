#! /usr/bin/env python

from model.dataset import ImageDataset
from model.network import ResnetObjectDetector as ObjectDetector
from model import config
import torch
from torch.utils.data import DataLoader
from torch.nn import functional as fun
from torch.optim import Adam
import matplotlib.pyplot as plt
from collections import defaultdict
import random
import time
import os

from PyQt5.QtCore import QLibraryInfo

if __name__ == '__main__':
    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = QLibraryInfo.location(QLibraryInfo.PluginsPath)

    # initialize the list of data (images), class labels, target bounding
    # box coordinates, and image paths
    print("**** loading dataset...")
    data = []

    # loop over all CSV files in the annotations directory
    for csv_file in os.listdir(config.ANNOTS_PATH):
        csv_file = os.path.join(config.ANNOTS_PATH, csv_file)
        # loop over CSV file rows (filename, startX, startY, endX, endY, label)
        for row in open(csv_file).read().strip().split("\n"):
            data.append(row.split(','))

    # randomly partition the data: 80% training, 10% validation, 10% testing
    random.seed(0)
    random.shuffle(data)

    cut_val = int(0.8 * len(data))   # 0.8
    cut_test = int(0.9 * len(data))  # 0.9
    train_data = data[:cut_val]
    val_data = data[cut_val:cut_test]
    test_data = data[cut_test:]

    # create Torch datasets for our training, validation and test data
    train_dataset = ImageDataset(train_data, transforms=config.TRANSFORMS)
    val_dataset = ImageDataset(val_data, transforms=config.TRANSFORMS)
    test_dataset = ImageDataset(test_data, transforms=config.TRANSFORMS)
    print(f"**** {len(train_data)} training, {len(val_data)} validation and "
          f"{len(test_data)} test samples")

    # create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE,
                              shuffle=True, num_workers=config.NB_WORKERS,
                              pin_memory=config.PIN_MEMORY)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE,
                              shuffle=True, num_workers=config.NB_WORKERS,
                              pin_memory=config.PIN_MEMORY)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE,
                             num_workers=config.NB_WORKERS,
                             pin_memory=config.PIN_MEMORY)

    # save testing image paths to use for evaluating/testing our object detector
    print("**** saving training, validation and testing split data as CSV...")
    with open(config.TEST_PATH, "w") as f:
        f.write("\n".join([','.join(row) for row in test_data]))
    with open(config.VAL_PATH, "w") as f:
        f.write("\n".join([','.join(row) for row in val_data]))
    with open(config.TRAIN_PATH, "w") as f:
        f.write("\n".join([','.join(row) for row in train_data]))

    # create our custom object detector model and upload to the current device
    print("**** initializing network...")
    object_detector = ObjectDetector(len(config.LABELS)).to(config.DEVICE)

    # initialize the optimizer, compile the model, and show the model summary
    optimizer = Adam(object_detector.parameters(), lr=config.INIT_LR)
    print(object_detector)

    # initialize history variables for future plot
    plots = defaultdict(list)

    # function to compute loss over a batch
    def compute_loss(loader, back_prop=False):
        # initialize the total loss and number of correct predictions
        total_loss, correct = 0, 0
        total_bbox_loss = 0

        # loop over batches of the training set
        for batch in loader:
            # send the inputs and training annotations to the device
            # Part3-3: modify line below to get bbox data
            images, labels, bboxs = [datum.to(config.DEVICE) for datum in batch]

            # perform a forward pass and calculate the training loss
            # Part3-2: modify line below to get bbox data
            predict = object_detector(images)
            predict_label = predict[0]
            predict_bbox = predict[1]

            # Part3-4: add loss term for bounding boxes
            # class_loss = 0
            class_loss = fun.cross_entropy(predict_label, labels, reduction="sum")
            # bbox_loss = 0
            # bbox_loss = fun.l1_loss(predict_bbox, bboxs, reduction='sum')
            bbox_loss = fun.mse_loss(predict_bbox, bboxs, reduction='sum')
            batch_loss = config.BBOXW * bbox_loss + config.LABELW * class_loss

            # zero out the gradients, perform backprop & update the weights
            if back_prop:
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()

            # add the loss to the total training loss so far and
            # calculate the number of correct predictions
            total_loss += batch_loss
            total_bbox_loss += bbox_loss
            correct_labels = predict_label.argmax(1) == labels
            correct += correct_labels.type(torch.float).sum().item()

        # return sample-level averages of the loss and accuracy
        return total_loss / len(loader.dataset), correct / len(loader.dataset), total_bbox_loss / len(loader.dataset)

    # loop over epochs
    print("**** training the network...")
    prev_val_acc = None
    prev_val_loss = None
    fileDatas = open("dataSimple.txt", "w")
    start_time = time.time()
    for e in range(config.NUM_EPOCHS):
        start_epoch_time = time.time()
        # set model in training mode & backpropagate train loss for all batches
        object_detector.train()

        # Do not use the returned loss
        # The loss of each batch is computed with a "different network"
        # as the weights are updated per batch
        _, _, _ = compute_loss(train_loader, back_prop=True)

        # switch off autograd
        with torch.no_grad():
            # set the model in evaluation mode and compute validation loss
            object_detector.eval()
            train_loss, train_acc, train_bbox_loss = compute_loss(train_loader)
            val_loss, val_acc, val_bbox_loss = compute_loss(val_loader)
        
        end_epoch_time = time.time()

        # update our training history
        plots['Training loss'].append(train_loss.cpu())
        plots['Training bbox loss'].append(train_bbox_loss.cpu())
        plots['Training class accuracy'].append(train_acc)

        plots['Validation loss'].append(val_loss.cpu())
        plots['Validation bbox loss'].append(val_bbox_loss.cpu())
        plots['Validation class accuracy'].append(val_acc)

        # print the model training and validation information
        print(f"**** EPOCH: {e + 1}/{config.NUM_EPOCHS}")
        print(f"**** Computation time of the EPOCH {e + 1}: {end_epoch_time - start_epoch_time:.2f}s****")
        print(f"Train loss: {train_loss:.8f}, Train accuracy: {train_acc:.8f}")
        print(f"Val loss: {val_loss:.8f}, Val accuracy: {val_acc:.8f}")
        print(f"BBox train loss: {train_bbox_loss:.8f}, BBox val loss: {val_bbox_loss:.8f}")

        # Part1-6: write code to store model with highest accuracy, lowest loss
        if (prev_val_acc is None) or (prev_val_acc < val_acc) or (prev_val_acc == val_acc and prev_val_loss > val_loss):
            prev_val_acc = val_acc
            prev_val_loss = val_loss

            # serialize the model to disk
            print("**** saving BEST object detector model...")
            # When a network has dropout and / or batchnorm layers
            # one needs to explicitly set the eval mode before saving
            object_detector.eval()
            torch.save(object_detector, config.BEST_MODEL_PATH)

        #Write epoch datas
        fileDatas.write(str(plots['Training loss'][-1]))
        fileDatas.write(" ")
        fileDatas.write(str(plots['Training class accuracy'][-1]))
        fileDatas.write(" ")
        fileDatas.write(str(plots['Validation loss'][-1]))
        fileDatas.write(" ")
        fileDatas.write(str(plots['Validation class accuracy'][-1]))
        fileDatas.write(" ")
        fileDatas.write(str(end_epoch_time-start_epoch_time))
        fileDatas.write(" ")
        fileDatas.write(str(time.time()-start_time))
        fileDatas.write("\n")

    print("**** saving LAST object detector model...")
    object_detector.eval()
    torch.save(object_detector, config.LAST_MODEL_PATH)

    end_time = time.time()
    print(f"**** total time to train the model: {end_time - start_time:.2f}s")

    # plot the training loss and accuracy
    plt.style.use("ggplot")
    plt.figure()

    # Part1-5: build and save matplotlib plot
    x = [e+1 for e in range(config.NUM_EPOCHS)]
    #Loss
    # plt.subplot(2, 1, 1)
    plt.plot(x, plots['Training loss'], label="Training loss")
    plt.plot(x, plots["Validation loss"], label="Validation loss")
    plt.plot(x, plots["Training bbox loss"], label="Training bbox loss", color="C0", linestyle="--")
    plt.plot(x, plots["Validation bbox loss"], label="Validation bbox loss", color="C1", linestyle="--")
    plt.legend()
    plt.title("Loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.xticks(x)

    #Accuracy
    # plt.subplot(2, 1, 2)
    # plt.plot(x, plots['Training class accuracy'], label="Training class accuracy")
    # plt.plot(x, plots["Validation class accuracy"], label="Validation class accuracy")
    # plt.legend()
    # plt.title("Accuracy")
    # plt.xlabel("epoch")
    # plt.ylabel("accuracy")
    # plt.xticks(x)

    plt.tight_layout() # Pour mieux fit les graphs

    # save the training plot
    plt.savefig(config.PLOT_PATH)
