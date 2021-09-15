#!/usr/bin/python

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
from data import FacialKeypointsDataset
from model import Net
from transforms import *

parser = argparse.ArgumentParser(description = 'Keypoint Detector')
parser.add_argument('-d','--dir', type=str, metavar='',help = 'data directory')
parser.add_argument('-b','--batch_size', type=int, metavar='',help = 'batch size')
parser.add_argument('-e','--epochs', type=int, metavar='',help = 'number of epochs')
group = parser.add_mutually_exclusive_group()
group.add_argument('-q', '--quiet', action='store_true', help='print quiet')
group.add_argument('-v', '--verbose', action='store_true', help='print verbose')
args = parser.parse_args()



def make_dataset(PATH_TO_DATA):
    data_transform = transforms.Compose([Rescale(250),
                                         RandomCrop(224), Normalize(), ToTensor()
                                                                                    ])
    # Construct the dataset
    transformed_dataset = FacialKeypointsDataset(csv_file=PATH_TO_DATA+'/training_frames_keypoints.csv',
                                                 dataset_location=PATH_TO_DATA+'/training',transforms=data_transform)


    test_dataset = FacialKeypointsDataset(csv_file=PATH_TO_DATA+'/test_frames_keypoints.csv',
                                          dataset_location=PATH_TO_DATA+'/test',
                                          transforms=data_transform)

    return transformed_dataset,test_dataset







def train_net(n_epochs, transformed_dataset, test_dataset, batch_size = 10):
    # load training data in batches
    batch_size = 10

    train_loader = DataLoader(transformed_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=2)
    test_loader = DataLoader(test_dataset,
                             batch_size=batch_size,
                             shuffle=True,
                             num_workers=2)
    net = Net()
    criterion = nn.MSELoss()
    # stochastic gradient descent with a small learning rate AND some momentum
    #optimizer = optim.Adam(net.parameters(),lr=0.001)
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    training_loss = []
    validation_loss = []

    # prepare the net for training
    net.train()
    print("__________________________TRAINING STARTED___________________________")
    for epoch in range(n_epochs):  # loop over the dataset multiple times
        print("__________________________EPOCH %s___________________________" % epoch )
        running_loss = 0.0
        average_loss = 0.0   # loss per epoch
        # train on batches of data, assumes you already have train_loader
        for batch_i, data in enumerate(train_loader):
            # get the input images and their corresponding labels
            images = data['image']
            key_pts = data['keypoints']

            # flatten pts
            key_pts = key_pts.view(key_pts.size(0), -1)

            # convert variables to floats for regression loss
            key_pts = key_pts.type(torch.FloatTensor)
            images = images.type(torch.FloatTensor)

            # forward pass to get outputs
            output_pts = net(images)

            # calculate the loss between predicted and target keypoints
            loss = criterion(output_pts, key_pts)

            # zero the parameter (weight) gradients
            optimizer.zero_grad()

            # backward pass to calculate the weight gradients
            loss.backward()

            # update the weights
            optimizer.step()

            # print loss statistics
            # to convert loss into a scalar and add it to the running_loss, use .item()
            running_loss += loss.item()
            average_loss += loss.item()*images[0]  # total loss for all the images
            if batch_i % 10 == 9:    # print every 10 batches
                print('Epoch: {}, Batch: {}, Avg. Loss: {}'.format(epoch + 1, batch_i+1, running_loss/1000))
                running_loss = 0.0
        training_loss.append(average_loss/len(train_loader.dataset))



        net.eval()
        with torch.no_grad():
            val_loss = 0
            for data in test_loader:

                # get the input images and their corresponding labels
                images = data['image']

                key_pts = data['keypoints']

                # flatten pts
                key_pts = key_pts.view(key_pts.size(0), -1)

                # convert variables to CUDA floats for regression loss
                key_pts = key_pts.type(torch.FloatTensor)
                images = images.type(torch.FloatTensor)

                # forward pass to get outputs
                output_pts = net(images)

                # calculate the loss between predicted and target keypoints
                loss = criterion(output_pts, key_pts)

                val_loss += loss.item()*images.size(0)
            validation_loss.append(val_loss/len(test_loader.dataset))



    print('Finished Training')
    model_dir = 'saved_models/'
    model_name = 'keypoints_model_1.pt'

    # after training, save your model parameters in the dir 'saved_models'
    torch.save(net.state_dict(), model_dir+model_name)
    print('Model Saved!')


if __name__ == "__main__":
    if not args.dir:
        raise NotImplementedError("Data directory not defined!")

    transformed_dataset, test_dataset = make_dataset(args.dir)

    if not args.epochs:
        raise NotImplementedError("Please define the number of epochs to train!")

    if not args.batch_size:
        print("Defaulting to batch size of 10")

    train_net(args.epochs, transformed_dataset, test_dataset, args.batch_size)


