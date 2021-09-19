"""
uses haar cascade to detect faces and then uses the trained neural network to detect keypoints
"""

import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from model import Net
from transforms import *
import torch

import argparse

parser = argparse.ArgumentParser(description='Live Inference')
parser.add_argument('-cam', '--cam_number', type=int, default=1, metavar='', help='0 for internal webcam or 1 for external webcam')
parser.add_argument('-p', '--path', type=str, default="./detector_architectures/haarcascade_frontalface_default.xml", metavar='', help='Path to haar cascade detector metadata')
group = parser.add_mutually_exclusive_group()
group.add_argument('-q', '--quiet', action='store_true', help='print quiet')
group.add_argument('-v', '--verbose', action='store_true', help='print verbose')
args = parser.parse_args()


def normalize(image):
    # convert image to grayscale
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # scale color range from [0, 255] to [0, 1]
    image = image / 255.0
    return image


# rescale image for the neural net
def rescale(image, output_size):
    h, w = image.shape[0:2]
    if h > w:
        # preserves the aspect ratio
        new_h, new_w = output_size * h / w, output_size
    else:
        new_h, new_w = output_size, output_size * w / h

    return cv2.resize(image, (int(new_w), int(new_h)))


def inference(cam_number, plot=False, PATH="./detector_architectures/haarcascade_frontalface_default.xml"):
    cap = cv2.VideoCapture(cam_number)
        # load in a haar cascade classifier for detecting frontal faces
    face_cascade = cv2.CascadeClassifier(
        './detector_architectures/haarcascade_frontalface_default.xml')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Net()
    model.load_state_dict(torch.load('./saved_models/keypoints_model_1-2.pt', map_location=torch.device('cpu')))
    model.to(device)
    model.eval()

    while True:
        ret, frame = cap.read()
        # run the detector
        # the output here is an array of detections; the corners of each detection box
        # if necessary, modify these parameters until you successfully identify every face in a given image
        #cv2.imwrite("file", frame,None)
        faces = face_cascade.detectMultiScale(frame, 1.5, 2, minSize=(100,100), maxSize=(500,500))

        if len(faces) == 0:
            print("No face detected!")
            continue


        image_copy = frame
        # loop over the detected faces, mark the image where each face is found

        with torch.no_grad():

                x, y, w, h = faces[0]
                # draw a rectangle around each detected face
                # you may also need to change the width of the rectangle drawn depending on image resolution
                roi = image_copy[y:y + h, x:x + w]

                # prepare the input frame for the neural net
                frame = normalize(roi)
                frame_to_plot = rescale(frame, 224)
                frame = torch.from_numpy(frame_to_plot)
                frame = frame.unsqueeze(0)
                frame = frame.unsqueeze(0)
                frame = frame.type(torch.FloatTensor)

                output = model(frame)  # outputs pytorch tensor
                output = output.view(output.size()[0], 68, -1)
                output = output.numpy()

                # un-normalize keypoints
                output[0] = output[0] * 50.0 + 100

                if plot:
                    # using opencv since it's faster than matplotlib
                    for i in range(0,67):
                        cv2.circle(frame_to_plot,(output[0][i, 0], output[0][i, 1]), radius=2,color=[255,0,0])
                    cv2.imshow("image", frame_to_plot)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    inference(args.cam_number, True)
