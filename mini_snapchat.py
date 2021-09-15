# import necessary resources
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import torch
from model import Net
from live_inference import inference, normalize, rescale
import cv2



def mini_snap(path = './images/sunglasses.png'):

    cap = cv2.VideoCapture(1)

    # load in a haar cascade classifier for detecting frontal faces
    # TODO: change to PATH
    face_cascade = cv2.CascadeClassifier(
        '/Users/gautamsharma/Desktop/Python/CVND_Exercises/P1_Facial_Keypoints/detector_architectures/haarcascade_frontalface_default.xml')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Net()
    model.load_state_dict(torch.load('saved_models/keypoints_model_1-2.pt', map_location=torch.device('cpu')))
    model.to(device)
    model.eval()

    while True:
        ret, frame = cap.read()
        # run the detector
        # the output here is an array of detections; the corners of each detection box
        # if necessary, modify these parameters until you successfully identify every face in a given image

        faces = face_cascade.detectMultiScale(frame, 1.5, 6)#, minSize=(100,100), maxSize=(500,500))

        if len(faces) == 0:
            print("No face detected!")
            continue


        # make a copy of the original image to plot detections on

        image_copy = np.copy(frame)
        # loop over the detected faces, mark the image where each face is found

        # read in sunglasses
        sunglasses = cv2.imread(path, cv2.IMREAD_UNCHANGED)

        if len(faces) > 1:
            faces = faces[0,:]
            faces.reshape(1,4)



        with torch.no_grad():
            for (x, y, w, h) in faces:
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


                # Display sunglasses on top of the image in the appropriate place

                # copy of the face image for overlay
                image_copy_roi = np.copy(rescale(roi, 224))

                key_pts = output[0]
                # top-left location for sunglasses to go
                # 17 = edge of left eyebrow
                x1 = int(key_pts[17, 0])
                y1 = int(key_pts[17, 1])

                # height and width of sunglasses
                # h = length of nose
                h1 = int(abs(key_pts[27,1] - key_pts[34,1]))
                # w = left to right eyebrow edges
                w1 = int(abs(key_pts[17,0] - key_pts[26,0]))


                # resize sunglasses
                new_sunglasses = cv2.resize(sunglasses, (w1, h1), interpolation = cv2.INTER_CUBIC)

                # get region of interest on the face to change
                roi_color = image_copy_roi[y1:y1+h1,x1:x1+w1]

                # find all non-transparent pts
                ind = np.argwhere(new_sunglasses[:,:,3] > 0)

                # for each non-transparent point, replace the original  image pixel with that of the new_sunglasses
                for i in range(3):
                    roi_color[ind[:,0],ind[:,1],i] = new_sunglasses[ind[:,0],ind[:,1],i]
                # set the area of the image to the changed region with sunglasses
                image_copy_roi[y1:y1+h1,x1:x1+w1] = roi_color
                plt.imshow(image_copy_roi)
                plt.pause()

            if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    cap.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    mini_snap()