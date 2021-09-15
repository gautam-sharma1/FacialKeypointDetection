import cv2
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from model import Net
from transforms import *
from live_inference import rescale,normalize
import matplotlib.image as mpimg
import argparse

parser = argparse.ArgumentParser(description='Custom image testing')
parser.add_argument('-ip', '--image_path', type=str, default="./detector_architectures/haarcascade_frontalface_default.xml", metavar='', help='Path to custom image')
parser.add_argument('-fp', '--filter_path', type=str, default="./detector_architectures/haarcascade_frontalface_default.xml", metavar='', help='Path to custom image')
group = parser.add_mutually_exclusive_group()
group.add_argument('-q', '--quiet', action='store_true', help='print quiet')
group.add_argument('-v', '--verbose', action='store_true', help='print verbose')
args = parser.parse_args()


def prepare_image_for_net(PATH):
    image = cv2.imread(PATH)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mod_image = normalize(image)
    mod_image = cv2.resize(mod_image, (224,224))
    mod_image = torch.from_numpy(mod_image)
    mod_image = mod_image.unsqueeze(0)
    mod_image = mod_image.unsqueeze(0)
    mod_image = mod_image.type(torch.FloatTensor)
    # return original image (RGB) and modified image
    return image, mod_image


def prepare_sunglass_filter(filter, key_pts):
    x1 = int(key_pts[17, 0])
    y1 = int(key_pts[17, 1])

    # height and width of sunglasses
    # h = length of nose
    h1 = int(abs(key_pts[27,1] - key_pts[34,1]))
    # w = left to right eyebrow edges
    w1 = int(abs(key_pts[17,0] - key_pts[26,0]))

    # resize sunglasses
    new_sunglasses = cv2.resize(filter, (w1, h1), interpolation = cv2.INTER_CUBIC)
    return new_sunglasses, (x1, y1, h1, w1)



def main(PATH='./images/gautam.png'):

    # Instantiate and setup model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Net()
    model.load_state_dict(torch.load('saved_models/keypoints_model_1-2.pt',map_location=torch.device('cpu')))
    model.to(device)
    model.eval()

    original_image, my_img = prepare_image_for_net(PATH)
    sunglasses = cv2.imread('./images/sunglasses.png', cv2.IMREAD_UNCHANGED)

    with torch.no_grad():
        output = model(my_img)
        output = output.view(output.size()[0], 68, -1)
        output[0] = output[0]*50.0+100

        image_copy_roi = cv2.resize(original_image, (224,224))
        key_pts = output[0]

        # resize sunglasses and get size of sunglass
        new_sunglasses,(x1, y1, h1, w1) = prepare_sunglass_filter(sunglasses,key_pts)

        # get region of interest on the face to change
        # denotes the region that will be covered by the glasses
        roi_color = image_copy_roi[y1:y1+h1,x1:x1+w1]

        # find all non-transparent pts
        ind = np.argwhere(new_sunglasses[:,:,3] > 0)

        # for each non-transparent point, replace the original  image pixel with that of the new_sunglasses
        for i in range(3):
            roi_color[ind[:,0],ind[:,1],i] = new_sunglasses[ind[:,0],ind[:,1],i]
        # set the area of the image to the changed region with sunglasses
        image_copy_roi[y1:y1+h1,x1:x1+w1] = roi_color
        plt.imshow(image_copy_roi)
        plt.pause(0.0001)


if __name__ == "__main__":
    main()