# Construct the dataset
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from model import Net
import cv2 as cv2
from transforms import *

data_transform = transforms.Compose([Rescale(250),
                                     RandomCrop(224), Normalize(), ToTensor()
                                     ])

test_dataset = FacialKeypointsDataset(csv_file='./data/test_frames_keypoints.csv',
                                      dataset_location='./data/test',
                                      transforms=data_transform)
test_loader = DataLoader(test_dataset,
                         batch_size=10,
                         shuffle=True,
                         num_workers=2)


def main():
    dataset_iter = iter(test_loader)

    # get the first sample
    sample = next(dataset_iter)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    image = sample['image']
    image = image.float()
    model = Net()
    model.load_state_dict(torch.load('./saved_models/keypoints_model_1-2.pt',map_location=torch.device('cpu')))
    model.to(device)
    model.eval()


    with torch.no_grad():
        # convert filter to numpy
        first_conv2D_filter = model.conv5.weight.data.numpy()
        src_image = image[0].numpy()
        src_image = np.transpose(src_image, (1, 2, 0))   # transpose to go from torch to numpy image

        # select a 2D filter from a 4D filter output
        plt.imshow(first_conv2D_filter[0][0],cmap="gray")
        plt.pause(0.001)
        filtered_image = cv2.filter2D(np.squeeze(src_image), -1, first_conv2D_filter[0][0])
        plt.imshow(filtered_image, cmap="gray")
        plt.waitforbuttonpress()


if __name__ == "__main__":
    main()