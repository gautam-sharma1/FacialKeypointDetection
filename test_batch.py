"""
test.py assumes that the best model has been saved under saved_models
"""
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from model import Net
from transforms import *
from live_inference import rescale,normalize
import matplotlib.image as mpimg
from PIL import Image
# Construct the dataset
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

dataset_iter = iter(test_loader)

def main():
    # get the first sample
    sample = next(dataset_iter)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    image = sample['image']
    image = image.float()
    model = Net()
    model.load_state_dict(torch.load('saved_models/keypoints_model_1-2.pt',map_location=torch.device('cpu')))
    model.to(device)
    model.eval()


    with torch.no_grad():
        output = model(image.to(device))
        output = output.view(output.size()[0], 68, -1)
        output[0] = output[0]*50.0+100
        gt = sample['keypoints'][0]*50.0+100
        Plot.visualize_output(image, output, gt_pts=sample['keypoints'],batch_size=10)

if __name__ == "__main__":
    main()