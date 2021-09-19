import matplotlib.pyplot as plt
import numpy as np
class Plot:
    def __init__(self, image_dataset, transforms=None):
        if not transforms:
            self.image_dataset = image_dataset
        else:
            self.image_dataset = transforms(image_dataset)

    def plot(self, num_to_display=3):
        for i in range(num_to_display):
            fig = plt.figure(figsize=(20,10))
            ax = plt.subplot(1, num_to_display, i+1)

            # randomly select a sample
            rand_i = np.random.randint(0, len(self.image_dataset))

            # extract dictionary of {images:keypoints}
            sample = self.image_dataset[rand_i]

            # extract keypoints and reshape into two columns
            keypoints = sample['keypoints']#.astype('float')#.reshape(-1, 2)
            print(keypoints)
            ax.set_title('Sample #{}'.format(i))

            # plot image
            #print(sample['image'].shape)
            plt.imshow(sample['image'])

            # plot keypoints
            plt.scatter(keypoints[:, 0], keypoints[:, 1], s=5, marker='*', c='c')

            # pause to flush out graphics
            plt.pause(0.01)

    @staticmethod
    def show_all_keypoints(image, predicted_key_pts, gt_pts=None):
        """Show image with predicted keypoints"""
        # image is grayscale
        plt.imshow(image, cmap='gray')
        plt.scatter(predicted_key_pts[:, 0], predicted_key_pts[:, 1], s=20, marker='.', c='m')
        # plot ground truth points as green pts
        if gt_pts is not None:
            plt.scatter(gt_pts[:, 0], gt_pts[:, 1], s=20, marker='.', c='g')
        plt.show()

    @staticmethod
    # visualize the output
# by default this shows a batch of 10 images
    def visualize_output(test_images, test_outputs, gt_pts=None, batch_size=10):

        for i in range(batch_size):
            plt.figure(figsize=(20,10))
            ax = plt.subplot(1, batch_size, i+1)

            # un-transform the image data
            image = test_images[i].data   # get the image from it's wrapper
            image = image.numpy()   # convert to numpy array from a Tensor
            image = np.transpose(image, (1, 2, 0))   # transpose to go from torch to numpy image

            # un-transform the predicted key_pts data
            predicted_key_pts = test_outputs[i].data
            predicted_key_pts = predicted_key_pts.numpy()
            # undo normalization of keypoints
            predicted_key_pts = predicted_key_pts*50.0+100

            # plot ground truth points for comparison, if they exist
            ground_truth_pts = None
            if gt_pts is not None:
                ground_truth_pts = gt_pts[i]
                ground_truth_pts = ground_truth_pts*50.0+100

            # call show_all_keypoints
            Plot.show_all_keypoints(np.squeeze(image), predicted_key_pts, ground_truth_pts)

            plt.axis('off')

        plt.show()


if __name__ == "__main__":
    from transforms import *

    # Construct the dataset
    face_dataset = FacialKeypointsDataset(csv_file='./data/training_frames_keypoints.csv',
                                          dataset_location='./data/training')

    # print some stats about the dataset
    print('Length of dataset: ', len(face_dataset))

    p = Plot(face_dataset)

    # plots three random images
    p.plot(num_to_display=3)


