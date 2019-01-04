import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from models import Net


use_gpu = False
if use_gpu:
    device = torch.device("cuda")
    tensor_type = torch.cuda.FloatTensor
else:
    device = torch.device("cpu")
    tensor_type = torch.FloatTensor

net = Net()
if use_gpu:
    net.cuda()
print(net)


model_dir = 'saved_models/'
model_name = 'keypoints_model_200_epochs.pt'
net.load_state_dict(torch.load(model_dir + model_name))


from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

from data_load import FacialKeypointsDataset
from data_load import Rescale, RandomCrop, Normalize, ToTensor

data_transform = transforms.Compose(
        [
            Rescale(250),
            RandomCrop(224),
            Normalize(),
            ToTensor()
        ]
)


# create the test dataset
test_dataset = FacialKeypointsDataset(
    csv_file='/data/test_frames_keypoints.csv',
    root_dir='/data/test/',
    transform=data_transform
)


# load test data in batches
batch_size = 10

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0
)


# test the model on a batch of test images

def net_sample_output():
    
    # iterate through the test dataset
    for i, sample in enumerate(test_loader):
        
        # get sample data: images and ground truth keypoints
        images = sample['image']
        key_pts = sample['keypoints']

        # convert images to FloatTensors
        images = images.type(tensor_type)

        # forward pass to get net output
        output_pts = net(images)
        
        # reshape to batch_size x 68 x 2 pts
        output_pts = output_pts.view(output_pts.size()[0], 68, -1)
        
        # break after first image is tested
        if i == 0:
            return images, output_pts, key_pts
            

def show_all_keypoints(ax, image, predicted_key_pts, gt_pts=None):
    """Show image with predicted keypoints"""
    # image is grayscale
    ax.imshow(image, cmap='gray')
    ax.scatter(predicted_key_pts[:, 0], predicted_key_pts[:, 1], s=80, marker='.', c='m')
    # plot ground truth points as green pts
    if gt_pts is not None:
        ax.scatter(gt_pts[:, 0], gt_pts[:, 1], s=80, marker='.', c='g')


# visualize the output
# by default this shows a batch of 10 images
def visualize_output(test_images, test_outputs, gt_pts=None, batch_size=10):

    fig, ax = plt.subplots(2, int(batch_size / 2), figsize=(40, 20))

    for j in range(2):
        for i in range(int(batch_size / 2)):
            index = 2 * j + i
            # un-transform the image data
            image = test_images[index].data   # get the image from it's Variable wrapper
            if use_gpu:
                image = image.cpu()
            image = image.numpy()   # convert to numpy array from a Tensor
            image = np.transpose(image, (1, 2, 0))   # transpose to go from torch to numpy image

            # un-transform the predicted key_pts data
            predicted_key_pts = test_outputs[index].data
            if use_gpu:
                predicted_key_pts = predicted_key_pts.cpu()
            predicted_key_pts = predicted_key_pts.numpy()
            # undo normalization of keypoints  
            predicted_key_pts = predicted_key_pts * 50.0 + 100

            # plot ground truth points for comparison, if they exist
            ground_truth_pts = None
            if gt_pts is not None:
                ground_truth_pts = gt_pts[index]         
                ground_truth_pts = ground_truth_pts * 50.0 + 100

            ax[j, i].axis("off")
            show_all_keypoints(ax[j][i], np.squeeze(image), predicted_key_pts, ground_truth_pts)

    plt.subplots_adjust(wspace=0.005, hspace=0.005)
    plt.show()



# get a sample of test data again
test_images, test_outputs, gt_pts = net_sample_output()

print(test_images.data.size())
print(test_outputs.data.size())
print(gt_pts.size())


visualize_output(test_images, test_outputs, gt_pts)


# Get the weights in the first conv layer, "conv1"
# if necessary, change this to reflect the name of your first conv layer
weights1 = net.convolution_1.weight.data
if use_gpu:
    weights1 = weights1.cpu()
w = weights1.numpy()

filter_index = 0

print(w[filter_index][0])
print(w[filter_index][0].shape)

# display the filter weights
plt.imshow(w[filter_index][0], cmap='gray')


#
# ### TODO: Filter an image to see the effect of a convolutional kernel
# ---

# In[ ]:


##TODO: load in and display any image from the transformed test dataset

## TODO: Using cv's filter2D function,
## apply a specific set of filter weights (like the one displayed above) to the test image

