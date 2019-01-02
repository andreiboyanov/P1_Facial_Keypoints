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

transformed_dataset = FacialKeypointsDataset(
    csv_file='/data/training_frames_keypoints.csv',
    root_dir='/data/training/',
    transform=data_transform
)


print('Number of images: ', len(transformed_dataset))

# iterate through the transformed dataset and print some stats about the first few samples
for i in range(4):
    sample = transformed_dataset[i]
    print(i, sample['image'].size(), sample['keypoints'].size())


# load training data in batches
batch_size = 30

train_loader = DataLoader(
    transformed_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0
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
            

# call the above function
# returns: test images, test predicted keypoints, test ground truth keypoints
test_images, test_outputs, gt_pts = net_sample_output()


# print out the dimensions of the data to see if they make sense
print(test_images.data.size())
print(test_outputs.data.size())
print(gt_pts.size())


def show_all_keypoints(ax, image, predicted_key_pts, gt_pts=None):
    """Show image with predicted keypoints"""
    # image is grayscale
    ax.imshow(image, cmap='gray')
    ax.scatter(predicted_key_pts[:, 0], predicted_key_pts[:, 1], s=80, marker='*', c='m')
    # plot ground truth points as green pts
    if gt_pts is not None:
        ax.scatter(gt_pts[:, 0], gt_pts[:, 1], s=80, marker='*', c='g')


# visualize the output
# by default this shows a batch of 10 images
def visualize_output(test_images, test_outputs, gt_pts=None, batch_size=10):

    fig, ax = plt.subplots(2, int(batch_size / 2), figsize=(40, 20))

    for j in range(2):
        for i in range(int(batch_size / 2)):
            index = 2 * j + i
            # un-transform the image data
            image = test_images[index].data   # get the image from it's Variable wrapper
            image = image.numpy()   # convert to numpy array from a Tensor
            image = np.transpose(image, (1, 2, 0))   # transpose to go from torch to numpy image

            # un-transform the predicted key_pts data
            predicted_key_pts = test_outputs[index].data
            predicted_key_pts = predicted_key_pts.numpy()
            # undo normalization of keypoints  
            predicted_key_pts = predicted_key_pts*50.0+100

            # plot ground truth points for comparison, if they exist
            ground_truth_pts = None
            if gt_pts is not None:
                ground_truth_pts = gt_pts[index]         
                ground_truth_pts = ground_truth_pts * 50.0 + 100

            # call show_all_keypoints
            ax[j, i].axis("off")
            show_all_keypoints(ax[j][i], np.squeeze(image), predicted_key_pts, ground_truth_pts)

    plt.subplots_adjust(wspace=0.005, hspace=0.005)
    plt.show()


# call it
visualize_output(test_images, test_outputs, gt_pts)


import torch.optim as optim

criterion = nn.SmoothL1Loss()
optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8)


def train_net(n_epochs):

    # prepare the net for training
    net.train()

    for epoch in range(n_epochs):  # loop over the dataset multiple times
        
        running_loss = 0.0

        # train on batches of data, assumes you already have train_loader
        for batch_i, data in enumerate(train_loader):
            # get the input images and their corresponding labels
            images = data['image']
            key_pts = data['keypoints']

            # flatten pts
            key_pts = key_pts.view(key_pts.size(0), -1)

            # convert variables to floats for regression loss
            key_pts = key_pts.type(tensor_type)
            images = images.type(tensor_type)

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
            running_loss += loss.item()
            if batch_i % 10 == 9:    # print every 10 batches
                print('Epoch: {}, Batch: {}, Avg. Loss: {}'.format(epoch + 1, batch_i+1, running_loss/10))
                running_loss = 0.0

    print('Finished Training')


# train your network
n_epochs = 2
train_net(n_epochs)

# get a sample of test data again
test_images, test_outputs, gt_pts = net_sample_output()

print(test_images.data.size())
print(test_outputs.data.size())
print(gt_pts.size())


visualize_output(test_images, test_outputs, gt_pts)


model_dir = 'saved_models/'
model_name = 'keypoints_model_temp.pt'

# after training, save your model parameters in the dir 'saved_models'
torch.save(net.state_dict(), model_dir + model_name)


# Get the weights in the first conv layer, "conv1"
# if necessary, change this to reflect the name of your first conv layer
weights1 = net.convolution_1.weight.data

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

