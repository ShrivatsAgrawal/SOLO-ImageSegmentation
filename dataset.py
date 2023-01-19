
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import h5py
from torchvision import transforms
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

#============================
#    Loading Data
#============================

#Function to load np files
def load_data_np(file_name):
    return np.load(file_name, allow_pickle=True, encoding='latin1')

#Function to load h5py files
def load_data_h5py(file_name):
    h5_file = h5py.File(file_name,'r+')  
    return np.array(h5_file['data'])

#Function to unflatten the masks according to the bounding boxes
def unflatten_masks(masks, bboxes):
    total = 0
    batched_masks = []
    for i in range(len(bboxes)):
        num_bboxes = len(bboxes[i])
        batched_masks.append(masks[total:total + num_bboxes])
        total += num_bboxes
    return np.array(batched_masks, dtype=object)

#Function to collate the batches of variable length into a tuple
def collate_fn(batch):
    images, labels, masks, bounding_boxes = list(zip(*batch))
    return torch.stack(images), labels, masks, bounding_boxes
#==========================
# Transforms and dataset
#==========================
resize_width = 1066
resize_height = 800
pad_h = 800
pad_w = 1088
orig_height = 300
orig_width = 400

width_pad = int((pad_w - resize_width)/ 2)
height_pad = int((pad_h - resize_height)/2)

def scale_y(y, orig_height, resize_height, height_pad):
    y = y * (resize_height/orig_height) + height_pad
    return y

def scale_x(x,orig_width, resize_width, width_pad):
    x = x * (resize_width/orig_width) + width_pad
    return x

transform = transforms.Compose(
    [
    transforms.ToTensor(),
    transforms.Resize((resize_height,resize_width)),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    transforms.Pad([width_pad,height_pad])
    ]
)

mask_transform = transforms.Compose(
    [
    transforms.ToTensor(),
    transforms.Resize((resize_height,resize_width)),
    transforms.Pad([width_pad,height_pad])
    ]
)

class SOLODataset(Dataset):
    def __init__(self, tensors, transform=transform, mask_transform=mask_transform):
        self.tensors = tensors
        self.transform = transform
        self.mask_transform = mask_transform
    
    def __len__(self):
        return self.tensors[0].shape[0]

    def __getitem__(self, index):
        image = self.tensors[0][index]
        bbox = self.tensors[1][index]
        label = self.tensors[2][index]
        mask = self.tensors[3][index]
        if self.transform:
            image = self.transform(image)
        
        all_masks = []
        if self.mask_transform:
            for mask_obj in mask:
                all_masks.append(torch.squeeze(mask_transform(mask_obj)))
            mask = torch.stack(all_masks)

        mod_bbox = []
        for box in bbox.copy():
            box[1], box[3] = scale_y(box[1], orig_height, resize_height, height_pad), scale_y(box[3], orig_height, resize_height, height_pad)
            box[0], box[2] = scale_x(box[0],orig_width, resize_width, width_pad), scale_x(box[2],orig_width, resize_width, width_pad)
            mod_bbox.append([box[0],box[1],box[2],box[3]])
        mod_bbox = np.array(mod_bbox)
        
        return image, label, mask, mod_bbox

#=================================
#     Visualizations
#=================================
#Plotting Image With Boxes
def plot_image_with_boxes(image,label, bbox):
  
  """Plotting bounding boxes on images."""

  fig, ax = plt.subplots(figsize = (8,8))
  
  mapping = {1:"b",2:"g",3:"r"}
  label_mapping = {1 : "Vehicle", 2 : "Person", 3 : "Animal"}
  ax.imshow(image)
  for i,box in enumerate(bbox):
    x1 = box[0]
    y1 = box[1]
    x2 = box[2]
    y2 = box[3]

    width = x2-x1
    height = y2-y1
    # print(height)
    rect = patches.Rectangle((x1, y1), width, height, linewidth=3, edgecolor=mapping[label[i]], facecolor='none')
    
    props = dict(boxstyle='round', facecolor=mapping[label[i]], alpha=0.5)
    # place a text box in upper left in axes coords
    ax.text(x1, y1-4, label_mapping[label[i]], fontsize=10, color="w",
        verticalalignment='top', bbox=props)
    ax.add_patch(rect)
  plt.show()

def apply_mask_to_image(image, masks, labels, alpha=.4):
    """Applying masks on images"""
    image = image.copy()
    label_dict = {1:2,2:1,3:0}
    for i, mask in enumerate(masks):
        label = labels[i] 
        for channel in range(3):
            label = label_dict[labels[i]]
            if label == channel:
                color_factor = 1
            else:
                color_factor = 0
            image[:, :, channel] = np.where(mask > 0,
                                    image[:, :, channel] *
                                    (1 - alpha) + alpha * color_factor ,
                                    image[:, :, channel])
    return image

#To normalize a numpy array for plotting with imshow
def normalize_np_array_for_plotting(np_array):
    return (np_array - np.min(np_array))/ (np.max(np_array) - np.min(np_array))
#==============================================