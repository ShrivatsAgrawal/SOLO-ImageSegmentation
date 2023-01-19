
import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
import pytorch_lightning.loggers as pl_loggers
import pytorch_lightning.callbacks as pl_callbacks
import matplotlib.patches as patches
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import torchvision
from torchvision import transforms
from dataset import SOLODataset
import dataset
import torch.nn.functional as F


#======================================
#HELPER Functions for generate targets
#======================================

def bbox_in_fpn_level(sqrt_wh, scale_range):
    return scale_range[0] <= sqrt_wh <= scale_range[1]

#Return the center region of bbox according to fpn level and epsilon
def return_scaled_bbox_center(bbox_obj, num_grids, epsilon, image_width=1088, image_height=800):
    width_multiplier = num_grids/image_width
    height_multiplier = num_grids/image_height
    scaled_bbox_center = bbox_obj.copy()
    scaled_bbox_center[[1,3]] *= height_multiplier 
    scaled_bbox_center[[0,2]] *= width_multiplier 
    
    epsilon_shift = (1 - epsilon)/2
    scaled_height_shift = (scaled_bbox_center[3] - scaled_bbox_center[1]) * epsilon_shift
    scaled_width_shift = (scaled_bbox_center[2] - scaled_bbox_center[0]) * epsilon_shift

    scaled_bbox_center[0] += scaled_width_shift
    scaled_bbox_center[1] += scaled_height_shift
    scaled_bbox_center[2] -= scaled_width_shift
    scaled_bbox_center[3] -= scaled_height_shift
    

    scaled_bbox_center = np.clip(scaled_bbox_center,0,num_grids-1)
    # print(scaled_bbox_center)
    scaled_bbox_center[:2] = np.floor(scaled_bbox_center[:2])
    scaled_bbox_center[2:] = np.ceil(scaled_bbox_center[2:])
    scaled_bbox_center = scaled_bbox_center.astype("int8")
    return scaled_bbox_center


def return_channels_from_bbox_center(scaled_bbox_center, grid_size):
    x1,y1,x2,y2 = scaled_bbox_center[0], scaled_bbox_center[1], scaled_bbox_center[2], scaled_bbox_center[3]
    channel_nums = []
    for i in range(x1,x2):
        for j in range(y1,y2):
            channel_num = j*grid_size + i
            channel_nums.append(channel_num)
    return channel_nums

def return_feature_height_width(stride,image_height = 800, image_width = 1088):
    return int(image_height/stride), int(image_width/stride)


#================================================
#    Model
#================================================
class SOLO(pl.LightningModule):
    _default_cfg = {
        'num_classes': 4,
        'in_channels': 256,
        'seg_feat_channels': 256,
        'stacked_convs': 7,
        'strides': [8, 8, 16, 32, 32],
        'scale_ranges': [(1, 96), (48, 192), (96, 384), (192, 768), (384, 2048)],
        'epsilon': 0.2,
        'num_grids': [40, 36, 24, 16, 12],
        'mask_loss_cfg': dict(weight=3),
        'categ_loss_cfg': dict(gamma=2, alpha=0.25, weight=1),
        'postprocess_cfg': dict(categ_thresh=0.2, mask_thresh=0.5, pre_NMS_num=50, keep_instance=5, IoU_thresh=0.5),
        'image_height': 800,
        'image_width' : 1088
    }
    
    def __init__(self, **kwargs):
        super().__init__()
        for k, v in {**self._default_cfg, **kwargs}.items():
            setattr(self, k, v)
        
        pretrained_model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=False, pretrained_backbone=True)
        self.backbone = pretrained_model.backbone
        
        #Category branch
        self.conv1_c= nn.Sequential(
            
            nn.Conv2d(self.in_channels, self.seg_feat_channels, kernel_size=3, stride = 1, padding = 1, bias = False),
            nn.GroupNorm(32,256),
            nn.ReLU())
        self.conv2_c = nn.Sequential(
            nn.Conv2d(self.seg_feat_channels, self.seg_feat_channels, kernel_size=3, stride = 1, padding = 1, bias = False),
            nn.GroupNorm(32,256),
            nn.ReLU())
        self.conv3_c = nn.Sequential(
            nn.Conv2d(self.seg_feat_channels, self.seg_feat_channels, kernel_size=3, stride = 1, padding = 1, bias = False),
            nn.GroupNorm(32,256),
            nn.ReLU())
        self.conv4_c = nn.Sequential(
            nn.Conv2d(self.seg_feat_channels, self.seg_feat_channels, kernel_size=3, stride = 1, padding = 1, bias = False),
            nn.GroupNorm(32,256),
            nn.ReLU())
        self.conv5_c = nn.Sequential(
            nn.Conv2d(self.seg_feat_channels, self.seg_feat_channels, kernel_size=3, stride = 1, padding = 1, bias = False),
            nn.GroupNorm(32,256),
            nn.ReLU())
        self.conv6_c = nn.Sequential(
            nn.Conv2d(self.seg_feat_channels, self.seg_feat_channels, kernel_size=3, stride = 1, padding = 1, bias = False),
            nn.GroupNorm(32,256),
            nn.ReLU())
        self.conv7_c = nn.Sequential(
            nn.Conv2d(self.seg_feat_channels, self.seg_feat_channels, kernel_size=3, stride = 1, padding = 1, bias = False),
            nn.GroupNorm(32,256),
            nn.ReLU())
        self.conv_out_c = nn.Sequential(
            nn.Conv2d(self.seg_feat_channels, self.num_classes-1, kernel_size=3, stride = 1, padding = 1, bias = True),
            nn.Sigmoid())

        self.categ_branch = nn.Sequential(self.conv1_c,self.conv2_c,self.conv3_c,self.conv4_c,self.conv5_c,\
            self.conv6_c,self.conv7_c,self.conv_out_c)
    
    #Mask brach
        self.conv1_m = nn.Sequential(
            nn.Conv2d(self.in_channels+2, self.seg_feat_channels, kernel_size=3, stride = 1, padding = 1, bias = False),
            nn.GroupNorm(32,256),
            nn.ReLU())
        self.conv2_m = nn.Sequential(
            nn.Conv2d(self.seg_feat_channels, self.seg_feat_channels, kernel_size=3, stride = 1, padding = 1, bias = False),
            nn.GroupNorm(32,256),
            nn.ReLU()
        )
        self.conv3_m = nn.Sequential(
            nn.Conv2d(self.seg_feat_channels, self.seg_feat_channels, kernel_size=3, stride = 1, padding = 1, bias = False),
            nn.GroupNorm(32,256),
            nn.ReLU()
        )
        self.conv4_m = nn.Sequential(
            nn.Conv2d(self.seg_feat_channels, self.seg_feat_channels, kernel_size=3, stride = 1, padding = 1, bias = False),
            nn.GroupNorm(32,256),
            nn.ReLU()
        )
        self.conv5_m = nn.Sequential(
            nn.Conv2d(self.seg_feat_channels, self.seg_feat_channels, kernel_size=3, stride = 1, padding = 1, bias = False),
            nn.GroupNorm(32,256),
            nn.ReLU()
        )
        self.conv6_m = nn.Sequential(
            nn.Conv2d(self.seg_feat_channels, self.seg_feat_channels, kernel_size=3, stride = 1, padding = 1, bias = False),
            nn.GroupNorm(32,256),
            nn.ReLU()
        )
        self.conv7_m = nn.Sequential(
            nn.Conv2d(self.seg_feat_channels, self.seg_feat_channels, kernel_size=3, stride = 1, padding = 1, bias = False),
            nn.GroupNorm(32,256),
            nn.ReLU()
        )

        self.conv_out_m_0 = nn.Sequential(
            nn.Conv2d(self.seg_feat_channels, self.num_grids[0]**2, kernel_size=1, stride = 1, padding = 0, bias = True),
            nn.Sigmoid()
        )
        self.conv_out_m_1 = nn.Sequential(
            nn.Conv2d(self.seg_feat_channels, self.num_grids[1]**2, kernel_size=1, stride = 1, padding = 0, bias = True),
            nn.Sigmoid()
        )
        self.conv_out_m_2 = nn.Sequential(
            nn.Conv2d(self.seg_feat_channels, self.num_grids[2]**2, kernel_size=1, stride = 1, padding = 0, bias = True),
            nn.Sigmoid()
        )
        self.conv_out_m_3 = nn.Sequential(
            nn.Conv2d(self.seg_feat_channels, self.num_grids[3]**2, kernel_size=1, stride = 1, padding = 0, bias = True),
            nn.Sigmoid()
        )
        self.conv_out_m_4 = nn.Sequential(
            nn.Conv2d(self.seg_feat_channels, self.num_grids[4]**2, kernel_size=1, stride = 1, padding = 0, bias = True),
            nn.Sigmoid()
        )

        self.final_mask_layer = {0:self.conv_out_m_0, 1:self.conv_out_m_1, 2:self.conv_out_m_2,\
            3:self.conv_out_m_3, 4:self.conv_out_m_4}
            
        self.mask_branch = nn.Sequential(self.conv1_m,self.conv2_m,self.conv3_m,\
            self.conv4_m,self.conv5_m,self.conv6_m,self.conv7_m)
        

    # Forward function should calculate across each level of the feature pyramid network.
    # Input:
    #     images: batch_size number of images
    # Output:
    #     if eval = False
    #         category_predictions: list, len(fpn_levels), each (batch_size, C-1, S, S)
    #         mask_predictions:     list, len(fpn_levels), each (batch_size, S^2, 2*feature_h, 2*feature_w)
    #     if eval==True
    #         category_predictions: list, len(fpn_levels), each (batch_size, S, S, C-1)
    #         / after point_NMS
    #         mask_predictions:     list, len(fpn_levels), each (batch_size, S^2, image_h/4, image_w/4)
    #         / after upsampling

    def forward(self, images, eval=True):
        # you can modify this if you want to train the backbone
        # images = images.to(self.device)
        feature_pyramid = [F.interpolate(v.detach(), \
            (int(self.image_height/self.strides[i]), int(self.image_width/self.strides[i]))\
                ) for i,v in enumerate(self.backbone(images).values())\
                          ] # modified to have strides [8,8,16,32,32]

        #Scaling fp to s x s for category branch
        categ_scaled_fp = [F.interpolate(v,(self.num_grids[i],self.num_grids[i])) for i,v in enumerate(feature_pyramid)]
        
        #Scaling fp to 256+2, h, w for mask branch 
        mask_scaled_fp = [torch.hstack((v,torch.stack(
                                torch.meshgrid(
                                    (torch.linspace(-1,1,v.shape[-2], device=self.device),torch.linspace(-1,1,v.shape[-1], device=self.device))
                                              )
                                                    ).unsqueeze(0).repeat(images.shape[0],1,1,1)\
                              )) for v in feature_pyramid]
                              
        # self.feature_p = feature_pyramid
        #Category branch forward
        categ_image = [self.categ_branch(v) for v in categ_scaled_fp]
        #Mask branch forward
        mask_image = [self.final_mask_layer[i](self.mask_branch(v)) for i,v in enumerate(mask_scaled_fp)]
        
        if not eval:
            mask_image = [F.interpolate(v,(2*v.shape[-2],2*v.shape[-1])) for v in mask_image] #Interpolation to 2h*2w
        else:
            categ_image = [v.permute(0,2,3,1) for v in categ_image]
            mask_image = [F.interpolate(v,(int(self.image_height/4),int(self.image_width/4))) for v in mask_image]
        # self.categ_scaled_fp = categ_image
        # self.masked_fp = mask_image

        return categ_image, mask_image 

    def SOLOLoss(self,pred_categ_batch, pred_mask_batch, trg_categ, trg_mask, trg_mask_act):
        
        #Category Loss
        focal_loss = 0
        for i in range(5): #Level of fpn
            trg_categ_fpn_batch = torch.stack([categ_image[i] for categ_image in trg_categ])
            trg_categ_fpn_batch_one_hot = F.one_hot(trg_categ_fpn_batch, num_classes=4)[:,:,:,1:].permute((0,3,1,2))
            focal_loss_fpn = -torch.mean((self.categ_loss_cfg['alpha'] * torch.pow(1 - pred_categ_batch[i], self.categ_loss_cfg['gamma']) \
                                        * trg_categ_fpn_batch_one_hot * torch.log(pred_categ_batch[i] + 1e-6)) + \
                                            ((1- self.categ_loss_cfg['alpha']) * torch.pow(pred_categ_batch[i], self.categ_loss_cfg['gamma']) \
                                        * (1-trg_categ_fpn_batch_one_hot) * torch.log(1 - pred_categ_batch[i] + 1e-6))
                                                )
            focal_loss += focal_loss_fpn

        #Mask Loss
        dice_loss = 0
        for i in range(5):
            trg_mask_fpn_batch = torch.stack([mask_image[i] for mask_image in trg_mask])
            trg_mask_act_fpn_batch = torch.stack([mask_act_image[i] for mask_act_image in trg_mask_act])
            dice_fpn_num = 2 * torch.sum(pred_mask_batch[i] * trg_mask_fpn_batch, dim = (-1,-2))
            dice_fpn_den = torch.sum(torch.square(pred_mask_batch[i]),  dim = (-1,-2)) + torch.sum(torch.square(trg_mask_fpn_batch),  dim = (-1,-2))
            dice_fpn = 1 - torch.div(dice_fpn_num,dice_fpn_den)
            dice_fpn_outer = trg_mask_act_fpn_batch * dice_fpn
            N_dice = torch.clamp(torch.sum(trg_mask_act_fpn_batch), 1e-6)
            dice_fpn_final = torch.sum(dice_fpn_outer)/N_dice
            dice_loss += dice_fpn_final

        loss = (self.categ_loss_cfg['weight'] * focal_loss) + (self.mask_loss_cfg['weight'] * dice_loss)
        return loss

    # This function build the ground truth tensor for each batch in the training
    # Input:
    #     bounding_boxes:   list, len(batch_size), each (n_object, 4) (x1 y1 x2 y2 system)
    #     labels:           list, len(batch_size), each (n_object, )
    #     masks:            list, len(batch_size), each (n_object, 800, 1088)
    # Output:
        # category_targets: list, len(batch_size), list, len(fpn), (S, S), values are {1, 2, 3}
        # mask_targets:     list, len(batch_size), list, len(fpn), (S^2, 2*feature_h, 2*feature_w)
        # active_masks:     list, len(batch_size), list, len(fpn), (S^2,)
        # / boolean array with positive mask predictions
    def generate_targets(self, bounding_boxes, labels, masks):
        """
        Args:
            bounding_boxes:   list, len(batch_size), each (n_object, 4) (x1 y1 x2 y2 system)
            labels:           list, len(batch_size), each (n_object, )
            masks:            list, len(batch_size), each (n_object, 800, 1088)
        Returns:
            all_target_category: list, len(batch_size), list, len(fpn), (S, S), values are {1, 2, 3}
            all_masks:     list, len(batch_size), list, len(fpn), (S^2, 2*feature_h, 2*feature_w)
            all_mask_activations:     list, len(batch_size), list, len(fpn), (S^2,)
            / boolean array with positive mask predictions
        """
        all_target_category = []
        all_mask_activations = []
        all_masks = []
        # print(bounding_boxes)
        for batch_num,bbox_image in enumerate(bounding_boxes):
            # print(batch_num, bbox_image)
            target_category_image_fpn = []
            target_mask_image_fpn = []
            target_mask_activation_image_fpn = []
            # print()
            for fpn_level,s in enumerate(self.num_grids): #Creating feature maps corresponding to FPN levels
                feature_height, feature_width = return_feature_height_width(self.strides[fpn_level],image_height = 800, image_width = 1088)
                # print(feature_height,feature_width)
                
                #For all fpn levels for each image
                target_category_image_fpn.append(torch.zeros((s,s), dtype=torch.long, device=self.device))
                target_mask_image_fpn.append((torch.zeros(s*s,2*feature_height, 2*feature_width, dtype=torch.float32, device=self.device)))
                target_mask_activation_image_fpn.append((torch.zeros(s*s, dtype=torch.float32, device=self.device)))

            for obj_num,bbox_obj in enumerate(bbox_image): 
                # print(bbox_obj)
                width = bbox_obj[2] - bbox_obj[0]
                height = bbox_obj[3] - bbox_obj[1]
                sqrt_wh = pow(width*height, 0.5)
                label_obj = labels[batch_num][obj_num]
                mask_obj = masks[batch_num][obj_num].unsqueeze(0).unsqueeze(0).type(torch.float32)
                
                
                # print(width, height, sqrt_wh)
                for i in range(len(self.num_grids)):
                    if bbox_in_fpn_level(sqrt_wh=sqrt_wh, scale_range=self.scale_ranges[i]):
                        # print(i)
                        scaled_bbox_center = return_scaled_bbox_center(bbox_obj=bbox_obj, num_grids=self.num_grids[i], epsilon=self.epsilon)
                        #Modifying feature maps corresponding to each image and FPN levels with object locations
                        target_category_image_fpn[i][scaled_bbox_center[1]:scaled_bbox_center[3],scaled_bbox_center[0]:scaled_bbox_center[2]] = label_obj
                        #Interpolating mask to relevant feature map
                        feature_height, feature_width = return_feature_height_width(stride = self.strides[i])
                        mask_obj_interpolate = F.interpolate(mask_obj, (2*feature_height, 2*feature_width)).squeeze()
                        channel_nums = return_channels_from_bbox_center(scaled_bbox_center=scaled_bbox_center, grid_size=self.num_grids[i])
                        target_mask_image_fpn[i][channel_nums] = mask_obj_interpolate
                        target_mask_activation_image_fpn[i][channel_nums] = torch.tensor(1, dtype = torch.float32)  
                    else:
                        continue
            
            all_target_category.append(target_category_image_fpn)
            all_masks.append(target_mask_image_fpn)  
            all_mask_activations.append(target_mask_activation_image_fpn)
        
        return all_target_category, all_masks, all_mask_activations