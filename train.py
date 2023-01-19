from sched import scheduler
import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
import pytorch_lightning.loggers as pl_loggers
import pytorch_lightning.callbacks as pl_callbacks
import matplotlib.patches as patches
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import torch.nn.functional as F
from model import SOLO

class SOLOTrainer(SOLO):
    
    def __init__(self, batch_size=8):
        super().__init__()
        self.train_losses_epoch = []
        self.training_losses = []
        self.validation_losses = []
        self.val_losses_epoch = []
        
        self.focal_loss = []
        self.focal_loss_val = []
        self.focal_loss_epoch = []
        self.focal_loss_epoch_val = []
        self.dice_loss = []
        self.dice_loss_val = []
        self.dice_loss_epoch = []
        self.dice_loss_epoch_val = []
        
        self.batch_size = batch_size
        
    def solo_loss(self,pred_categ_batch, pred_mask_batch, trg_categ, trg_mask, trg_mask_act):
        
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
        
        return focal_loss, dice_loss

    def training_step(self, batch, batch_idx):
        # print("TRAIN STEP:", batch_idx)
        image, label, mask, bbox = batch
        pred_categ_batch,pred_mask_batch = self.forward(image, eval=False)
        trg_categ, trg_mask, trg_mask_act = self.generate_targets(bbox, label, mask)
        focal_loss, dice_loss = self.solo_loss(pred_categ_batch, pred_mask_batch, trg_categ, trg_mask, trg_mask_act)
        train_loss = (self.categ_loss_cfg['weight'] * focal_loss) + (self.mask_loss_cfg['weight'] * dice_loss)
        # logs metrics for each training_step,and the average across the epoch, to the progress bar and logger
        self.log("train_loss", train_loss, prog_bar = True, logger=True, on_step=True, on_epoch=True)#, prog_bar=True, logger=True)
        self.log("focal_loss",focal_loss, prog_bar = True, logger=True, on_step=True, on_epoch=True)
        self.log("dice_loss",dice_loss, prog_bar = True, logger=True, on_step=True, on_epoch=True)
        self.training_losses.append(train_loss.item())
        self.focal_loss.append(focal_loss.item())
        self.dice_loss.append(dice_loss.item())
        # print("BATCH IDX LR:", self.optimizer.param_groups[0]['lr'])
        return train_loss

    def validation_step(self, batch, batch_idx):
        image, label, mask, bbox = batch
        pred_categ_batch,pred_mask_batch = self.forward(image, eval=False)
        trg_categ, trg_mask, trg_mask_act = self.generate_targets(bbox, label, mask)
        focal_loss, dice_loss = self.solo_loss(pred_categ_batch, pred_mask_batch, trg_categ, trg_mask, trg_mask_act)
        val_loss = (self.categ_loss_cfg['weight'] * focal_loss) + (self.mask_loss_cfg['weight'] * dice_loss)
        # logs metrics for each training_step,and the average across the epoch, to the progress bar and logger
        self.log("val_loss", val_loss, prog_bar = True, logger=True, on_step=True, on_epoch=True)#, prog_bar=True, logger=True)
        self.log("val_focal_loss",focal_loss, prog_bar = True, logger=True, on_step=True, on_epoch=True)
        self.log("val_dice_loss",dice_loss, prog_bar = True, logger=True, on_step=True, on_epoch=True)
        self.validation_losses.append(val_loss.item())
        self.focal_loss_val.append(focal_loss.item())
        self.dice_loss_val.append(dice_loss.item())
        # print("BATCH IDX LR:", self.optimizer.param_groups[0]['lr'])
        return val_loss

    def training_epoch_end(self, outputs):
        # print("Train epoch ends:",len(self.train_losses_epoch))
        if outputs:
            self.train_losses_epoch.append(torch.tensor([x['loss'] for x in outputs]).mean().item())
        
        self.focal_loss_epoch.append(np.array(self.focal_loss).mean())
        self.dice_loss_epoch.append(np.array(self.dice_loss).mean())
        self.focal_loss = []
        self.dice_loss = []

    def validation_epoch_end(self, outputs):
        # print("Val epoch")
        if outputs:
            self.val_losses_epoch.append(torch.tensor(outputs).mean().item())
        self.focal_loss_epoch_val.append(np.array(self.focal_loss_val).mean())
        self.dice_loss_epoch_val.append(np.array(self.dice_loss_val).mean())
        self.focal_loss_val = []
        self.dice_loss_val = []
    
    def configure_optimizers(self):
        lr_multiplier = (self.batch_size/16) 
        self.optimizer = torch.optim.SGD(self.parameters(), weight_decay=1e-4, momentum=0.9, lr=1e-2*lr_multiplier)
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[27,33], gamma=0.1)
        scheduler_config = {
            'scheduler' : lr_scheduler,
            'interval' : 'epoch',
        }
        return {"optimizer":self.optimizer, "lr_scheduler":scheduler_config}