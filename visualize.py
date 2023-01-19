
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import dataset


#Retrieve each mask and corresponding label from a target image category, mask and mask activation
def generate_masks_labels_from_target(target_categ_image, target_mask_image, target_activation_image, image_height=800,image_width=1088):
    masks_idx = target_activation_image != 0
    mask_set = set()
    non_zero_masks = target_mask_image[masks_idx]
    mask_labels = target_categ_image.flatten()[masks_idx]
    mask_list = []
    label_list = []
    for i,mask_obj in enumerate(non_zero_masks):
        if (torch.sum(mask_obj).item(), mask_labels[i].item()) not in mask_set:
            mask_set.add((torch.sum(mask_obj).item(), mask_labels[i].item()))
            mask_list.append(F.interpolate(mask_obj.unsqueeze(0).unsqueeze(0), (image_height,image_width)).squeeze())
            label_list.append(mask_labels[i].item())
    return mask_list, label_list
    
#Function to generate mask plots corresponding to each level in fpn for one image
def generate_fpn_mask_plots(image, image_idx, target_categ_fpn, target_mask_fpn, target_activation_fpn):
    fpn_masks_list = []
    fpn_label_list = []
    for i in range(len(target_categ_fpn)):
        mask_list, label_list = generate_masks_labels_from_target(target_categ_fpn[i],target_mask_fpn[i],target_activation_fpn[i])
        fpn_masks_list.append(mask_list), fpn_label_list.append(label_list)
    strides = [8, 8, 16, 32, 32]
    fig,ax = plt.subplots(nrows=3,ncols=2, figsize=(8,10))
    for i in range(len(fpn_masks_list)):
        # print(i)
        plot_image = dataset.normalize_np_array_for_plotting(image[image_idx].permute(1,2,0).numpy())
        ax[i//2,i%2].imshow(dataset.apply_mask_to_image(plot_image,fpn_masks_list[i], fpn_label_list[i],.5).astype("float"))
        ax[i//2,i%2].set_title(f"FPN Level {i}", fontsize = 10)
        
    plt.show()

#Function to plot an image tensor
def plottable_image_tensor(image):
    return dataset.normalize_np_array_for_plotting(image.permute(1,2,0).detach().numpy())

#Function to plot the different losses from training
def plot_training_losses(values1, label1 , title, values2 = None, label2 = None, save_path = None):
    xticks = [i for i in range(len(values1))]
    plt.figure(figsize = (12,8))
    plt.plot(values1,label = label1, linestyle = "-", color = "black")
    if values2 and label2:
        plt.plot(values2, label = label2, linestyle = "--", color = "red")
    plt.legend()
    plt.xlabel("Epochs", fontsize = 12)
    plt.ylabel("Loss", fontsize = 12)
    plt.xticks(xticks)
    plt.title(title)
    if save_path:
        plt.savefig(save_path+title.replace(" ",""))
    plt.show()