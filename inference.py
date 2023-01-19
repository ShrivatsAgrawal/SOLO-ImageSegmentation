import torch
import matplotlib.pyplot as plt
import dataset
import torch.nn.functional as F

def points_nms(heat, kernel=2):
    # kernel must be 2
    hmax = F.max_pool2d(
        heat, (kernel, kernel), stride=1, padding=1)
    keep = (hmax[:, :, :-1, :-1] == heat).float()
    return heat * keep

def MatrixNMS(sorted_masks, sorted_scores, method='gauss', gauss_sigma=0.5):
    n = len(sorted_scores)
    sorted_masks = sorted_masks.reshape(n, -1)
    intersection = torch.mm(sorted_masks, sorted_masks.T)
    areas = sorted_masks.sum(dim=1).expand(n, n)
    union = areas + areas.T - intersection
    ious = (intersection / union).triu(diagonal=1)

    ious_cmax = ious.max(0)[0].expand(n, n).T
    if method == 'gauss':
        decay = torch.exp(-(ious ** 2 - ious_cmax ** 2) / gauss_sigma)
    else:
        decay = (1 - ious) / (1 - ious_cmax)
    decay = decay.min(dim=0)[0]
    return sorted_scores * decay
    
# For performing nms and thresholding of categ outputs
def point_nms_and_threshold(categ_fpn, categ_thresh):
    """
    Input Shape:
        categ_fpn: B x s x s x c
    """
    categ_nms = points_nms(categ_fpn.permute(0,3,1,2)).permute(0,2,3,1)
    categ_nms_thresh = torch.where(categ_nms >= categ_thresh, categ_nms, 0.0)
    return categ_nms_thresh

#Stacking together different fpn levels
def stack_fpn_levels(categ_fpn, mask_fpn):
    categ_thresh_all = torch.hstack([categ_fpn_level.flatten(1,2) for categ_fpn_level in categ_fpn])
    masks_all = torch.hstack(mask_fpn)
    return categ_thresh_all, masks_all

# ==========================================
#           Inference Visualizations
# ==========================================
def plot_category_predictions(images, pred_categ, image_idx=0, title = ""):
    """
    Inputs:
        image: B x c x h x w
        pred_categ: len(fpn) x B x s x s
    """
    print(title)
    fig, ax = plt.subplots(5,4, figsize = (12,20))

    for i in range(5):
        # print("Shape:",pred_categ[i].shape)
        ax[i,0].imshow(dataset.normalize_np_array_for_plotting(images[image_idx].detach().permute(1,2,0).numpy()))
        ax[i,0].set_title("Original Image")
        ax[i,1].imshow(pred_categ[i][image_idx][:,:,0])
        ax[i,1].set_title(f"FPN Level {i}, Vehicle")
        ax[i,2].imshow(pred_categ[i][image_idx][:,:,1])
        ax[i,2].set_title(f"FPN Level {i}, Person")
        ax[i,3].imshow(pred_categ[i][image_idx][:,:,2])
        ax[i,3].set_title(f"FPN Level {i}, Animal")

#=========================================================
#           POST - PROCESSING
#=========================================================

#Get activated masks and category grid cells
def get_activations(categ_max_value_image, categ_max_indice_image, masks_image):
    activated_idx = categ_max_value_image > 0
    categ_max_value_image_act = categ_max_value_image[activated_idx]
    categ_max_indice_image_act = categ_max_indice_image[activated_idx]
    masks_image_act = masks_image[activated_idx]
    return categ_max_value_image_act, categ_max_indice_image_act, masks_image_act

def get_postprocessed_masks(SOLO_model_ckpt,categ_max_value_image_act, categ_max_indice_image_act, masks_image_act):
    top_k = SOLO_model_ckpt.postprocess_cfg['pre_NMS_num']
    sort_idx = categ_max_value_image_act.sort(descending=True)[1][:top_k]
    categ_image_score_sorted = categ_max_value_image_act[sort_idx][:top_k]
    categ_image_indice_sorted = categ_max_indice_image_act[sort_idx][:top_k]
    mask_thresh = SOLO_model_ckpt.postprocess_cfg['mask_thresh']
    masks_image_sorted = torch.where(masks_image_act[sort_idx] > mask_thresh, 1.0, 0.0)[:top_k]

    iou_thresh = SOLO_model_ckpt.postprocess_cfg['IoU_thresh'] - 0.1
    nms_idx = MatrixNMS(masks_image_sorted, categ_image_score_sorted) > iou_thresh

    keep_instances = SOLO_model_ckpt.postprocess_cfg['keep_instance']
    masks_image_nms = masks_image_sorted[nms_idx][:keep_instances]
    categ_image_nms = (categ_image_indice_sorted[nms_idx] + 1).type(torch.int8).numpy()[:keep_instances]
    return masks_image_nms, categ_image_nms

#Return the image after performing masking
def return_image_with_masking(SOLO_model_ckpt,masks_image_nms, categ_image_nms, image_instance, alpha = 0.5):
    masks_nms_interpolated = F.interpolate(masks_image_nms.unsqueeze(1), (SOLO_model_ckpt.image_height, SOLO_model_ckpt.image_width)).squeeze(1)
    normalized_image = dataset.normalize_np_array_for_plotting(image_instance.permute(1,2,0).detach().numpy())
    mask_thresh = SOLO_model_ckpt.postprocess_cfg['mask_thresh']
    masks_thresholded = torch.where(masks_nms_interpolated > mask_thresh, 1.0, 0)
    masked_image = dataset.apply_mask_to_image(normalized_image,  masks_thresholded ,categ_image_nms, alpha = alpha)
    return masked_image

#Parent function for performing postprocessing on image and returning masked image
def postprocess_image(categ_max_value_image, categ_max_indice_image, masks_image, image_instance, SOLO_model_ckpt, alpha):
    categ_max_value_image_act, categ_max_indice_image_act, masks_image_act = get_activations(categ_max_value_image, categ_max_indice_image, masks_image)
    masks_image_nms, categ_image_nms = get_postprocessed_masks(SOLO_model_ckpt,categ_max_value_image_act, categ_max_indice_image_act, masks_image_act)
    masked_image = return_image_with_masking(SOLO_model_ckpt, masks_image_nms, categ_image_nms, image_instance , alpha)
    return masked_image

#Function for plotting masked images in a grid
def plot_masked_images(SOLO_model_ckpt, categ_op, mask_op, image):
    categ_thresh = SOLO_model_ckpt.postprocess_cfg['categ_thresh']
    categ_fpn_thresh_op = [point_nms_and_threshold(categ_fpn, categ_thresh) for categ_fpn in categ_op]
    categ_thresh_all, masks_all = stack_fpn_levels(categ_fpn = categ_fpn_thresh_op, mask_fpn = mask_op)
    categ_max_values, categ_max_indices = torch.max(categ_thresh_all, dim = -1)
    ncols = 2
    nrows = int(len(categ_max_values)/ ncols)
    fig, ax = plt.subplots(nrows, ncols, figsize = (ncols * 6,nrows * 4.5))
    for image_idx in range(min(len(categ_max_values),nrows*ncols)):
        # print(image_idx)
        categ_max_value_image = categ_max_values[image_idx]
        categ_max_indice_image = categ_max_indices[image_idx]
        masks_image = masks_all[image_idx]
        image_instance = image[image_idx]
        masked_image = postprocess_image(categ_max_value_image, categ_max_indice_image, masks_image, image_instance, SOLO_model_ckpt, alpha=0.4)
        ax[image_idx//ncols, image_idx%ncols].imshow(masked_image)