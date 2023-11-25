# This is the first experiment towards deep segmentation improvements
# with super pixels hierarchy tree post-processing.

# ------------------------------------------------------------------------------------------
# 1. Create image segmentation named SEG-1 with SAM and a prompt of one point 's coordinates.
# 2. Create modified image segmentation SEG-2 from SEG-1 with the help of
#    the image's super pixels hierarchy tree and an algorithm,
#    from the paper "Assessing hierarchies by their consistent segmentations", which finds the best (IoU wise)
#    nodes from the tree to match SEG-1, this nodes create SEG-2 .
#    SEG-2 is created multiple times with different K values, where K is the maximum number of nodes,
#    The best SEG-2 is chosen ad FNL-SEG-2
# 3. compare SEG-1 and FNL-SEG-2, hopping FNL-SEG-2 is better.
# ------------------------------------------------------------------------------------------

import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
import cv2
import sys
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

def calculate_iou(mask1, mask2):
    """
    Calculate Intersection over Union (IoU) between two binary masks.

    Parameters:
    - mask1: NumPy array, first binary mask (True/False).
    - mask2: NumPy array, second binary mask (True/False).

    Returns:
    - IoU: Intersection over Union score.
    """
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)

    IoU = np.sum(intersection) / np.sum(union)
    return IoU

def print_resources_info():
    print("PyTorch version:",torch.__version__)
    print("Torchvision version ",torchvision.__version__)
    print("CUDA is_available:", torch.cuda.is_available())


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    sys.path.append("..")

    print_resources_info()
    image = cv2.imread('test-images/cheeky_penguin.png')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10,10))
    plt.imshow(image)
    plt.axis('off')
    device = "cuda"
    model_type = "vit_h"
    model_checkpoints = "../sam-checkpoints/sam_vit_h_4b8939.pth"
    sam = sam_model_registry[model_type](model_checkpoints)
    sam.to(device=device)

    # mask_generator = SamAutomaticMaskGenerator(model=sam, points_per_side=32, pred_iou_thresh=0.9,
    #                                            stability_score_thresh=0.96, crop_n_layers=1, crop_n_points_downscale_factor=2,
    #                                            min_mask_region_area=100) # requires openCV to run post processing
    #
    # masks = mask_generator.generate(image)
    # print("Length of 'masks' is", len(masks))
    predictor = SamPredictor(sam)
    predictor.set_image(image)
    input_point = np.array([[150, 222]])
    input_label = np.array([1])

    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    show_points(input_point, input_label, plt.gca())
    plt.axis('on')
    plt.show()

    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True,
    )

    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        IoU = calculate_iou(mask, mask)
        show_mask(mask, plt.gca())
        show_points(input_point, input_label, plt.gca())
        plt.title(f"Mask {i + 1}, Score: {score:.3f} - IoU: {IoU}", fontsize=18)
        plt.axis('off')
        plt.show()


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
