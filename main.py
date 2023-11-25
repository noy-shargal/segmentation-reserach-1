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
from segment_anything import  sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
def print_resources_info():
    print("PyTorch version:",torch.__version__)
    print("Torchvision version ",torchvision.__version__)
    print("CUDA is_available:", torch.cuda.is_available())


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

    mask_generator = SamAutomaticMaskGenerator(model=sam, points_per_side=32, pred_iou_thresh=0.9,
                                               stability_score_thresh=0.96, crop_n_layers=1, crop_n_points_downscale_factor=2,
                                               min_mask_region_area=100) # requires openCV to run post processing

    masks = mask_generator.generate(image)
    print("Length of 'masks' is", len(masks))
    

    plt.show()


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
