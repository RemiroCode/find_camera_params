from RDD.RDD import build
from RDD.RDD_helper import RDD_helper
from matplotlib import pyplot as plt
import cv2

RDD_model = build(weights='./weights/RDD-v2.pth')
RDD_model.eval()
RDD = RDD_helper(RDD_model)

import cv2
import numpy as np
import os
# Load images
location = "uleila"
LOCATIONS_PATH = os.path.join("C:/Users/dario/work/blender_img_from_coordinates/",location)
source_img = cv2.imread(os.path.join(LOCATIONS_PATH, location + ".jpeg"))  # Source image (reference)
target_img = cv2.imread(os.path.join(LOCATIONS_PATH, location + "_final.png"))  # Destination image
img3 =       cv2.imread(os.path.join(LOCATIONS_PATH, location + "_mask.png",),cv2.IMREAD_UNCHANGED)  # Destination image


mkpts_0, mkpts_1, conf = RDD.match(source_img, target_img)

# Compute homography matrix
H, mask = cv2.findHomography(mkpts_0, mkpts_1, cv2.RANSAC)

print("Homography Matrix:\n", H)
# Apply homography to the third image
h, w = source_img.shape[:2]

warped = cv2.warpPerspective(
    img3,
    H,
    (w, h),
    flags=cv2.INTER_LINEAR,
    borderMode=cv2.BORDER_CONSTANT,
    borderValue=(0, 0, 0, 0)  # Fully transparent background
)

# mask = np.ones((img3.shape[0], img3.shape[1]), dtype=np.uint8) * 255
# warped_mask = cv2.warpPerspective(mask, H, (w, h),borderMode=cv2.BORDER_CONSTANT,borderValue=0)

# warped[:, :, 3] = warped_mask
# Save or display results
cv2.imwrite(os.path.join(LOCATIONS_PATH, location + "_mask_warped.png"), warped)