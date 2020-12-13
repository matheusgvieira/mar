import cv2 as cv;
import numpy as np;
import matplotlib.pyplot as plt;

# Tests
img1 = cv.imread('../images/m1.jpeg', 0);  # Read image
img2 = cv.imread('../images/m2.jpeg', 0);  # Read image
mask_img = cv.bitwise_xor(img1, img2);
# Print images
fig, axes = plt.subplots(1, 3,figsize=(10, 10));
ax = axes.ravel();

ax[0].imshow(img1, cmap=plt.cm.gray);
ax[0].set_title('Image Original');
ax[1].imshow(img2, cmap=plt.cm.gray);
ax[1].set_title('Máscara');
ax[2].imshow(mask_img, cmap=plt.cm.gray);
ax[2].set_title('Output ');

for a in ax:
    a.axis('off')
fig.tight_layout();
plt.show() ;

cv.imwrite('../outputs/mask123.png', mask_img);