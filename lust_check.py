import cv2
import numpy as np
from skimage.metrics import structural_similarity

img1 = cv2.imread('to_check/cropped_0.png')
img2 = cv2.imread('cropped_images/cropped_1.png')

img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
img2 = cv2.resize(img2, dsize=(len(img1[0]), len(img1)), interpolation=cv2.INTER_CUBIC)
print("Dimensione in pixel:", len(img1), len(img1[0]) )
print("Dimensione in pixel:", len(img2), len(img2[0]) )



(p, difference_image) = structural_similarity(img1, img2, full=True)

print("SSIM: {}".format(p))

cv2.imshow("Difference", difference_image)
cv2.waitKey(0)
