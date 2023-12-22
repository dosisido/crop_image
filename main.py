import cv2

import json
from matplotlib import pyplot as plt
import numpy as np
import os
from skimage.metrics import structural_similarity

os.chdir(os.path.dirname(os.path.abspath(__file__)))

FILE = "img.png"
DIR_TO_SAVE = "cropped_images"
DIR_TO_CHECK = "to_check"
INDEX = .95

PRINT = False


img = plt.imread( FILE )
img_new = np.empty( img.shape )
print("Dimensione in pixel:", len(img), len(img[0]) )


image = cv2.imread(FILE)
original_image = cv2.imread(FILE)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
  
# Find Canny edges 
edged = cv2.Canny(gray, 30, 200) 
cv2.waitKey(0) 
  
# Finding Contours 
# Use a copy of the image e.g. edged.copy() 
# since findContours alters the image 
contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
  
if PRINT:
    cv2.imshow('Canny Edges After Contouring', edged) 
    cv2.waitKey(0) 
  
print("Number of Contours found = " + str(len(contours))) 
  
# Draw all contours 
# -1 signifies drawing all contours 
cv2.drawContours(image, contours, -1, (0, 255, 0), 3) 

# Create a directory to save the cropped images
os.makedirs(DIR_TO_SAVE, exist_ok=True)

l = []

for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)


    l.append( {'x':x,'y':y,'w':w,'h':h} )
    print("Subimage coordinates:", x, y, x + w, y + h)

    i = len(l)-1
    cropped_img = original_image[y+1:y+h, x+1:x+w]
    name = f"cropped_{i}.png"
    cv2.imwrite(f"{DIR_TO_SAVE}/{name}", cropped_img)
    l[i]['name_cropped'] = name


if os.path.exists(DIR_TO_CHECK):
    for i in range(len(l)):
        img1 = cv2.imread(f"{DIR_TO_SAVE}/cropped_{i}.png")
        if PRINT:
            cv2.imshow("Cropped Image", img1)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)


        similar_images = []
        for file_path in os.listdir(DIR_TO_CHECK):
            try:

                img2 = cv2.imread(os.path.join(DIR_TO_CHECK, file_path))
                img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
                img2 = cv2.resize(img2, dsize=(len(img1[0]), len(img1)), interpolation=cv2.INTER_CUBIC)

                (similarity, difference_image) = structural_similarity(img1, img2, full=True)
                if similarity > INDEX:
                    similar_images.append(file_path)

            except Exception as e:
                print(f"Error: {e}")
                pass

        if similar_images:
            print("Similar images found:")
            originals = []
            for image_path in similar_images:
                originals.append(image_path)
                print(f"\t{image_path}")
            l[i]['similar_images'] = originals
        else:
            print("No similar images found.")



if PRINT:
    cv2.imshow('Contours', image) 
    cv2.waitKey(0) 
cv2.destroyAllWindows()


with open('output.json', 'w') as f:
    f.write(json.dumps(l, indent=4))