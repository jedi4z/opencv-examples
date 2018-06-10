import numpy as np
import cv2
import glob

# for imagePath in glob.glob('images' + "/*.JPG"):
#     image = cv2.imread(imagePath)

#     cv2.imshow('img', image)
    
    
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cv2 - destroyAllWindows()

image = cv2.imread('images/1.JPG', 0)


cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.imshow('image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
