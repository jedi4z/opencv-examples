import numpy as np
import cv2

img = cv2.imread('images/6.JPG')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)

median = cv2.medianBlur(blur, 5)
edges = cv2.Canny(median, 100, 100)

_, contours, _ = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

img = cv2.drawContours(img, contours, -1, (0, 255, 0), 3)

img = cv2.resize(img, (0, 0), fx=0.2, fy=0.2)

cv2.imshow('edges', img)

cv2.waitKey(0)
cv2.destroyAllWindows()
