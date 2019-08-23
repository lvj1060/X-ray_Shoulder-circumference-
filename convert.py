import numpy as np
import cv2
import  os
im = cv2.imread('123123.jpg')
gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
gradX = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
gradY = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=-1)

# subtract the y-gradient from the x-gradient
gradient = cv2.subtract(gradX, gradY)
gradient = cv2.convertScaleAbs(gradient)

# blur and threshold the image
blurred = cv2.blur(gradient, (9, 9))
(_, thresh) = cv2.threshold(blurred, 90, 255, cv2.THRESH_BINARY)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

cv2.imshow("RedThresh", closed)
cv2.waitKey(0)
cv2.destroyAllWindows()
# perform a series of erosions and dilations
closed = cv2.erode(closed, None, iterations=2)
closed = cv2.dilate(closed, None, iterations=2)

contours, hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
contours=np.array(contours)
print(contours[0])
# cv2.floodFill(bgr_img, contours[0], (0, 80), (0, 100, 255), (100, 100, 50), (50, 50, 50), cv2.FLOODFILL_FIXED_RANGE)
cv2.polylines(im,[contours[0]],True,(0,255,0),1)
cv2.imshow("RedThresh", im)
cv2.waitKey(0)
cv2.destroyAllWindows()

