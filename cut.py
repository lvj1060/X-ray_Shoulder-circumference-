import numpy as np
import cv2
import  os
import win32ui
import math
dlg = win32ui.CreateFileDialog(1)  # 1表示打开文件对话框
dlg.SetOFNInitialDir(r'C:\Users\lvj1060\Desktop\data')  # 设置打开文件对话框中的初始显示目录
dlg.DoModal()
filename = dlg.GetPathName()
im = cv2.imread(filename)
# print(im)
# cv2.imwrite(r"C:\Users\lvj1060\Desktop\data\test(2).jpg", im)
red1=im[61:261,147:485]
red2=im[360:580,139:447]
blue1=im[65:266,552:857]
blue2=im[370:609,541:865]
cv2.imshow('bgr_img',im)
cv2.waitKey(0)
cv2.destroyAllWindows()
def cut_red(im):
    img=im.copy()
    h, w = img.shape[:2]
    img = np.ones((h,w),dtype=np.uint8)
    bgr_img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)

    bgr_img[:,:,0] = 0
    bgr_img[:,:,1] = 0
    bgr_img[:,:,2] = 0
    # cv2.imshow('bgr_img',bgr_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    B_channel, G_channel, R_channel = cv2.split(im)
    # cv2.imshow("RedThresh", R_channel)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(R_channel, 160, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    # cv2.imshow("RedThresh", thresh)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    thresh = cv2.dilate(thresh, None, iterations=2)
    # cv2.imshow("RedThresh", thresh)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # thresh = cv2.erode(thresh, None, iterations=1)
    # cv2.imshow("RedThresh", thresh)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    #image, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) 旧版本返回三个参数，新版本返回2个
    binary,contours, hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    contours=np.array(contours)
    # print(contours[1])
    # cv2.floodFill(bgr_img, contours[0], (0, 80), (0, 100, 255), (100, 100, 50), (50, 50, 50), cv2.FLOODFILL_FIXED_RANGE)
    cv2.polylines(bgr_img,[contours[0]],True,(0,0,255),1)
    # cv2.imshow("RedThresh", bgr_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    mask = np.zeros([h+2, w+2], np.uint8)
    cv2.floodFill(bgr_img, mask, (40, 10), (255, 255, 255), (100, 100, 100), (50, 50, 50), cv2.FLOODFILL_FIXED_RANGE)
    # cv2.imshow("RedThresh", bgr_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


    # img = cv2.cvtColor(bgr_img, cv2.COLOR_GRAY2BGR)
    # cv2.imshow("RedThresh", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # num=contours[1][0][0].split(' ', 1 )
    # last=len(contours[1])
    #
    # print((contours[1][0][0][0])*(contours[1][0][0][1]))
    image=cv2.bitwise_or(im, bgr_img, dst=None, mask=None)
    # cv2.imshow("RedThresh", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # print(image.shape)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray=gray.reshape(1,-1)
    # print(len(gray[0]))
    gray=gray[0]
    gray=gray-255
    gray=np.trim_zeros(gray)
    gray=gray+255
    # print(len(gray))
    gray=list(gray)
    num=filter(lambda x: x != 255, gray)
    num=list(num)
    num=np.array(num)
    # print((num))
    # print(gray)
    # print(type(gray))
    # print(len(gray))
    # print(len(gray))
    print("亮度",(np.sum(num))/(255*len(gray)))
    print("密度",math.log(((255*len(num))/np.sum(num)),10))
    return round((np.sum(num))/(255*len(gray)),5),round(math.log(((255*len(num))/np.sum(num)),10),5)
def cut_blue(im):
    img=im.copy()
    h, w = img.shape[:2]
    img = np.ones((h,w),dtype=np.uint8)
    bgr_img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)

    bgr_img[:,:,0] = 0
    bgr_img[:,:,1] = 0
    bgr_img[:,:,2] = 0
    # cv2.imshow('bgr_img',bgr_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    B_channel, G_channel, R_channel = cv2.split(im)
    # cv2.imshow("RedThresh", B_channel)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(B_channel, 150, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    # cv2.imshow("RedThresh", thresh)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    thresh = cv2.dilate(thresh, None, iterations=2)
    # cv2.imshow("RedThresh", thresh)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # thresh = cv2.erode(thresh, None, iterations=1)
    # cv2.imshow("RedThresh", thresh)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    #image, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) 旧版本返回三个参数，新版本返回2个
    binary, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    contours=np.array(contours)
    # print(contours[1])
    # cv2.floodFill(bgr_img, contours[0], (0, 80), (0, 100, 255), (100, 100, 50), (50, 50, 50), cv2.FLOODFILL_FIXED_RANGE)
    cv2.polylines(bgr_img,[contours[0]],True,(0,0,255),1)
    # cv2.imshow("RedThresh", bgr_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    mask = np.zeros([h+2, w+2], np.uint8)
    cv2.floodFill(bgr_img, mask, (40, 10), (255, 255, 255), (100, 100, 100), (50, 50, 50), cv2.FLOODFILL_FIXED_RANGE)
    # cv2.imshow("RedThresh", bgr_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


    # img = cv2.cvtColor(bgr_img, cv2.COLOR_GRAY2BGR)
    # cv2.imshow("RedThresh", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # num=contours[1][0][0].split(' ', 1 )
    # last=len(contours[1])
    #
    # print((contours[1][0][0][0])*(contours[1][0][0][1]))
    image=cv2.bitwise_or(im, bgr_img, dst=None, mask=None)
    # cv2.imshow("RedThresh", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # print(image.shape)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray=gray.reshape(1,-1)
    # print(len(gray[0]))
    gray=gray[0]
    # print(gray)
    gray=gray-255
    gray=np.trim_zeros(gray)
    gray=gray+255
    # print(len(gray))
    gray=list(gray)
    num=filter(lambda x: x != 255, gray)
    num=list(num)
    num=np.array(num)
    # print((num))
    # print(gray)
    # print(type(gray))
    # print(len(gray))
    # print(len(gray))
    print("亮度",(np.sum(num))/(255*len(gray)))
    print("密度",math.log(((255*len(num))/np.sum(num)),10))
    return round((np.sum(num)) / (255 * len(gray)),5), round(math.log(((255 * len(num)) / np.sum(num)), 10),5)

brightness_red1,density_red1=cut_red(red1)
brightness_red2,density_red2=cut_red(red2)
brightness_blue1,density_blue1=cut_blue(blue1)
brightness_blue2,density_blue2=cut_blue(blue2)

cv2.putText(im,'brightness '+str(brightness_red1) , (147, 61), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 1)
cv2.putText(im,'density '+str(density_red1) , (147, 81), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 1)

cv2.putText(im, 'brightness '+str(brightness_red2), (139, 360), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 1)
cv2.putText(im,'density '+str(density_red2) , (139, 380), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 1)

cv2.putText(im, 'brightness '+str(brightness_blue1), (552, 61), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 1)
cv2.putText(im,'density '+str(density_blue1) , (552, 81), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 1)

cv2.putText(im, 'brightness '+str(brightness_blue2), (541, 360), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 1)
cv2.putText(im,'density '+str(density_blue2) , (541, 380), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 1)

cv2.imshow("image", im)
cv2.waitKey(0)
cv2.destroyAllWindows()
os.system("pause")
#
#
# cv2.imwrite(r"C:\Users\lvj1060\Desktop\data\test(1).jpg", image)


# nrootdir = ("./")
# if not os.path.isdir(nrootdir):
# 	os.makedirs(nrootdir)
# cv2.imwrite("123.jpg", newimage)
#
# print(i)




# import cv2
# import numpy as np
#
# if __name__ == '__main__':
#     # read image and convert to gray
#     img = cv2.imread('123123.jpg')
#     # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     B_channel, G_channel, R_channel = cv2.split(img)
# # imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
#     ret,thresh = cv2.threshold(R_channel, 200, 255, cv2.THRESH_BINARY)
#     # threshold the gray image to binarize, and negate it
#     # _,binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
#     # cv2.imshow("RedThresh", thresh)
#     # cv2.waitKey(0)
#     # cv2.destroyAllWindows()
#     thresh = cv2.bitwise_not(thresh)
#
#     # find external contours of all shapes
#     contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#
#     # create a mask for floodfill function, see documentation
#     h,w,_ = img.shape
#     mask = np.zeros((h+2,w+2), np.uint8)
#
#     # determine which contour belongs to a square or rectangle
#     for cnt in contours:
#         poly = cv2.approxPolyDP(cnt, 0.02*cv2.arcLength(cnt,True),True)
#         if len(poly) == 4:
#             # if the contour has 4 vertices then floodfill that contour with black color
#             cnt = np.vstack(cnt).squeeze()
#             _,thresh,_,_ = cv2.floodFill(thresh, mask, tuple(cnt[0]), 0)
#     # convert image back to original color
#     # thresh = cv2.bitwise_not(thresh)
#
#     cv2.imshow('Image', thresh)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
