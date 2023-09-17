import numpy as np
import matplotlib.pyplot as plt
import cv2
import random
-------------------------------------------- new cell
def Balls_dataset(IMAGE):
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]]) #vertical edges

    sobel_y = np.array([[-1, -2, -1],
                        [0, 0, 0],
                        [1, 2, 1]]) #Horizontal edge

    img=cv2.imread(IMAGE+".jpg") 

    rgb_img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) 
    HSV_img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV) 

    #blue color HSV range 
    lower_dark_blue = np.array([110,110,0] )
    upper_dark_blue = np.array([130,255,255])
    mask1=cv2.inRange(HSV_img,lower_dark_blue,upper_dark_blue)
    mask1 = cv2.medianBlur(mask1,17)

    #red color HSV range
    lower_red = np.array([0, 100, 100])
    upper_red = np.array([10, 255, 255])
    red_mask1 = cv2.inRange(HSV_img, lower_red, upper_red)

    lower_red = np.array([160, 100, 100])
    upper_red = np.array([179, 255, 255])
    red_mask2 = cv2.inRange(HSV_img, lower_red, upper_red)

    mask2 = cv2.bitwise_or(red_mask1, red_mask2)
    mask2 = cv2.medianBlur(mask2,17)

    #merged Colors
    maskb=cv2.bitwise_or(mask1,mask2)



    # calculating the aprroximate boundary of the balls in each image (extra)
    kernel=np.ones((15,15),np.uint8)
    closing = cv2.morphologyEx(maskb,cv2.MORPH_CLOSE,kernel)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    erosion=cv2.erode(closing,kernel,iterations=1)
    boundaryimg=closing-erosion

    

    #sobel filtering
    Gx = cv2.filter2D(boundaryimg,-1,sobel_x)
    Gy = cv2.filter2D(boundaryimg,-1,sobel_y)
    gradient_magnitude = np.sqrt(Gx**2 + Gy**2)
    gradient_add1 = cv2.addWeighted(Gx, 1, Gy, 1, 0)
    gradient_magnitude *= 255.0 / gradient_magnitude.max()
    result = cv2.Canny(gradient_add1,100,200)




    result = cv2.Laplacian(result, cv2.CV_8U, ksize=5)
    result=cv2.medianBlur(result,7)

   




    # finfing continuous contour like edges to idetify the balls
    blue_contours, _ = cv2.findContours(mask1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    red_contours, _ = cv2.findContours(mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)



    max_blue = -99999
    for contour in blue_contours:
        if(max_blue < cv2.contourArea(contour)):
            max_blue = cv2.contourArea(contour)
    print(f"maxBlue ={max_blue}")

    max_red = -99999
    for contour in red_contours:
        if(max_red < cv2.contourArea(contour)):
            max_red = cv2.contourArea(contour)
    print(f"maxRed={max_red}")

    

    # Draw circles around blue balls
    for contour in blue_contours:
        (x, y), radius = cv2.minEnclosingCircle(contour)
        print(cv2.contourArea(contour))
        center = (int(x), int(y))
        radius = int(radius)
        if(cv2.contourArea(contour)>max_blue*0.75 and cv2.contourArea(contour)<100000 and cv2.contourArea(contour)>200):
            cv2.circle(rgb_img, center, radius+5, (255, 0, 0), 2)

    



    # Draw circles around red balls
    for contour in red_contours:
        (x, y), radius = cv2.minEnclosingCircle(contour)
        print(cv2.contourArea(contour))
        center = (int(x), int(y))
        radius = int(radius)
        if(cv2.contourArea(contour)>max_red*0.75 and cv2.contourArea(contour)<100000 and cv2.contourArea(contour)>200):
            cv2.circle(rgb_img, center, radius+5, (0, 0, 255), 2)

    

    




    # Display the image with the detected balls
    fig , axarr = plt.subplots(1,5,figsize=(20,10))
    axarr[0].imshow(rgb_img)
    axarr[1].imshow(gray_img,cmap = 'gray')
    axarr[2].imshow(maskb,cmap='gray')
    axarr[3].imshow(result,cmap='gray')
    axarr[4].imshow(gradient_magnitude, cmap="gray")

----------------------------------new cell

Balls_dataset("b_001")

----------------------------------new cell


Balls_dataset("b_002")

----------------------------------new cell

Balls_dataset("b_003")

----------------------------------new cell

Balls_dataset("b_004")

----------------------------------new cell

Balls_dataset("b_005")

----------------------------------new cell

Balls_dataset("b_006")

----------------------------------new cell

Balls_dataset("b_007")

----------------------------------new cell

Balls_dataset("b_008")

----------------------------------new cell

Balls_dataset("b_009")

----------------------------------new cell

Balls_dataset("r_051")

----------------------------------new cell

Balls_dataset("rb_000")

----------------------------------new cell


Balls_dataset("rb_001")

----------------------------------new cell


Balls_dataset("rb_002")

----------------------------------new cell


Balls_dataset("rb_004")

----------------------------------new cell


Balls_dataset("rb_005")

----------------------------------new cell


Balls_dataset("rb_006")

----------------------------------new cell


Balls_dataset("rb_016")

----------------------------------new cell


Balls_dataset("b_000")