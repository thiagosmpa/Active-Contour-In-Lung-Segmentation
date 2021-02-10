import numpy as np
import matplotlib.pyplot as plt
import snake as sn
import cv2


def image_prepare():
    original = cv2.imread('chestxray.jpg')
    # Convert the image to gray scale
    grayImg = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    # Use the gaussian / median filter in order to remove noise
    blured = cv2.GaussianBlur(grayImg,(3,3),0)
    
    _, th = cv2.threshold(blured, 127, 255, cv2.THRESH_OTSU)
    cannyEdge = cv2.Canny(th, 40, 90)
    
    return cannyEdge, original


def right_lung(src):
    x,y = np.mgrid[-4:4:256j, -4:4:256j]
    rad = (x**2 + y**2)**0.5
    tht = np.arctan2(y, x)

    # first shape of the snake as an elipse
    t = np.arange(0, 2*np.pi, 0.1)
    x = 500+100*np.cos(t)
    y = 450+280*np.sin(t)
    
    alpha = 0.0001
    beta  = 0.001
    gamma = 100
    iterations = 500
    
    # fx and fy are callable functions
    fx, fy = sn.create_external_edge_force_gradients_from_img(src, sigma=10 )
    
    snakes = sn.iterate_snake(
        x = x,
        y = y,
        a = alpha,
        b = beta,
        fx = fx,
        fy = fy,
        gamma = gamma,
        n_iters = iterations,
        return_all = True
    )
    return snakes



def left_lung(src):
    x,y = np.mgrid[-4:4:256j, -4:4:256j]
    rad = (x**2 + y**2)**0.5
    tht = np.arctan2(y, x)
    
    # first shape of the snake
    t = np.arange(0, 2*np.pi, 0.1)
    x = 200+120*np.cos(t)
    y = 450+280*np.sin(t)
    
    alpha = 0.0005
    beta  = 0.001
    gamma = 100
    iterations = 500
    
    # fx and fy are callable functions
    fx, fy = sn.create_external_edge_force_gradients_from_img(src, sigma=10 )
    
    snakes = sn.iterate_snake(
        x = x,
        y = y,
        a = alpha,
        b = beta,
        fx = fx,
        fy = fy,
        gamma = gamma,
        n_iters = iterations,
        return_all = True
    )
    return snakes
    

image, original_image = image_prepare()
leftLungSeg = left_lung(image)
rightLungSeg = right_lung(image)


# plot images
fig = plt.figure()
ax  = fig.add_subplot(111)
# Plot the original image
ax.imshow(original_image, cmap=plt.cm.gray)

ax.set_xticks([])
ax.set_yticks([])
ax.set_xlim(0,image.shape[1])
ax.set_ylim(image.shape[0],0)
# Draw the Lung Segmentations in red
# ax.plot(np.r_[leftLungSeg[-1][0], leftLungSeg[-1][0][0]], np.r_[leftLungSeg[-1][1], leftLungSeg[-1][1][0]], c=(1,0,0), lw=2)

contourLeft_x = np.around(np.r_[leftLungSeg[-1][0], leftLungSeg[-1][0][0]])
contourLeft_y = np.around(np.r_[leftLungSeg[-1][1], leftLungSeg[-1][1][0]])
# contourLeft = tuple(map(list, zip(contourLeft_y, contourLeft_x)))

contourRight_x = np.r_[rightLungSeg[-1][0], rightLungSeg[-1][0][0]]
contourRight_y = np.r_[rightLungSeg[-1][1], rightLungSeg[-1][1][0]]

ax.plot(contourLeft_x, contourLeft_y, c=(1,0,0), lw=2)
ax.plot(contourRight_x, contourRight_y, c=(1,0,0), lw=2)

# Draw the first snake shapes in green
# ax.plot(np.r_[x_r,x_r[0]], np.r_[y_r,y_r[0]], c=(0,1,0), lw=2)
# ax.plot(np.r_[x_l,x_l[0]], np.r_[y_l,y_l[0]], c=(0,1,0), lw=2)

