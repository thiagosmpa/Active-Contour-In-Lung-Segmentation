import numpy as np
import matplotlib.pyplot as plt
import snake as sn
import cv2


def image_prepare():
    original = cv2.imread('chestxray.jpg')
    img2 = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    
    thresValue, th = cv2.threshold(img2, 127, 255, cv2.THRESH_OTSU)
    # cv2.imshow('Otsu Image' + self.title, img2)
    image = cv2.Canny(th, 40, 90)
    return image, original


def right_lung(src):
    x,y = np.mgrid[-4:4:256j, -4:4:256j]
    rad = (x**2 + y**2)**0.5
    tht = np.arctan2(y, x)

    # first shape of the snake
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
ax.imshow(original_image, cmap=plt.cm.gray)

ax.set_xticks([])
ax.set_yticks([])
ax.set_xlim(0,image.shape[1])
ax.set_ylim(image.shape[0],0)
ax.plot(np.r_[leftLungSeg[-1][0], leftLungSeg[-1][0][0]], np.r_[leftLungSeg[-1][1], leftLungSeg[-1][1][0]], c=(1,0,0), lw=2)
ax.plot(np.r_[rightLungSeg[-1][0], rightLungSeg[-1][0][0]], np.r_[rightLungSeg[-1][1], rightLungSeg[-1][1][0]], c=(1,0,0), lw=2)


