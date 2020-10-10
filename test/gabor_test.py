import cv2 as cv
import gabor
import numpy as np
import matplotlib.pyplot as plt

# Goal is to show that opencv and my code are the same

# Gabor filter parameters
kwidth = 51
kheight = 51

sigma = 2
theta = np.pi/4
lambd = 20
gamma = 0.5
psi = 0

# Get openCV gabor kernel
gb_opencv = cv.getGaborKernel((kwidth, kheight), sigma, theta, lambd, gamma, psi)
gb_custom = gabor.gabor(kwidth, kheight, sigma, theta, lambd, gamma, psi)

plt.imshow(gb_opencv)
plt.show()

plt.imshow(gb_custom)
plt.show()
