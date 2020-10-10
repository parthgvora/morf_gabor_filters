import cv2 as cv
import gabor
import numpy as np
import matplotlib.pyplot as plt

# Goal is to show that opencv and my code are the same

# Gabor filter parameters
kwidth = 51
kheight = 51

sigma = [2, 4, 6, 8]
theta = [0, 0.25 * np.pi, 0.5 * np.pi, 0.75 * np.pi]
lambd = [20, 10, 5, 3]
gamma = [0.5, 1, 1.5, 2]
psi = [0, 0.25 * np.pi, 0.5 * np.pi, 0.75 * np.pi]

# Get openCV gabor kernel
#gb_opencv = cv.getGaborKernel((kwidth, kheight), sigma, theta, lambd, gamma, psi)
#gb_custom = gabor.gabor(kwidth, kheight, sigma, theta, lambd, gamma, psi)

# Get gabor filters
gb_sigma_cv = [cv.getGaborKernel((kwidth, kheight), i, theta[0], lambd[0], gamma[0], psi[0]) for i in sigma]
gb_theta_cv = [cv.getGaborKernel((kwidth, kheight), sigma[0], i, lambd[0], gamma[0], psi[0]) for i in theta]
gb_lambd_cv = [cv.getGaborKernel((kwidth, kheight), sigma[0], theta[0], i, gamma[0], psi[0]) for i in lambd]
gb_gamma_cv = [cv.getGaborKernel((kwidth, kheight), sigma[0], theta[0], lambd[0], i, psi[0]) for i in gamma]
gb_psi_cv   = [cv.getGaborKernel((kwidth, kheight), sigma[0], theta[0], lambd[0], gamma[0], i) for i in psi]

gb_sigma = [gabor.gabor(kwidth, kheight, i, theta[0], lambd[0], gamma[0], psi[0]) for i in sigma]
gb_theta = [gabor.gabor(kwidth, kheight, sigma[0], i, lambd[0], gamma[0], psi[0]) for i in theta]
gb_lambd = [gabor.gabor(kwidth, kheight, sigma[0], theta[0], i, gamma[0], psi[0]) for i in lambd]
gb_gamma = [gabor.gabor(kwidth, kheight, sigma[0], theta[0], lambd[0], i, psi[0]) for i in gamma]
gb_psi = [gabor.gabor(kwidth, kheight, sigma[0], theta[0], lambd[0], gamma[0], i) for i in psi]


# Compare gabor filters
def compare(cv, custom, title_x, fn):
    fig, axes = plt.subplots(2, 4, figsize=(10, 10))
    
    axes[0, 0].set_title(title_x[0])
    axes[0, 1].set_title(title_x[1])
    axes[0, 2].set_title(title_x[2])
    axes[0, 3].set_title(title_x[3])

    axes[0, 0].set_ylabel("OPENCV")
    axes[1, 0].set_ylabel("Custom")

    for i in range(0, 4):
        axes[0, i].imshow(cv[i])
        axes[1, i].imshow(custom[i])

    plt.savefig(fn)


title = ["sigma = 2", "sigma = 4", "sigma = 6", "sigma = 8"]
fn = "sigma.png"
compare(gb_sigma_cv, gb_sigma, title, fn)

title = ["theta = 0", "theta = pi/4", "theta = pi/2", "theta = 3*pi/2"]
fn = "theta.png"
compare(gb_theta_cv, gb_theta, title, fn)

title = ["lambd = 20", "lambd = 10", "lambd = 5", "lambd = 3"]
fn = "lambd.png"
compare(gb_lambd_cv, gb_lambd, title, fn)

title = ["gamma = 0.5", "gamma = 1", "gamma = 1.5", "gamma = 2"]
fn = "gamma.png"
compare(gb_gamma_cv, gb_gamma, title, fn)

title = ["psi = 0", "psi = pi/4", "psi = pi/2", "psi = 3*pi/2"]
fn = "psi.png"
compare(gb_psi_cv, gb_psi, title, fn)



