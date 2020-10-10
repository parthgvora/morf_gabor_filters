import numpy as np
import gabor 
import cv2 as cv

kwidth = [5, 11, 21, 51, 101]
kheight = [5, 11, 21, 51, 101]
sigma = [2, 4, 6, 8]
theta = [0, 0.25 * np.pi, 0.5 * np.pi, 0.75 * np.pi]
lambd = [20, 10, 5, 3]
gamma = [0.5, 1, 1.5, 2]
psi = [0, 0.25 * np.pi, 0.5 * np.pi, 0.75 * np.pi]

#gb_cv = cv.getGaborKernel((kwidth, kheight), sigma[0], theta[0], lambd[0], gamma[0], psi[0])
#gb = gabor.gabor(kwidth, kheight, sigma[0], theta[0], lambd[0], gamma[0], psi[0])

for kw in kwidth:
    for kh in kheight:
        for s in sigma:
            for th in theta:
                for l in lambd:
                    for g in gamma:
                        for p in psi:
                            gb_cv = cv.getGaborKernel((kw, kh), s, th, l, g, p)
                            gb = gabor.gabor(kw, kh, s, th, l, g, p)

                            gb = np.asarray(gb)
                            gb_cv = np.asarray(gb_cv)

                            if not np.allclose(gb, gb_cv):
                                assert False;
