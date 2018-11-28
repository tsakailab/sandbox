# -*- coding: utf-8 -*-
import numpy as np
from scipy.linalg import eigh


def __min_eigenvec_rf(A):
    v = np.ones(4)
    for i in range(16):
        v = A.dot(v)
    return v[:3]/v[3]

def range_flow_SJB(Z0, Z1, window=np.ones((3,3)), yx=None):
    """
    Range flow estimation

    Parameters
    ----------
    Z0: ndarray, shape (`height`, `width`)
        `height` x `width` depth image of the first frame
    Z1: ndarray, shape (`height`, `width`)
        `height` x `width` depth image of the second frame
    window: ndarray, optional, default np.ones((3,3))
        weighting window.
    yx: ndarray, optional, default None
        Currently not in use, will be used for specifing the positions to compute the range flow.

    Returns
    -------
    rf: ndarray
        3D Range-flow field in shape (`height`, `width`, 3).

    References
    ----------
    H. Spies, Bernd Jahne, and J.L. Barron
    "Range flow estimation"
    Computer Vision and Image Understanding 85, 209–231, 2002.
    doi:10.1006/cviu.2002.0970

    Example
    -------
    >>> rflow = range_flow_SJB(Z0, Z1)
    
    """

    height, width = Z0.shape[:2]
    grad = np.zeros((height, width, 4))
    grad[:,:,:2] = np.gradient(Z0)
    grad[:,:,2] = - np.ones((height, width))
    grad[:,:,3] = Z1 - Z0

    win = np.sqrt(window)
    hwin = win.shape[0]//2, win.shape[1]//2
    rf = np.zeros((height, width, 3))
    #rf = np.zeros((height-2*hwin[0], width-2*hwin[1], 3))
    for p0 in range(hwin[0], height-hwin[0]):
        for p1 in range(hwin[1], width-hwin[1]):
            d = grad[p0-hwin[0]:p0+hwin[0],p1-hwin[1]:p1+hwin[1]]
            d = (d.T * win.T).T
            J = np.tensordot(d, d, ([0,1], [0,1]))
            # 4d eigenvector corresponding to the smallest eigenvalue
            # f = eigh(J)[1][:,0]
            #rf[p0,p1] = f[:3] / f[3]
            rf[p0,p1] = __min_eigenvec_rf(J)

    return rf




if __name__ == '__main__':
    
    import cv2

    cap = cv2.VideoCapture(0)
    ret, im1 = cap.read()
    if ret == None:
        print("Not found camera")
        exit(1)

    height = im1.shape[0]
    width = im1.shape[1]
    sheight = height / 2
    swidth = width / 2

    im1 = cv2.resize(im1, (swidth, sheight))
    im1 = im1[:,::-1]

    im1_gray = cv2.cvtColor(im1,cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(im1)
    hsv[...,2] = 255

    while(1):
        im2 = cap.read()[1]
        im3 = cv2.flip(im2, 1)
	
        im2_gray = cv2.cvtColor(im2,cv2.COLOR_BGR2GRAY)

    	im2_gray = cv2.resize(im2_gray, (swidth, sheight))
        im2_gray = im2_gray[:,::-1]
        flow = np.ndarray((sheight,swidth,2), dtype='float32')

        # pyrScale, levels, winsize, iterations, polyN, polySigma, flags
        #flow = cv2.calcOpticalFlowFarneback(im1_gray,im2_gray, 0.5, 3, 7, 3, 5, 1.1, 0)
        cv2.calcOpticalFlowFarneback(im1_gray,im2_gray, flow, 0.5, 3, 7, 3, 5, 1.1, 0)

        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        hsv[...,0] = ang*180/np.pi/2
        #hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
        hsv[...,1] = mag*4
        rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
        rgb = cv2.resize(rgb, (width, height));

        # image magnification
        #rgb = cv2.resize(rgb, None, fx = 1.3, fy = 1.3)
        #im3 = cv2.resize(im3, None, fx = 1.3, fy = 1.3)
        rgb = cv2.resize(rgb, None, fx = 0.5, fy = 0.5)
        im3 = cv2.resize(im3, None, fx = 0.5, fy = 0.5)
        # 結果表示
        cv2.imshow("Optical Flow",rgb)
        cv2.imshow("camera view",im3)
        im1_gray = im2_gray
        # 任意のキーが押されたら終了
        if cv2.waitKey(30) > 0:
            #cv2.imwrite("optflow.png",rgb)
            cv2.destroyAllWindows()
            cap.release()
            break

