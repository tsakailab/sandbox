# -*- coding: utf-8 -*-
import numpy as np
from scipy.linalg import eigh
from matplotlib.pyplot import quiver

def _cartToPolar(y, x):
    mag = np.hypot(y,x) #np.sqrt(y*y+x*x)
    ang = np.arctan2(y, x)
    ang[ang < 0.] += 2.0 * np.pi
    return mag, ang

def vfshow(flow, every=1, scale=1.):
    h, w = flow.shape[:2]
    X, Y = np.meshgrid(np.arange(0,w,every), np.arange(0,h,every))
    ax = quiver(X, Y, flow[::every,::every,1], -flow[::every,::every,0], scale_units='xy',scale=scale)
    return ax

def ___min_eigenvec_rf(A,itr=16):
    v = np.ones(4)
    #A = -AA/np.trace(AA)
    for i in range(itr):
        v = -A.dot(v)
        v *= 1./abs(v[3])
        #v *= 1./np.max(np.abs(v))
    return v[:3]/v[3]


def __min_eigenvec_rf(A,itr=16,confidence=1.0):
    v = np.ones(4)
    trA = np.trace(A)
    for i in range(itr):
        v *= 1 / v[3]
        v = -A.dot(v)
    if abs(v[3]) < trA*confidence:
        return np.zeros(3)
    return v[:3]/v[3]



def range_flow_SJBs(Z, window=np.ones((3,3)), yx=None, scale=1., confidence=1.0, itr=16):
    """
    Range flow estimation

    Parameters
    ----------
    Z: ndarray, shape (`nframes`, `height`, `width`)
        sequence of depth images
    window: ndarray, optional, default np.ones((3,3))
        weighting window.
    yx: ndarray, optional, default None
        Currently not in use, will be used for specifing the positions to compute the range flows.
    scale: float, optional
        scale factor.

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
    >>> rflow = range_flow_SJBs(Z)
    
    """

    nframes, height, width = Z.shape
    grad = np.zeros((nframes-1, height, width, 4))
    for t in range(nframes-1):
        grad[t,:,:,0], grad[t,:,:,1] = np.gradient(Z[t,:].astype(float))
        grad[t,:,:,2] = - np.ones((height, width))
        grad[t,:,:,3] = Z[t+1,:].astype(float) - Z[t,:].astype(float)

    win = np.sqrt(window)
    hwin = win.shape[0]//2, win.shape[1]//2
    rf = np.zeros((height, width, 3))
    for p0 in range(hwin[0], height-hwin[0]):
        for p1 in range(hwin[1], width-hwin[1]):
            J = np.zeros((4,4))
            for t in range(nframes-1):
                d = grad[t,p0-hwin[0]:p0+hwin[0]+1,p1-hwin[1]:p1+hwin[1]+1]
                d = (d.T * win.T).T
                J += np.tensordot(d, d, ([0,1], [0,1]))
            # 4d eigenvector corresponding to the smallest eigenvalue
            rf[p0,p1] = __min_eigenvec_rf(J/(nframes-1), itr=itr, confidence=confidence) * scale
            #f = eigh(J)[1][:,0]
            #if f[3] == 0: f[3] = 1.
            #rf[p0,p1] = f[:3] / f[3] * scale
            #eival, eivec = eigh(J)
            #eivec, eival = np.linalg.svd(J)[:2]
            #eivec = eivec[:,::-1]
            #eival = eival[::-1]
            #if eival[3]/eival[0] > 0.001:
            #    rf[p0,p1] = eivec[:,0][:3] / eivec[:,0][3] * scale
    return rf, grad, J



if __name__ == '__main__':

    import numpy as np
    import glob
    from PIL import Image
    from scipy import ndimage
    import matplotlib.pyplot as plt
    #from time import sleep

    resize_depthXXXX_tiff = glob.glob('crop_depth/resize_depth*.tiff')
    imshape = np.asarray(Image.open(resize_depthXXXX_tiff[0]), dtype=np.float32).shape

    plt.figure()
    ax = plt.axes()

    fbegin = 600
    fend = 700

    #every = 4
    #resize_depthXXXX_tiff = resize_depthXXXX_tiff[fbegin:fend:every]
    #for f in zip(resize_depthXXXX_tiff[0:-4+1],
    #             resize_depthXXXX_tiff[1:-4+2],
    #             resize_depthXXXX_tiff[2:-4+3],
    #             resize_depthXXXX_tiff[3:]):

    every = 3
    resize_depthXXXX_tiff = resize_depthXXXX_tiff[fbegin:fend:every]
    for f in zip(resize_depthXXXX_tiff[0:-8+1],
                 resize_depthXXXX_tiff[1:-8+2],
                 resize_depthXXXX_tiff[2:-8+3],
                 resize_depthXXXX_tiff[3:-8+4],
                 resize_depthXXXX_tiff[4:-8+5],
                 resize_depthXXXX_tiff[5:-8+6],
                 resize_depthXXXX_tiff[6:-8+7],
                 resize_depthXXXX_tiff[7:]):

        im = np.empty((len(f),imshape[0],imshape[1]), dtype=np.float32)
        for t in range(len(f)):
            im[t] = np.asarray(Image.open(f[t]), dtype=np.float32)
            im[t] = ndimage.gaussian_filter(im[t], sigma=3.0)

        flow, g, J = range_flow_SJBs(im, window=np.ones((5,5)), confidence=0.7, itr=16)
        ax.clear()
        vfshow(flow[...,:2], scale=0.1, every=3)

        mag, ang = _cartToPolar(flow[...,0], flow[...,1])
        hsv = np.zeros((imshape[0],imshape[1], 3), dtype=np.uint8)
        hsv[...,0] = 255 * ang * 0.5 / np.pi
        #hsv[...,1] = np.minimum(255,128 * mag / np.median(mag.ravel()))
        hsv[...,1] = np.minimum(255,255. * mag)
        hsv[...,2] = 255
        #hsv[...,2] = cv2.normalize(flow[...,2],None,0,255,cv2.NORM_MINMAX)
        #bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
        rgb = Image.fromarray(hsv, mode="HSV").convert("RGB")#.show()
        plt.imshow(np.array(rgb))
        plt.pause(0.01)

    
    #plt.figure()
    #plt.imshow(np.concatenate((im0,im1)), cmap='gray', interpolation='nearest')
    #plt.figure()
    #plt.imshow(im1-im0, cmap='seismic', interpolation='nearest', vmin=-30, vmax=30)
    #plt.colorbar()