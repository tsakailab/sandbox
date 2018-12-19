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

def __min_eigenvec_rf(A,itr=16,confidence=1.0):
    v = np.ones(4)
    trA = np.trace(A)
    for i in range(itr):
        v *= 1 / v[3]
        v = -A.dot(v)
    if abs(v[3]) < trA*confidence:
        return np.zeros(3)
    return v[:3]/v[3]


def range_flow_SJB(Z0, Z1, window=np.ones((3,3)), yx=None, scale=1., confidence=1.0):
    """
    Range flow estimation

    Parameters
    ----------
    Z0: ndarray, shape (`height`, `width`)
        depth image of the first frame
    Z1: ndarray, shape (`height`, `width`)
        depth image of the second frame
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
    >>> rflow = range_flow_SJB(Z0, Z1)
    
    """

    height, width = Z0.shape[:2]
    grad = np.zeros((height, width, 4))
    grad[:,:,0], grad[:,:,1] = np.gradient(Z0.astype(float))
    grad[:,:,2] = - np.ones((height, width))
    grad[:,:,3] = Z1.astype(float) - Z0.astype(float)
    #vfshow(grad[...,:2], scale=0.9, every=5)

    win = np.sqrt(window)
    hwin = win.shape[0]//2, win.shape[1]//2
    rf = np.zeros((height, width, 3))
    for p0 in range(hwin[0], height-hwin[0]):
        for p1 in range(hwin[1], width-hwin[1]):
            d = grad[p0-hwin[0]:p0+hwin[0]+1,p1-hwin[1]:p1+hwin[1]+1]
            d = (d.T * win.T).T
            J = np.tensordot(d, d, ([0,1], [0,1]))
            # 4d eigenvector corresponding to the smallest eigenvalue
            rf[p0,p1] = __min_eigenvec_rf(J, itr=8, confidence=confidence) * scale
            #f = eigh(J)[1][:,0]
            #if f[3] == 0: f[3] = 1.
            #rf[p0,p1] = f[:3] / f[3] * scale
            #eival, eivec = eigh(J)
            #eivec, eival = np.linalg.svd(J)[:2]
            #eivec = eivec[:,::-1]
            #eival = eival[::-1]
            #if eival[3]/eival[0] > 0.1:
            #    rf[p0,p1] = eivec[:,0][:3] / eivec[:,0][3] * scale
    return rf, grad, J



if __name__ == '__main__':
    
    import numpy as np
    from PIL import Image
    from scipy import ndimage
    import matplotlib.pyplot as plt
    f0 = 'crop_depth/resize_depth0620.tiff'
    f1 = 'crop_depth/resize_depth0630.tiff'
    im0 = np.asarray(Image.open(f0), dtype=np.float32)
    im1 = np.asarray(Image.open(f1), dtype=np.float32)

    im0 = ndimage.gaussian_filter(im0, sigma=3.0)
    im1 = ndimage.gaussian_filter(im1, sigma=3.0)

    flow, g, J = range_flow_SJB(im0, im1, window=np.ones((5,5)), confidence=0.999)
    plt.figure()
    vfshow(flow[...,:2], scale=0.01, every=3)

    mag, ang = _cartToPolar(flow[...,0], flow[...,1])
    hsv = np.zeros((im0.shape[0],im1.shape[1], 3), dtype=np.uint8)
    hsv[...,0] = 255 * ang * 0.5 / np.pi
    #hsv[...,1] = np.minimum(255,128 * mag / np.median(mag.ravel()))
    hsv[...,1] = np.minimum(255,255. * mag)
    hsv[...,2] = 255
    #hsv[...,2] = cv2.normalize(flow[...,2],None,0,255,cv2.NORM_MINMAX)
    #bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
    rgb = Image.fromarray(hsv, mode="HSV").convert("RGB")#.show()
    plt.imshow(np.array(rgb))
    
    plt.figure()
    plt.imshow(np.concatenate((im0,im1)), cmap='gray', interpolation='nearest')
    plt.figure()
    plt.imshow(im1-im0, cmap='seismic', interpolation='nearest', vmin=-30, vmax=30)
    plt.colorbar()