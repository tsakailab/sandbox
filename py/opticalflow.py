# -*- coding: utf-8 -*-
import cv2
import numpy as np

def main():
    # カメラのキャプチャ
    cap = cv2.VideoCapture(0)
    # 画像取得
    im1 = cap.read()[1]
    # エラー処理
    if im1 is None:
        print("Not found camera")
        exit(1)

    height = im1.shape[0]
    width = im1.shape[1]
    sheight = height / 2
    swidth = width / 2

    im1 = cv2.resize(im1, (swidth, sheight))
    im1 = im1[:,::-1]

    # 画像をグレースケール変換
    im1_gray = cv2.cvtColor(im1,cv2.COLOR_BGR2GRAY)

    # 画像をコピーしてhsv色空間に変換
    hsv = np.zeros_like(im1)
    hsv[...,2] = 255

    while(1):
        # 画像取得
        im2 = cap.read()[1]
        # 画像をグレースケール変換
        im2_gray = cv2.cvtColor(im2,cv2.COLOR_BGR2GRAY)

    	im2_gray = cv2.resize(im2_gray, (swidth, sheight))
        im2_gray = im2_gray[:,::-1]

        # 連続する2枚の画像からオプティカルフローを計算
        # pyrScale, levels, winsize, iterations, polyN, polySigma, flags
        flow = cv2.calcOpticalFlowFarneback(im1_gray,im2_gray, flow=None,
                                            pyr_scale=0.5, levels=3, winsize=7,
                                            iterations=3, poly_n=5, poly_sigma=1.1, flags=0)
        # 2次元ベクトル(フロー)の角度と大きさを計算
        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        # 計算した角度と大きさによって色を決定
        hsv[...,0] = ang*180/np.pi/2
        #hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
        hsv[...,1] = mag*4
        rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
        rgb = cv2.resize(rgb, (width, height));
        # 結果表示
        cv2.imshow("Optical Flow",rgb)
        im1_gray = im2_gray
        # 任意のキーが押されたら終了
        if cv2.waitKey(30) > 0:
            cv2.imwrite("optflow.png",rgb)
            cap.release()
            cv2.destroyAllWindows()
            break


if __name__ == '__main__':
    main()

