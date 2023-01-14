# -*- coding: utf-8 -*-

# 画像のしきい値処理
#   - http://labs.eecs.tottori-u.ac.jp/sd/Member/oyamada/OpenCV/html/py_tutorials/py_imgproc/py_thresholding/py_thresholding.html#thresholding

# モルフォロジー変換
#   - http://labs.eecs.tottori-u.ac.jp/sd/Member/oyamada/OpenCV/html/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html

# ハフ変換
#   - http://labs.eecs.tottori-u.ac.jp/sd/Member/oyamada/OpenCV/html/py_tutorials/py_imgproc/py_houghlines/py_houghlines.html#hough-lines 
#   - http://www.allisone.co.jp/html/Notes/image/Hough/index.html

import os
import sys

import cv2
import numpy as np

NumpyImage = np.ndarray


class ImageProcessor:
    def __init__(self):
        pass

    @staticmethod
    def load(path:str) -> NumpyImage:
        """ Load an image as a type of BGR """
        if not os.path.isfile(path):
            raise FileNotFoundError
        return cv2.imread(path)

    @staticmethod
    def bgr2rgb(image_bgr:NumpyImage) -> NumpyImage:
        return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    @staticmethod
    def bgr2gray(image_bgr:NumpyImage) -> NumpyImage:
        return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def inverse(image_binary:NumpyImage) -> NumpyImage:
        return 255 - image_binary

    def binarize(self, image_gray:NumpyImage, in_shadow=True) -> NumpyImage:
        if in_shadow:
            # 画像全体に影や照明のムラがある場合 -> local binarize
            return cv2.adaptiveThreshold(
                image_gray,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C, # 閾値計算のタイプ -> ガウス分布による重み付けを行った平均値を採用
                cv2.THRESH_BINARY,              # 閾値処理の種類
                51,   # 閾値計算に使用する近傍領域のサイズ． 3以上の奇数
                21,   # 計算された閾値から引く定数
            )
        else:
            # 画像全体に影や照明のムラがない場合 -> global binarize
            _, image_binary = cv2.threshold(
                image_gray,
                0,                  # 閾値（大津 -> 0）
                255,                # vmax
                cv2.THRESH_OTSU,  # 閾値処理の種類（大津の二値化 - 閾値の値を自動で決定）
            )
            return image_binary

    def dilate(self, image_binary_inv:NumpyImage, kernel, iterations:int=3) -> NumpyImage:
        """ 白色領域の膨張 """
        return cv2.dilate(image_binary_inv, kernel, iterations=iterations)

    def erode(self, image_binary_inv:NumpyImage, kernel, iterations:int=3) -> NumpyImage:
        """ 白色領域の収縮 """
        return cv2.erode(image_binary_inv, kernel, iterations=iterations)

    def opening(self, image_binary_inv:NumpyImage, kernel) -> NumpyImage:
        """ erode -> dilate
        黒色領域内の白点ノイズの除去に有効
        """
        return cv2.morphologyEx(image_binary_inv, cv2.MORPH_OPEN, kernel)

    def closing(self, image_binary_inv:NumpyImage, kernel) -> NumpyImage:
        """ dilate -> erode
        白色領域内の黒点ノイズの除去に有効
        """
        return cv2.morphologyEx(image_binary_inv, cv2.MORPH_CLOSE, kernel)

    def morphology_gradient(self, image_binary_inv:NumpyImage, kernel) -> NumpyImage:
        """ 物体の境界線を取得（膨張した画像と収縮した画像の差分をとる） """
        return cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)

    def hough_line(self,
        image_binary_inv:NumpyImage,
        threshold:int=120,       # 点の数に基づく線の判断基準。直線を動かしたときに直線上に存在する点の数の閾値。
        min_line_length:int=120, # 最短長に基づく線の判断基準。閾値以上の長さを持つ線を検出する。
        max_line_gap:int=7,      # 2点が同一線上にある場合に同一線と見なす間隔。指定値より小さいものを同一の線とみなす。
    ) -> np.ndarray:
        """ ハフ変換を用いた線の抽出 """
        return cv2.HoughLinesP(
            image_binary_inv,
            rho=1,
            theta=np.pi/360,
            threshold=threshold,
            minLineLength=min_line_length,
            maxLineGap=max_line_gap,
        )



if __name__ == "__main__":
    path = "imgs/081shitei_ika_ibaraki_r0501_0.jpg"
    proc = ImageProcessor()
    image_bgr = proc.load(path)
    image_gray = proc.bgr2gray(image_bgr)
    image_binary = proc.binarize(image_gray, in_shadow=False)
    print(type(image_binary))
