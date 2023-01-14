# -*- coding: utf-8 -*-

import cv2
import gradio as gr
import numpy as np

from modules import ImageProcessor


NumpyImage = np.ndarray



class MyGradio:
    def __init__(self):
        self.proc = ImageProcessor()

    def predict_fn(self, image_bgr:NumpyImage) -> NumpyImage:
        width, height, channel = image_bgr.shape
        image_gray = self.proc.bgr2gray(image_bgr)
        image_binary = self.proc.binarize(image_gray)
        image_binary_inv = self.proc.inverse(image_binary)
        """ dilate (vertical) """
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, height//200))
        image_binary_inv = self.proc.dilate(image_binary_inv, vertical_kernel, iterations=1)
        image_binary_inv = self.proc.erode(image_binary_inv, vertical_kernel, iterations=1)
        # image_binary_inv = self.proc.closing(image_binary_inv, vertical_kernel)
        """ dilate (horizontal) """
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (width//100, 1))
        image_binary_inv = self.proc.dilate(image_binary_inv, horizontal_kernel, iterations=1)
        image_binary_inv = self.proc.erode(image_binary_inv, horizontal_kernel, iterations=1)
        # image_binary_inv = self.proc.closing(image_binary_inv, horizontal_kernel)
        """ hough transform """
        lines = self.proc.hough_line(
            image_binary_inv,
            threshold=int(width/1.5),
            min_line_length=int(width/1.5),
            max_line_gap=7,
        )
        """ drawing """
        red = (0, 0, 255)
        line_px = 3
        for line in lines:
            left, top, right, bottom = line[0]
            cv2.line(image_bgr, (left, top), (right, bottom), color=red, thickness=line_px)
        return image_bgr

    def launch(self):
        with gr.Blocks() as demo:
            # layout
            with gr.Column():
                with gr.Row():
                    image_input = gr.Image(type="numpy")
                    image_output = gr.Image(type="numpy")
                buttom = gr.Button(value="submit")
            # submit
            buttom.click(
                self.predict_fn,
                inputs=[image_input],
                outputs=[image_output],
            )
            # examples
            gr.Examples(
                examples=[
                    ["imgs/081shitei_ika_ibaraki_r0501_0.jpg"],
                ],
                inputs=[image_input],
                outputs=[image_output],
                fn=self.predict_fn,
                cache_examples=False
            )
        # submit
        demo.launch(share=True)



if __name__ == "__main__":
    web = MyGradio()
    web.launch()
