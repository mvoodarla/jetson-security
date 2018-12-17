import cv2
from base_camera import BaseCamera
from evaluator import ImageCreator
import sys

class Camera(BaseCamera):

    @staticmethod
    def set_video_source(source):
        Camera.video_source = source

    @staticmethod
    def frames():
        creator = ImageCreator(1)
        while True:
            yield cv2.imencode('.jpg', creator.predict())[1].tobytes()
