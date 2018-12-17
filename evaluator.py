import time
import logging.config
import math
import cv2
import tensorflow as tf
import datetime
from models import yolo
from log_config import LOGGING
from utils.general import format_predictions, find_class_by_name, is_url

import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
import base64
import os

import onesignal as onesignal_sdk
cred = credentials.Certificate('/path/to/your/key.json')
default_app = firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://your-base.firebaseio.com'
})

pusher_image = db.reference('/images')

onesignal_client = onesignal_sdk.Client(user_auth_key="key", app={"app_auth_key": "key", "app_id": "id"})


logging.config.dictConfig(LOGGING)

logger = logging.getLogger('detector')

class ImageCreator:
    def __init__(self, vid):
        self.win_name = 'Detector'
        self.cam = cv2.VideoCapture(vid)

        self.source_h = self.cam.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.source_w = self.cam.get(cv2.CAP_PROP_FRAME_WIDTH)

        self.model_cls = find_class_by_name("Yolo2Model", [yolo])
        self.model = self.model_cls(input_shape=(self.source_h, self.source_w, 3))
        self.model.init()

        self.start_time = time.time()
        self.fps = 0
        self.last_time = time.time()

    def predict(self):
        ret, frame = self.cam.read()
        predictions = self.model.evaluate(frame)
        num_persons = 0
        for o in predictions:
            x1 = o['box']['left']
            x2 = o['box']['right']

            y1 = o['box']['top']
            y2 = o['box']['bottom']

            color = o['color']
            class_name = o['class_name']
            if class_name == 'person':
                if (abs(x1-x2) * abs(y1-y2)) > 1000:
                    num_persons = num_persons+1
                    # Draw box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                    # Draw label
                    (test_width, text_height), baseline = cv2.getTextSize(
                        class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 1)
                    cv2.rectangle(frame, (x1, y1),
                                    (x1+test_width, y1-text_height-baseline),
                                    color, thickness=cv2.FILLED)
                    cv2.putText(frame, class_name, (x1, y1-baseline),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 1)

        end_time = time.time()
        self.fps = self.fps * 0.9 + 1/(end_time - self.start_time) * 0.1
        self.start_time = end_time
        # Draw additional info
        frame_info = '{0}, FPS: {1:.2f}'.format(datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y"), self.fps)
        cv2.putText(frame, frame_info, (10, frame.shape[0]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        logger.info(frame_info)

        if predictions:
            logger.info('Predictions: {}'.format(
                format_predictions(predictions)))
        if num_persons > 0 and time.time() - self.last_time > 20:
            name = datetime.datetime.now().strftime("%I:%M%p-%B-%d-%Y")
            cv2.imwrite("/home/nvidia/footage/"+name+".png", frame)
            self.last_time = time.time()
            cv2.imwrite('yo.png', frame)
            img_f = open('yo.png', 'r')
            out = base64.b64encode(img_f.read())
            pusher_image.push(str(name)+" "+str(out))
            os.remove('yo.png')
            new_notification = onesignal_sdk.Notification(contents={"en": "A person may be at your house at "+str(name), "tr": "Person!"})
            new_notification.set_included_segments(["All"])
            onesignal_response = onesignal_client.send_notification(new_notification)

        return frame
