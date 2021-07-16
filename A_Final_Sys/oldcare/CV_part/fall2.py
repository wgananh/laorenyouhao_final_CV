# -*- coding: utf-8 -*-
'''
摔倒检测模型主程序

'''

# import the necessary packages
import imutils
from keras.preprocessing.image import img_to_array
import numpy as np
import cv2
import os
import time
import subprocess
import argparse


def check_fall_detection(frame, output_fall_path, id_card_to_name, id_card_to_type,
                         fall_time_controller, faceutil, fall_model,
                         python_path, roomID):
    TARGET_WIDTH = 64
    TARGET_HEIGHT = 64

    roi = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT))
    roi = roi.astype("float") / 255.0
    roi = img_to_array(roi)
    roi = np.expand_dims(roi, axis=0)

    # determine facial expression
    (fall, normal) = fall_model.predict(roi)[0]
    label = "Fall (%.2f)" % fall if fall > normal else "Normal (%.2f)" % normal

    # display the label and bounding box rectangle on the output frame
    cv2.putText(frame, label, (frame.shape[1] - 150, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

    if fall > normal:
        if fall_time_controller.fall_timing == 0:  # just start timing
            fall_time_controller.set_fall_timing(1)
            fall_time_controller.start_fall_start_time()
        else:  # alredy started timing
            fall_end_time = time.time()
            difference = fall_end_time - fall_time_controller.fall_start_time

            current_time = time.strftime('%Y-%m-%d %H:%M:%S',
                                         time.localtime(time.time()))

            if difference > fall_time_controller.fall_limit_time:
                # 有人摔倒
                fall_time_controller.set_fall_timing(0)
                fall_image_path = os.path.join(output_fall_path,
                                               'snapshot_%s.jpg' % (time.strftime('%Y%m%d_%H%M%S')))
                cv2.imwrite(fall_image_path, frame)  # snapshot

                # 传输抓拍的摔倒照片
                # insert into database
                command = '%s inserting.py --image_path %s --eventName %s --room %s --renyuanType ' \
                          '%s --renyuan %s' % (
                              python_path, fall_image_path, "2", roomID, "0", "null")
                p = subprocess.Popen(command, shell=True)

    return frame
