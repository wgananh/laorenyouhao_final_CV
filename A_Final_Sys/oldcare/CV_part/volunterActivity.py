# -*- coding: utf-8 -*-
"""
义工是否和老人有互动主程序
"""

from scipy.spatial import distance as dist
from PIL import Image, ImageDraw, ImageFont
import cv2
import time
import os
import imutils
import numpy as np
import subprocess


def checkingvolunteeractivity(frame, output_activity_path,
                              FACE_ACTUAL_WIDTH, ACTUAL_DISTANCE_LIMIT,
                              id_card_to_name, id_card_to_type,
                              faceutil, python_path, roomID,
                              activity_time_controller):
    camera_turned = 0
    frame = imutils.resize(frame,
                           width=640,
                           height=480)  # 压缩，为了加快识别速度

    face_location_list, names = faceutil.get_face_location_and_name(frame)

    # 得到画面的四分之一位置和四分之三位置，并垂直划线
    one_sixth_image_center = (int(640 / 6), int(480 / 6))
    five_sixth_image_center = (int(640 / 6 * 5),
                               int(480 / 6 * 5))

    cv2.line(frame, (one_sixth_image_center[0], 0),
             (one_sixth_image_center[0], 480),
             (0, 255, 255), 1)
    cv2.line(frame, (five_sixth_image_center[0], 0),
             (five_sixth_image_center[0], 480),
             (0, 255, 255), 1)

    people_type_list = list(set([id_card_to_type[i] for i in names]))

    volunteer_name_direction_dict = {}
    volunteer_centroids = []
    volunteer_name = ""
    old_people_centroids = []
    old_people_name = []

    # loop over the face bounding boxes
    for ((left, top, right, bottom), name) in zip(face_location_list, names):  # 处理单个人

        person_type = id_card_to_type[name]
        # 将人脸框出来
        rectangle_color = (0, 0, 255)
        if person_type == 'old_people':
            rectangle_color = (0, 0, 128)
        elif person_type == 'employee':
            rectangle_color = (255, 0, 0)
        elif person_type == 'volunteer':
            rectangle_color = (0, 255, 0)
        else:
            pass
        cv2.rectangle(frame, (left, top), (right, bottom),
                      rectangle_color, 2)

        if 'volunteer' not in people_type_list:  # 如果没有义工，直接跳出本次循环
            continue

        if person_type == 'volunteer':  # 如果检测到有义工存在
            # 获得义工位置
            volunteer_face_center = (int((right + left) / 2),
                                     int((top + bottom) / 2))
            volunteer_centroids.append(volunteer_face_center)
            volunteer_name = name

            cv2.circle(frame,
                       (volunteer_face_center[0], volunteer_face_center[1]),
                       8, (255, 0, 0), -1)

            adjust_direction = ''
            # face locates too left, servo need to turn right,
            # so that face turn right as well
            if volunteer_face_center[0] < one_sixth_image_center[0]:
                adjust_direction = 'right'
            elif volunteer_face_center[0] > five_sixth_image_center[0]:
                adjust_direction = 'left'

            volunteer_name_direction_dict[name] = adjust_direction

        elif person_type == 'old_people':  # 如果没有发现义工
            old_people_face_center = (int((right + left) / 2),
                                      int((top + bottom) / 2))
            old_people_centroids.append(old_people_face_center)
            old_people_name.append(name)

            cv2.circle(frame,
                       (old_people_face_center[0], old_people_face_center[1]),
                       4, (0, 255, 0), -1)
        else:
            pass

        # 人脸识别和表情识别都结束后，把表情和人名写上 (同时处理中文显示问题)
        img_PIL = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_PIL)
        final_label = id_card_to_name[name]
        draw.text((left, top - 30), final_label,
                  font=ImageFont.truetype('C:\\Windows\\Fonts\\SIMLI.TTF', 40),
                  fill=(255, 0, 0))  # linux
        # 转换回OpenCV格式
        frame = cv2.cvtColor(np.asarray(img_PIL), cv2.COLOR_RGB2BGR)

    # 义工追踪逻辑
    if 'volunteer' in people_type_list:
        volunteer_adjust_direction_list = list(volunteer_name_direction_dict.values())
        if '' in volunteer_adjust_direction_list:  # 有的义工恰好在范围内，所以不需要调整舵机
            x=1
            # print('有义工恰好在可见范围内，摄像头不需要转动')
        else:
            adjust_direction = volunteer_adjust_direction_list[0]
            camera_turned = 1
            # print('摄像头需要 turn %s %d 度' % (adjust_direction, 20))

    # 在义工和老人之间划线
    is_activity = False

    if camera_turned == 0:
        for i in volunteer_centroids:
            for j_index, j in enumerate(old_people_centroids):
                pixel_distance = dist.euclidean(i, j)
                face_pixel_width = sum([i[2] - i[0] for i in face_location_list]) / len(face_location_list)
                pixel_per_metric = face_pixel_width / FACE_ACTUAL_WIDTH
                actual_distance = pixel_distance / pixel_per_metric

                if actual_distance < ACTUAL_DISTANCE_LIMIT:
                    cv2.line(frame, (int(i[0]), int(i[1])),
                             (int(j[0]), int(j[1])), (255, 0, 255), 2)

                    label = 'distance: %dcm' % actual_distance
                    cv2.putText(frame, label, (frame.shape[1] - 150, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (0, 0, 255), 2)
                    is_activity = True

    if is_activity:
        if activity_time_controller.activity_timing == 0:
            activity_time_controller.set_activity_timing(1)
            activity_time_controller.start_activity_start_time()

        else:
            activity_end_time = time.time()
            difference = activity_end_time - activity_time_controller.activity_start_time

            if difference > activity_time_controller.activity_limit_time:
                activity_image_path = os.path.join(output_activity_path,
                                                   'snapshot_%s.jpg' % (time.strftime('%Y%m%d_%H%M%S')))
                cv2.imwrite(activity_image_path, frame)  # snapshot
                activity_time_controller.set_activity_timing(0)

                # 传输抓拍的义工交互的照片
                # insert into database
                old_volunteer_types = ""
                old_volunteer_names = ""
                for i in range(0, len(old_people_name)):
                    old_volunteer_names += id_card_to_name[old_people_name[i]]
                    old_volunteer_types += "0"
                    old_volunteer_names += "_"
                    old_volunteer_types += "_"

                old_volunteer_types += "2"
                old_volunteer_names += id_card_to_name[volunteer_name]

                print(old_volunteer_types)
                print(old_volunteer_names)

                command = '%s inserting.py --image_path %s --eventName %s --room %s --renyuanType ' \
                          '%s --renyuan %s' % (
                              python_path, activity_image_path, "3", roomID, old_volunteer_types, old_volunteer_names)
                p = subprocess.Popen(command, shell=True)
    return frame
