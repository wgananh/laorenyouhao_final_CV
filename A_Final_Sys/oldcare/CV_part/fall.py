# -*- coding: utf-8 -*-
'''
摔倒检测模型主程序

'''

# import the necessary packages
import imutils
from keras.preprocessing.image import img_to_array
from A_Final_Sys.modules.pose import Pose
from A_Final_Sys.modules.keypoints import extract_keypoints, group_keypoints
from A_Final_Sys.models.action_detect.detect import action_detect
from math import ceil, floor
from torch import from_numpy, jit
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


def check_fall_detection2(frame, output_fall_path, id_card_to_name, id_card_to_type,
                          fall_time_controller, faceutil, fall_net, fall_action_net,
                          python_path, roomID):

    is_fall = False

    stride = 8
    upsample_ratio = 4
    num_keypoints = Pose.num_kpts
    height_size = 256

    orig_img = frame.copy()

    heatmaps, pafs, scale, pad = infer_fast(fall_net, frame, height_size, stride, upsample_ratio)

    total_keypoints_num = 0
    all_keypoints_by_type = []
    for kpt_idx in range(num_keypoints):  # 19th for bg
        total_keypoints_num += extract_keypoints(heatmaps[:, :, kpt_idx], all_keypoints_by_type,
                                                 total_keypoints_num)

    pose_entries, all_keypoints = group_keypoints(all_keypoints_by_type, pafs, demo=True)
    for kpt_id in range(all_keypoints.shape[0]):
        all_keypoints[kpt_id, 0] = (all_keypoints[kpt_id, 0] * stride / upsample_ratio - pad[1]) / scale
        all_keypoints[kpt_id, 1] = (all_keypoints[kpt_id, 1] * stride / upsample_ratio - pad[0]) / scale
    current_poses = []
    for n in range(len(pose_entries)):
        if len(pose_entries[n]) == 0:
            continue
        pose_keypoints = np.ones((num_keypoints, 2), dtype=np.int32) * -1
        for kpt_id in range(num_keypoints):
            if pose_entries[n][kpt_id] != -1.0:  # keypoint was found
                pose_keypoints[kpt_id, 0] = int(all_keypoints[int(pose_entries[n][kpt_id]), 0])
                pose_keypoints[kpt_id, 1] = int(all_keypoints[int(pose_entries[n][kpt_id]), 1])
        pose = Pose(pose_keypoints, pose_entries[n][18])
        if len(pose.getKeyPoints()) >= 10:
            current_poses.append(pose)
        # current_poses.append(pose)

    for pose in current_poses:
        pose.img_pose = pose.draw(frame, show_draw=True)
        crown_proportion = pose.bbox[2] / pose.bbox[3]  # 宽高比
        pose = action_detect(fall_action_net, pose, crown_proportion)

        if pose.pose_action == 'fall':
            cv2.rectangle(frame, (pose.bbox[0], pose.bbox[1]),
                          (pose.bbox[0] + pose.bbox[2], pose.bbox[1] + pose.bbox[3]), (0, 0, 255), thickness=3)
            cv2.putText(frame, 'state: {}'.format(pose.pose_action), (pose.bbox[0], pose.bbox[1] - 16),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255))
            is_fall = True

        else:
            cv2.rectangle(frame, (pose.bbox[0], pose.bbox[1]),
                          (pose.bbox[0] + pose.bbox[2], pose.bbox[1] + pose.bbox[3]), (0, 255, 0))
            cv2.putText(frame, 'state: {}'.format(pose.pose_action), (pose.bbox[0], pose.bbox[1] - 16),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0))

    img = cv2.addWeighted(orig_img, 0.6, frame, 0.4, 0)

    if is_fall:
        if fall_time_controller.fall_timing == 0:  # just start timing
            fall_time_controller.set_fall_timing(1)
            fall_time_controller.start_fall_start_time()
        else:  # alredy started timing
            fall_end_time = time.time()
            difference = fall_end_time - fall_time_controller.fall_start_time

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

    return img


def infer_fast(net, img, net_input_height_size, stride, upsample_ratio,
               pad_value=(0, 0, 0), img_mean=(128, 128, 128), img_scale=1 / 256):
    height, width, _ = img.shape
    scale = net_input_height_size / height

    scaled_img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    scaled_img = normalize(scaled_img, img_mean, img_scale)
    min_dims = [net_input_height_size, max(scaled_img.shape[1], net_input_height_size)]
    padded_img, pad = pad_width(scaled_img, stride, pad_value, min_dims)

    tensor_img = from_numpy(padded_img).permute(2, 0, 1).unsqueeze(0).float()
    tensor_img = tensor_img.cuda()  # whg

    stages_output = net(tensor_img)

    # print(stages_output)

    stage2_heatmaps = stages_output[-2]
    heatmaps = np.transpose(stage2_heatmaps.squeeze().cpu().data.numpy(), (1, 2, 0))
    heatmaps = cv2.resize(heatmaps, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)

    stage2_pafs = stages_output[-1]
    pafs = np.transpose(stage2_pafs.squeeze().cpu().data.numpy(), (1, 2, 0))
    pafs = cv2.resize(pafs, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)

    return heatmaps, pafs, scale, pad


def normalize(img, img_mean, img_scale):
    img = np.array(img, dtype=np.float32)
    img = (img - img_mean) * img_scale
    return img


def pad_width(img, stride, pad_value, min_dims):
    h, w, _ = img.shape
    h = min(min_dims[0], h)
    min_dims[0] = ceil(min_dims[0] / float(stride)) * stride
    min_dims[1] = max(min_dims[1], w)
    min_dims[1] = ceil(min_dims[1] / float(stride)) * stride
    pad = []
    pad.append(int(floor((min_dims[0] - h) / 2.0)))
    pad.append(int(floor((min_dims[1] - w) / 2.0)))
    pad.append(int(min_dims[0] - h - pad[0]))
    pad.append(int(min_dims[1] - w - pad[1]))
    padded_img = cv2.copyMakeBorder(img, pad[0], pad[2], pad[1], pad[3],
                                    cv2.BORDER_CONSTANT, value=pad_value)
    return padded_img, pad
