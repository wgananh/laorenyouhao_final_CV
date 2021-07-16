# -*- coding: utf-8 -*-
'''
图像采集程序-人脸检测
由于外部程序需要调用它，所以不能使用相对路径


'''
import argparse

from imutils import paths
from oldcare.facial import FaceUtil
from oldcare.audio import audioplayer
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
import os
import shutil
import time
import A_Final_Sys.oldcare.utils.getfromURL as URLTool

# 全局参数
# audio_dir = '/home/whg/code/laorenyouhao/face_collection/audios'
audio_dir = 'audios'

# 控制参数
error = 0
start_time = None
limit_time = 2  # 2 秒

image_base_path = "images"
people_info_path = "info/people_info.csv"
output_encoding_file_path = 'models/face_recognition_hog.pickle'

input_type = "laoren"
user_name = "杨斌"
user_id = time.strftime("%Y%m%d%H%M%S")
ope_name = "save"


api_name = ""  # yigong, yuangong, laoren
control_name = ""  # yigonginfo, yuangonginfo, laoreninfo
user_type = ""  # volunteer, old_people, employee
if input_type == "laoren":
    api_name = "laoren"
    control_name = "laoreninfo"
    user_type = "old_people"
if input_type == "yigong":
    api_name = "yigong"
    control_name = "yigonginfo"
    user_type = "volunteer"
if input_type == "yuangong":
    api_name = "yuangong"
    control_name = "yuangonginfo"
    user_type = "employee"

action_list = ['blink', 'open_mouth', 'smile', 'rise_head', 'bow_head',
               'look_left', 'look_right']
action_map = {'blink': '请眨眼', 'open_mouth': '请张嘴',
              'smile': '请笑一笑', 'rise_head': '请抬头',
              'bow_head': '请低头', 'look_left': '请看左边',
              'look_right': '请看右边'}
# 设置摄像头
cam = cv2.VideoCapture(1)
cam.set(3, 640)  # set video widht
cam.set(4, 480)  # set video height

faceutil = FaceUtil()

counter = 0
while True:
    counter += 1
    _, image = cam.read()
    if counter <= 10:  # 放弃前10帧
        continue
    image = cv2.flip(image, 1)

    if error == 1:
        end_time = time.time()
        difference = end_time - start_time
        print(difference)
        if difference >= limit_time:
            error = 0

    face_location_list = faceutil.get_face_location(image)
    for (left, top, right, bottom) in face_location_list:
        cv2.rectangle(image, (left, top), (right, bottom),
                      (0, 0, 255), 2)

    cv2.imshow('Collecting Faces', image)  # show the image
    # Press 'ESC' for exiting video
    k = cv2.waitKey(100) & 0xff
    if k == 27:
        break

    face_count = len(face_location_list)
    if error == 0 and face_count == 0:  # 没有检测到人脸
        print('[WARNING] 没有检测到人脸')
        audioplayer.play_audio(os.path.join(audio_dir,
                                            'no_face_detected.mp3'))
        error = 1
        start_time = time.time()
    elif error == 0 and face_count == 1:  # 可以开始采集图像了
        print('[INFO] 可以开始采集图像了')
        audioplayer.play_audio(os.path.join(audio_dir,
                                            'start_image_capturing.mp3'))
        break
    elif error == 0 and face_count > 1:  # 检测到多张人脸
        print('[WARNING] 检测到多张人脸')
        audioplayer.play_audio(os.path.join(audio_dir,
                                            'multi_faces_detected.mp3'))
        error = 1
        start_time = time.time()
    else:
        pass

# 新建目录

if os.path.exists(os.path.join(image_base_path, user_type, user_id)):
    shutil.rmtree(os.path.join(image_base_path, user_type, user_id), True)
os.mkdir(os.path.join(image_base_path, user_type, user_id))

# 开始采集人脸
for action in action_list:
    audioplayer.play_audio(os.path.join(audio_dir, action + '.mp3'))
    action_name = action_map[action]

    counter = 1
    for i in range(15):
        print('%s-%d' % (action_name, i))
        _, img_OpenCV = cam.read()
        img_OpenCV = cv2.flip(img_OpenCV, 1)
        origin_img = img_OpenCV.copy()  # 保存时使用

        face_location_list = faceutil.get_face_location(img_OpenCV)
        for (left, top, right, bottom) in face_location_list:
            cv2.rectangle(img_OpenCV, (left, top),
                          (right, bottom), (0, 0, 255), 2)

        img_PIL = Image.fromarray(cv2.cvtColor(img_OpenCV,
                                               cv2.COLOR_BGR2RGB))

        draw = ImageDraw.Draw(img_PIL)
        draw.text((int(image.shape[1] / 2), 30), action_name,
                  font=ImageFont.truetype("C:\\Windows\\Fonts\\SIMLI.TTF", 40),
                  fill=(255, 0, 0))  # linux

        # 转换回OpenCV格式
        img_OpenCV = cv2.cvtColor(np.asarray(img_PIL),
                                  cv2.COLOR_RGB2BGR)

        cv2.imshow('Collecting Faces', img_OpenCV)  # show the image

        image_name = os.path.join(image_base_path, user_type, user_id,
                                  action + '_' + str(counter) + '.jpg')
        cv2.imwrite(image_name, origin_img)
        # Press 'ESC' for exiting video
        k = cv2.waitKey(100) & 0xff
        if k == 27:
            break
        counter += 1

# 结束
print('[INFO] 采集完毕')
audioplayer.play_audio(os.path.join(audio_dir, 'end_capturing.mp3'))

# 释放全部资源
cam.release()
cv2.destroyAllWindows()

dataset_path = os.path.join(image_base_path, user_type, user_id)
image_paths = list(paths.list_images(dataset_path))
image_path = image_paths[0]

key = URLTool.upload_img(image_path)

inputdata = {}
inputdata["name"] = user_name
inputdata["profilePhoto"] = key
inputdata["imgsetDir"] = dataset_path
result = URLTool.HttpRequest(api_name, control_name, ope_name, "post", inputdata)

print(inputdata)
print(result["msg"])

if result["msg"] == "success":

    # 将采集的人脸构建模型
    # grab the paths to the input images in our dataset
    print("[INFO] quantifying faces...")
    image_paths = list(paths.list_images(dataset_path))

    if len(image_paths) == 0:
        print('[ERROR] no images to train.')
    else:
        faceutil = FaceUtil()
        print("[INFO] training face embeddings...")
        faceutil.save_embeddings(image_paths, output_encoding_file_path)

    # 将信息录入people_info.csv
    f = open(people_info_path, "ab")
    info = "\n" + user_id + "," + user_name + "," + user_type
    f.write(info.encode())
    f.close()
