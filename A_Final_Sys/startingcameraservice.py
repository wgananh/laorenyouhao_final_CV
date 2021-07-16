# -*- coding: utf-8 -*-

"""
启动摄像头主程序

用法:
python startingcameraservice.py
python startingcameraservice.py --location room

"""

import argparse
import os
import time
import cv2
import tensorflow as tf
import keras
from flask import Flask, render_template, Response, request
from keras.models import load_model
from oldcare.facial import FaceUtil
from oldcare.camera import VideoCamera
from oldcare.utils import Time_Controller
from oldcare.utils import Fence_Tools
from oldcare.utils import InsertController
from oldcare.utils import getIP
from flask_cors import *
from oldcare.utils import fileassistant
import keras.backend.tensorflow_backend as K
from torch import from_numpy, jit

# from time import sleep
# import threadpool
# import threading
# from concurrent.futures import ThreadPoolExecutor

# 传入参数
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--location", required=False,
                default='room', help="")
args = vars(ap.parse_args())
location = args['location']

if location not in ['room', 'yard', 'corridor', 'desk']:
    raise ValueError('location must be one of room, yard, corridor or desk')

# API
app = Flask(__name__)
CORS(app, supports_credentials=True)

os.environ["PYTORCH_JIT"] = "0"

cap1 = cv2.VideoCapture(0)
cap2 = cv2.VideoCapture(1)
cap1.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
cap2.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))

video_camera1 = VideoCamera(cap1)
video_camera2 = VideoCamera(cap2)

global_frame = None
global_frame_activity = None
global_frame_stranger = None
global_frame_collection = None
global_frame_fall = None
global_frame_fence = None

# 设置GPU显存的动态加载
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
graph = tf.get_default_graph()  ##################!!!!!!!!###############
K.set_session(sess)

roomID = "房间202"

facial_recognition_model_path = 'models/face_recognition_hog.pickle'
facial_expression_model_path = 'models/miniGOOGLE_emotion_100_7.hdf5'
fall_model_path = 'models/miniVGG_fall_old.hdf5'
fall_net = jit.load('models/weights/openpose.jit')
fall_action_net = jit.load('models/action_detect/checkPoint/action.jit')

fall_net = fall_net.eval()
fall_net = fall_net.cuda()

prototxt_file_path = 'models/mobilenet_ssd/MobileNetSSD_deploy.prototxt'
model_file_path = 'models/mobilenet_ssd/MobileNetSSD_deploy.caffemodel'

people_info_path = 'info/people_info.csv'
facial_expression_info_path = 'info/facial_expression_info.csv'

output_stranger_path = 'supervision/strangers'
output_smile_path = 'supervision/smile'
output_activity_path = 'supervision/activity'  # 义工交互图片保存路径
output_fall_path = 'supervision/fall'  # 老人摔倒图片的保存路径
output_fence_path = 'supervision/fence'  # 禁止区域闯入人员照片

python_path = "D:/whg/anaconda3/envs/tf/python"

FACE_ACTUAL_WIDTH = 20  # 单位厘米   姑且认为所有人的脸都是相同大小
ACTUAL_DISTANCE_LIMIT = 100  # cm

# 得到 ID->姓名的map 、 ID->职位类型的map、
# 摄像头ID->摄像头名字的map、表情ID->表情名字的map
id_card_to_name, id_card_to_type = fileassistant.get_people_info(
    people_info_path)
facial_expression_id_to_name = fileassistant.get_facial_expression_info(
    facial_expression_info_path)

action_list = ['blink', 'open_mouth', 'smile', 'rise_head', 'bow_head',
               'look_left', 'look_right']
action_map = {'blink': '请眨眼', 'open_mouth': '请张嘴',
              'smile': '请笑一笑', 'rise_head': '请抬头',
              'bow_head': '请低头', 'look_left': '请看左边',
              'look_right': '请看右边'}
# 物体识别模型能识别的物体（21种）
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair",
           "cow", "diningtable", "dog", "horse", "motorbike",
           "person", "pottedplant", "sheep", "sofa", "train",
           "tvmonitor"]

# 初始化人脸识别模型
# keras.backend.clear_session()
faceutil = FaceUtil(facial_recognition_model_path)
facial_expression_model = load_model(facial_expression_model_path)
fall_model = load_model(fall_model_path)
fence_model = cv2.dnn.readNetFromCaffe(prototxt_file_path, model_file_path)


@app.route('/')
def index():
    return render_template('room_camera.html')


@app.route('/emotion')
def index2():
    return render_template('stranger_emotion.html')


@app.route('/activity')
def index3():
    return render_template('activity.html')


@app.route('/fall')
def index4():
    return render_template('fall.html')


@app.route('/fall2')
def index42():
    return render_template('fall2.html')


@app.route('/fence')
def index5():
    return render_template('fence.html')


@app.route('/record_status', methods=['POST'])
def record_status():
    global video_camera1

    status = request.form.get('status')
    save_video_path = request.form.get('save_video_path')

    if status == "true":
        video_camera1.start_record(save_video_path)
        return 'start record'
    else:
        video_camera1.stop_record()
        return 'stop record'


# 默认监控话画面
def video_stream():
    global video_camera1
    global video_camera2
    global global_frame

    while True:
        frame = video_camera1.get_frame()

        if frame is not None:
            global_frame = frame
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        else:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + global_frame + b'\r\n\r\n')


# 老人与义工交互检测
def video_stream_activity(activity_time_controller):
    global video_camera1
    global video_camera2
    global global_frame_activity
    global output_activity_path
    global FACE_ACTUAL_WIDTH
    global ACTUAL_DISTANCE_LIMIT
    global id_card_to_name
    global id_card_to_type
    global faceutil
    global python_path
    global roomID

    global graph
    with graph.as_default():
        while True:
            frame = video_camera2.get_frame_activity(output_activity_path,
                                                     FACE_ACTUAL_WIDTH, ACTUAL_DISTANCE_LIMIT,
                                                     id_card_to_name, id_card_to_type,
                                                     faceutil, python_path, roomID,
                                                     activity_time_controller)

            if frame is not None:
                global_frame_activity = frame
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
            else:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + global_frame_activity + b'\r\n\r\n')


# 陌生人检测及老人表情检测
def video_stream_stranger(stranger_time_controller, face_time_controller, insert_controller):
    global video_camera1
    global video_camera2
    global global_frame_stranger
    global output_stranger_path
    global output_smile_path
    global id_card_to_name
    global id_card_to_type
    global facial_expression_id_to_name
    global faceutil
    global facial_expression_model
    global python_path
    global roomID

    global graph
    with graph.as_default():
        while True:
            frame = video_camera2.get_frame_stranger(output_stranger_path, output_smile_path,
                                                     id_card_to_name, id_card_to_type, facial_expression_id_to_name,
                                                     stranger_time_controller, face_time_controller, insert_controller,
                                                     faceutil, facial_expression_model,
                                                     python_path, roomID)

            if frame is not None:
                global_frame_stranger = frame
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
            else:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + global_frame_stranger + b'\r\n\r\n')


# 摔倒检测
def video_stream_fall(fall_time_controller):
    global video_camera1
    global video_camera2
    global global_frame_fall
    global output_fall_path
    global id_card_to_name
    global id_card_to_type
    global faceutil
    global fall_model
    global python_path

    global graph
    with graph.as_default():
        while True:
            frame = video_camera2.get_frame_fall(output_fall_path, id_card_to_name, id_card_to_type,
                                                 fall_time_controller, faceutil, fall_model,
                                                 python_path, roomID)

            if frame is not None:
                global_frame_fall = frame
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
            else:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + global_frame_fall + b'\r\n\r\n')


def video_stream_fall2(fall_time_controller):
    global video_camera1
    global video_camera2
    global global_frame_fall
    global output_fall_path
    global id_card_to_name
    global id_card_to_type
    global faceutil
    global fall_net
    global fall_action_net
    global python_path
    global roomID

    global graph
    with graph.as_default():
        while True:
            frame = video_camera2.get_frame_fall2(output_fall_path, id_card_to_name, id_card_to_type,
                                                  fall_time_controller, faceutil, fall_net, fall_action_net,
                                                  python_path, roomID)

            if frame is not None:
                global_frame_fall = frame
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
            else:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + global_frame_fall + b'\r\n\r\n')


# 禁止区域检测
def video_stream_fence(fence_tool, fence_time_controller):
    global video_camera1
    global video_camera2
    global global_frame_fence
    global output_fence_path
    global fence_model
    global python_path
    global roomID

    global graph
    with graph.as_default():
        while True:
            frame = video_camera2.get_frame_fence(output_fence_path, CLASSES,
                                                  fence_tool, fence_time_controller,
                                                  fence_model, python_path, roomID)

            if frame is not None:
                global_frame_fence = frame
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
            else:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + global_frame_fence + b'\r\n\r\n')


@app.route('/video_viewer')
def video_viewer():
    return Response(video_stream(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/video_viewer_stranger')
def video_viewer_stranger():
    stranger_time_controller = Time_Controller()
    face_time_controller = Time_Controller()
    insert_controller = InsertController()
    return Response(video_stream_stranger(stranger_time_controller, face_time_controller, insert_controller),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/video_viewer_activity')
def video_viewer_activity():
    activity_time_controller = Time_Controller()
    return Response(video_stream_activity(activity_time_controller),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/video_viewer_fall')
def video_viewer_fall():
    fall_time_controller = Time_Controller()
    return Response(video_stream_fall(fall_time_controller),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/video_viewer_fall2')
def video_viewer_fall2():
    fall_time_controller = Time_Controller()
    return Response(video_stream_fall2(fall_time_controller),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/video_viewer_fence')
def video_viewer_fence():
    fence_tool = Fence_Tools()
    fence_time_controller = Time_Controller()
    return Response(video_stream_fence(fence_tool, fence_time_controller),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    # app.run(host=getIP.get_host_ip(), threaded=True, port=5001)
    app.run(host='0.0.0.0', threaded=True, port=5001)
