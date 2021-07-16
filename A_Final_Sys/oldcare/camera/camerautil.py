# -*- coding: utf-8 -*-

import threading
import A_Final_Sys.oldcare.CV_part.emotion as Emoion
import A_Final_Sys.oldcare.CV_part.volunterActivity as Activity
import A_Final_Sys.oldcare.CV_part.fall as Fall
import A_Final_Sys.oldcare.CV_part.fence as Fence
import A_Final_Sys.oldcare.CV_part.collection as Collection
import cv2
import subprocess


class RecordingThread(threading.Thread):
    def __init__(self, name, camera, save_video_path):
        threading.Thread.__init__(self)
        self.name = name
        self.isRunning = True

        self.cap = camera
        fourcc = cv2.VideoWriter_fourcc(*'XVID')  # MJPG
        self.out = cv2.VideoWriter(save_video_path, fourcc, 20.0,
                                   (640, 480), True)

    def run(self):
        while self.isRunning:
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.flip(frame, 1)
                self.out.write(frame)

        self.out.release()

    def stop(self):
        self.isRunning = False

    def __del__(self):
        self.out.release()


class VideoCamera(object):
    def __init__(self, cap):
        # Open a camera
        self.cap = cap

        # Initialize video recording environment
        self.is_record = False
        self.out = None

        # Thread for recording
        self.recordingThread = None

    def __del__(self):
        self.cap.release()

    def get_frame(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            ret, jpeg = cv2.imencode('.jpg', frame)
            return jpeg.tobytes()

        else:
            return None

    def get_frame_stranger(self, output_stranger_path, output_smile_path,
                           id_card_to_name, id_card_to_type, facial_expression_id_to_name,
                           stranger_time_controller, face_time_controller, insert_controller,
                           faceutil, facial_expression_model, python_path, roomID):

        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            frame = Emoion.check_stranger_and_emotion(frame, output_stranger_path, output_smile_path,
                                                      id_card_to_name, id_card_to_type,
                                                      facial_expression_id_to_name,
                                                      stranger_time_controller,
                                                      face_time_controller,
                                                      insert_controller,
                                                      faceutil, facial_expression_model,
                                                      python_path, roomID)
            ret, jpeg = cv2.imencode('.jpg', frame)
            return jpeg.tobytes()

        else:
            return None

    def get_frame_activity(self, output_activity_path,
                           FACE_ACTUAL_WIDTH, ACTUAL_DISTANCE_LIMIT,
                           id_card_to_name, id_card_to_type,
                           faceutil, python_path, roomID,
                           activity_time_controller):

        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            frame = Activity.checkingvolunteeractivity(frame, output_activity_path,
                                                       FACE_ACTUAL_WIDTH, ACTUAL_DISTANCE_LIMIT,
                                                       id_card_to_name, id_card_to_type,
                                                       faceutil, python_path, roomID,
                                                       activity_time_controller)
            ret, jpeg = cv2.imencode('.jpg', frame)
            return jpeg.tobytes()
        else:
            return None

    def get_frame_collection(self, counter, output_activity_path,
                             FACE_ACTUAL_WIDTH, ACTUAL_DISTANCE_LIMIT,
                             id_card_to_name, id_card_to_type,
                             faceutil):

        ret, frame = self.cap.read()
        counter += 1

        if ret:
            if counter <= 10:  # 放弃前10帧
                frame = cv2.flip(frame, 1)
                frame = Activity.checkingvolunteeractivity(frame, output_activity_path,
                                                           FACE_ACTUAL_WIDTH, ACTUAL_DISTANCE_LIMIT,
                                                           id_card_to_name, id_card_to_type,
                                                           faceutil)
                ret, jpeg = cv2.imencode('.jpg', frame)
                return jpeg.tobytes()
            else:
                ret, jpeg = cv2.imencode('.jpg', frame)
                return jpeg.tobytes()
        else:
            return None

    def get_frame_fall(self, output_fall_path, id_card_to_name, id_card_to_type,
                       fall_time_controller, faceutil, fall_model,
                       python_path, roomID):

        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            frame = Fall.check_fall_detection(frame, output_fall_path, id_card_to_name, id_card_to_type,
                                              fall_time_controller, faceutil, fall_model,
                                              python_path, roomID)
            ret, jpeg = cv2.imencode('.jpg', frame)
            return jpeg.tobytes()
        else:
            return None

    def get_frame_fall2(self, output_fall_path, id_card_to_name, id_card_to_type,
                        fall_time_controller, faceutil, fall_net, fall_action_net,
                        python_path, roomID):

        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            frame = Fall.check_fall_detection2(frame, output_fall_path, id_card_to_name, id_card_to_type,
                                               fall_time_controller, faceutil, fall_net, fall_action_net,
                                               python_path, roomID)
            ret, jpeg = cv2.imencode('.jpg', frame)
            return jpeg.tobytes()
        else:
            return None

    def get_frame_fence(self, output_fence_path, CLASSES,
                        fence_tool, fence_time_controller,
                        fence_model, python_path, roomID):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            frame = Fence.check_fence(frame, output_fence_path, CLASSES,
                                      fence_tool, fence_time_controller,
                                      fence_model, python_path, roomID)
            ret, jpeg = cv2.imencode('.jpg', frame)
            return jpeg.tobytes()
        else:
            return None

    def start_record(self, save_video_path):
        self.is_record = True
        self.recordingThread = RecordingThread(
            "Video Recording Thread",
            self.cap, save_video_path)
        self.recordingThread.start()

    def stop_record(self):
        self.is_record = False

        if self.recordingThread != None:
            self.recordingThread.stop()
