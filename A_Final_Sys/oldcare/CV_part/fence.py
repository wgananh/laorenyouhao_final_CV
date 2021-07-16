# -*- coding: utf-8 -*-

'''
禁止区域检测主程序
摄像头对准围墙那一侧
'''

# import the necessary packages
from ..track import TrackableObject
import numpy as np
import imutils
import time
import dlib
import cv2
import os
import subprocess


def check_fence(frame, output_fence_path, CLASSES,
                fence_tool, fence_time_controller,
                fence_model, python_path, roomID):
    minimum_confidence = 0.80
    W = None
    H = None
    # resize the frame to have a maximum width of 500 pixels (the
    # less data we have, the faster we can process it), then convert
    # the frame from BGR to RGB for dlib
    frame = imutils.resize(frame, width=500)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # if the frame dimensions are empty, set them
    if W is None or H is None:
        (H, W) = frame.shape[:2]

    # initialize the current status along with our list of bounding
    # box rectangles returned by either (1) our object detector or
    # (2) the correlation trackers
    status = "Waiting"
    rects = []

    # check to see if we should run a more computationally expensive
    # object detection method to aid our tracker
    if fence_tool.totalFrames % fence_tool.skip_frames == 0:
        # set the status and initialize our new set of object trackers
        status = "Detecting"
        fence_tool.trackers = []

        # convert the frame to a blob and pass the blob through the
        # network and obtain the detections
        blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)
        fence_model.setInput(blob)
        detections = fence_model.forward()

        # loop over the detections
        for i in np.arange(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated
            # with the prediction
            confidence = detections[0, 0, i, 2]

            # filter out weak detections by requiring a minimum
            # confidence
            if confidence > minimum_confidence:
                # extract the index of the class label from the
                # detections list
                idx = int(detections[0, 0, i, 1])

                # if the class label is not a person, ignore it
                if CLASSES[idx] != "person":
                    continue

                # compute the (x, y)-coordinates of the bounding box
                # for the object
                box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                (startX, startY, endX, endY) = box.astype("int")

                # construct a dlib rectangle object from the bounding
                # box coordinates and then start the dlib correlation
                # tracker
                tracker = dlib.correlation_tracker()
                rect = dlib.rectangle(int(startX), int(startY), int(endX), int(endY))
                # rect = dlib.rectangle(startX, startY, endX, endY)
                tracker.start_track(rgb, rect)

                # add the tracker to our list of trackers so we can
                # utilize it during skip frames
                fence_tool.trackers.append(tracker)

    # otherwise, we should utilize our object *trackers* rather than
    # object *detectors* to obtain a higher frame processing throughput
    else:
        # loop over the trackers
        for tracker in fence_tool.trackers:
            # set the status of our system to be 'tracking' rather
            # than 'waiting' or 'detecting'
            status = "Tracking"

            # update the tracker and grab the updated position
            tracker.update(rgb)
            pos = tracker.get_position()

            # unpack the position object
            startX = int(pos.left())
            startY = int(pos.top())
            endX = int(pos.right())
            endY = int(pos.bottom())

            # draw a rectangle around the people
            cv2.rectangle(frame, (startX, startY), (endX, endY),
                          (0, 255, 0), 2)

            # add the bounding box coordinates to the rectangles list
            rects.append((startX, startY, endX, endY))

    # draw a horizontal line in the center of the frame -- once an
    # object crosses this line we will determine whether they were
    # moving 'up' or 'down'
    cv2.line(frame, (0, H // 2), (W, H // 2), (0, 255, 255), 2)
    # cv2.line(frame, (0, 375 // 2), (500, 375 // 2), (0, 255, 255), 2)

    # use the centroid tracker to associate the (1) old object
    # centroids with (2) the newly computed object centroids
    objects = fence_tool.ct.update(rects)

    # loop over the tracked objects
    for (objectID, centroid) in objects.items():
        # check to see if a trackable object exists for the current
        # object ID
        to = fence_tool.trackableObjects.get(objectID, None)

        # if there is no existing trackable object, create one
        if to is None:
            to = TrackableObject(objectID, centroid)

        # otherwise, there is a trackable object so we can utilize it
        # to determine direction
        else:
            # the difference between the y-coordinate of the *current*
            # centroid and the mean of *previous* centroids will tell
            # us in which direction the object is moving (negative for
            # 'up' and positive for 'down')
            y = [c[1] for c in to.centroids]
            direction = centroid[1] - np.mean(y)
            to.centroids.append(centroid)

            # check to see if the object has been counted or not
            if not to.counted:
                # if the direction is negative (indicating the object
                # is moving up) AND the centroid is above the center
                # line, count the object
                if direction < 0 and centroid[1] < H // 2:
                    fence_tool.totalUp += 1
                    to.counted = True

                # if the direction is positive (indicating the object
                # is moving down) AND the centroid is below the
                # center line, count the object
                elif direction > 0 and centroid[1] > H // 2:
                    fence_tool.totalDown += 1
                    to.counted = True

                    current_time = time.strftime('%Y-%m-%d %H:%M:%S',
                                                 time.localtime(time.time()))
                    print('[EVENT] %s, 院子, 有人闯入禁止区域!!!' % current_time)

                    if fence_time_controller.fence_timing == 0:
                        fence_time_controller.set_fence_timing(1)
                        fence_time_controller.start_fence_timing()

                    else:
                        fence_end_time = time.time()
                        difference = fence_end_time - fence_time_controller.fence_start_time

                        if difference > fence_time_controller.fence_limit_time:
                            fence_image_path = os.path.join(output_fence_path,
                                                            'snapshot_%s.jpg' % (time.strftime('%Y%m%d_%H%M%S')))
                            cv2.imwrite(fence_image_path, frame)  # snapshot

                            fence_time_controller.set_fence_timing(0)
                            # 传输抓拍的高兴照片
                            # insert into database
                            command = '%s inserting.py --image_path %s --eventName %s --room %s --renyuanType ' \
                                      '%s --renyuan %s' % (
                                          python_path, fence_image_path, "4", roomID, "0", "null")
                            p = subprocess.Popen(command, shell=True)

                # store the trackable object in our dictionary
        fence_tool.trackableObjects[objectID] = to

        # draw both the ID of the object and the centroid of the
        # object on the output frame
        text = "ID {}".format(objectID)
        cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.circle(frame, (centroid[0], centroid[1]), 4,
                   (0, 255, 0), -1)

    # construct a tuple of information we will be displaying on the
    # frame
    info = [
        # ("Up", totalUp),
        ("Down", fence_tool.totalDown),
        ("Status", status),
    ]

    # loop over the info tuples and draw them on our frame
    for (i, (k, v)) in enumerate(info):
        text = "{}: {}".format(k, v)
        cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # increment the total number of frames processed thus far and
    # then update the FPS counter
    fence_tool.totalFrames += 1

    return frame
