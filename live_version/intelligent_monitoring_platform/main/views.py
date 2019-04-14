import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import sqlite3
import time
import cv2
import base64
import json
import queue
import threading

from django.contrib import auth
from django.shortcuts import render, redirect
from django.contrib.auth.models import User
from django.contrib.auth.decorators import login_required
from django.conf import settings
from .models import MonitoringImage, Alarm, Record
from django.http import HttpResponse, HttpResponseServerError, StreamingHttpResponse
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from main.utils import label_map_util
from main.utils import visualization_utils as vis_util

# # Model preparation
# Any model exported using the `export_inference_graph.py` tool can be loaded here simply by changing `PATH_TO_CKPT` to point to a new .pb file.
# By default we use an "SSD with Mobilenet" model here. See the [detection model zoo](https://github.com/tensorflow/models/blob/master/object_detection/g3doc/detection_model_zoo.md) for a list of other models that can be run out-of-the-box with varying speeds and accuracies.

# What model to download.
MODEL_NAME = 'intelligent_monitoring_platform/main/ssd_mobilenet_v1_coco_11_06_2017'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join(
    'intelligent_monitoring_platform', 'main', 'data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90

# ## Download Model
if not os.path.exists(MODEL_NAME + '/frozen_inference_graph.pb'):
    print('Downloading the model')
    opener = urllib.request.URLopener()
    opener.retrieve(DOWNLOAD_BASE +
                    MODEL_FILE, MODEL_FILE)
    tar_file = tarfile.open(MODEL_FILE)
    for file in tar_file.getmembers():
        file_name = os.path.basename(file.name)
        if 'frozen_inference_graph.pb' in file_name:
            tar_file.extract(file, os.getcwd())
    print('Download complete')
else:
    print('Model already exists')

# ## Load a (frozen) Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

# ## Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(
    label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(
    categories)


class Static:
    thread_queue = queue.Queue()


class FramesHandler(threading.Thread):
    def __init__(self, queue_with_frames, quit_signal, address=0):
        threading.Thread.__init__(self)
        self.queue_with_frames = queue_with_frames
        self.address = address
        self.quit_signal = quit_signal

    def __del__(self):
        pass

    def handle_frames(self):
        # 数据库资源
        database_connect = sqlite3.connect(
            "intelligent_monitoring_platform\\db.sqlite3")
        print("Connected to database successfully...")
        cursor = database_connect.cursor()

        cap = cv2.VideoCapture(self.address)
        frame_count = 0

        # Running the tensorflow session
        with detection_graph.as_default():
            with tf.Session(graph=detection_graph) as sess:
                while True:
                    try:
                        if self.quit_signal.qsize() != 0:
                            print(
                                "----------------------------------------------------------------------------------")
                            print("Camera at %s was released!  " %
                                  (self.address))
                            print(
                                "----------------------------------------------------------------------------------")
                            cap.release()
                            database_connect.close()
                            return

                        if self.address == 0:
                            ret, image_np = cap.read()
                        else:
                            for i in range(10):
                                ret, image_np = cap.read()
                        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                        image_np_expanded = np.expand_dims(image_np, axis=0)
                        image_tensor = detection_graph.get_tensor_by_name(
                            'image_tensor:0')
                        # Each box represents a part of the image where a particular object was detected.
                        boxes = detection_graph.get_tensor_by_name(
                            'detection_boxes:0')
                        # Each score represent how level of confidence for each of the objects.
                        # Score is shown on the result image, together with the class label.
                        scores = detection_graph.get_tensor_by_name(
                            'detection_scores:0')
                        classes = detection_graph.get_tensor_by_name(
                            'detection_classes:0')
                        num_detections = detection_graph.get_tensor_by_name(
                            'num_detections:0')
                        # Actual detection.
                        (boxes, scores, classes, num_detections) = sess.run(
                            [boxes, scores, classes, num_detections],
                            feed_dict={image_tensor: image_np_expanded})

                        # Visualization of the results of a detection.
                        vis_util.visualize_boxes_and_labels_on_image_array(
                            image_np,
                            np.squeeze(boxes),
                            np.squeeze(classes).astype(np.int32),
                            np.squeeze(scores),
                            category_index,
                            use_normalized_coordinates=True,
                            line_thickness=8)

                        print(frame_count)
                        frame_count = frame_count + 1
                        current_time_float = time.time()
                        current_time_int = int(current_time_float)
                        current_time = str(current_time_float)
                        raw_image_filename = "main/static/raw/" + current_time
                        raw_image_format = "jpg"
                        cooked_image_filename = "main/static/cooked/" + current_time
                        cooked_image_format = "jpg"
                        cv2.imwrite("intelligent_monitoring_platform\\" +
                                    cooked_image_filename + ".jpg", image_np)
                        max_id_in_database = 1
                        all_database_rows = cursor.execute(
                            "SELECT * FROM main_monitoringimage").fetchall()
                        if len(all_database_rows) != 0:
                            max_id_in_database = all_database_rows[len(
                                all_database_rows) - 1][0]

                        boxes_list = np.squeeze(boxes).tolist()
                        scores_list = np.squeeze(scores).tolist()
                        classes_list = np.squeeze(classes).tolist()
                        results_length = len(scores_list)
                        high_score_objects_detected = 0
                        for i in range(results_length):
                            if scores_list[i] > 0.5:
                                # print(scores_list[i], category_index[classes_list[i]]["name"])
                                # 置信度, 种类
                                high_score_objects_detected = high_score_objects_detected + 1
                                cursor.execute("""
                                    INSERT INTO main_record (id, class_result, image_id, time, confidence)
                                    VALUES (NULL, '""" + category_index[classes_list[i]]["name"] + "'," + str(max_id_in_database + 1) + "," + current_time + "," + str(scores_list[i]) + """)
                                """)
                        if high_score_objects_detected == 0:
                            cursor.execute("""
                                INSERT INTO main_record (id, class_result, image_id, time)
                                VALUES (NULL, 'none', """ + str(max_id_in_database + 1) + "," + current_time + """)
                            """)

                        cursor.execute("""
                            INSERT INTO main_monitoringimage (id, raw_image_filename, raw_image_format, cooked_image_filename, cooked_image_format)
                            VALUES (NULL, '""" + raw_image_filename + "','" + raw_image_format + "','" + cooked_image_filename + "','" + cooked_image_format + """')
                        """)
                        database_connect.commit()
                        ret, jpeg = cv2.imencode('.jpg', image_np)
                        self.queue_with_frames.put(jpeg.tobytes())
                    except:
                        print("error..")
                        pass

    def run(self):
        self.handle_frames()


def frame_generator(address):
    quit_signal = queue.Queue()
    queue_with_frames = queue.Queue()
    frames_handler = FramesHandler(queue_with_frames, quit_signal, address)
    while Static.thread_queue.qsize() != 0:
        Static.thread_queue.get()[1].put("quit signal")
    Static.thread_queue.put((frames_handler, quit_signal))
    frames_handler.start()
    while True:
        frame = queue_with_frames.get()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@login_required(login_url='login')
def camera_video(request):
    return StreamingHttpResponse(frame_generator(0), content_type="multipart/x-mixed-replace; boundary=frame")


@login_required(login_url='login')
def live_video(request):
    return StreamingHttpResponse(frame_generator("rtsp://admin:admin@59.66.68.38:554/cam/realmonitor?channel=1&subtype=0"), content_type="multipart/x-mixed-replace; boundary=frame")


@login_required(login_url='login')
def localcam(request):
    user = User.objects.filter(username=request.user)
    if time.time() - 8 * 3600 - time.mktime(time.strptime(str(user[0].last_login)[0:19], "%Y-%m-%d %H:%M:%S")) > 1800:
        auth.logout(request)
        return redirect("login")

    alarms = Alarm.objects.order_by("-id").all()
    for alarm in alarms:
        alarm.alarm_class = "检测到人"
        alarm.time_begin = time.strftime(
            "%Y-%m-%d %H:%M:%S", time.localtime(alarm.time_begin))
        alarm.time_end = time.strftime(
            "%Y-%m-%d %H:%M:%S", time.localtime(alarm.time_end))
    return render(request, "index.html", {
        "alarms": alarms,
        "camera": "local",
    })


@login_required(login_url='login')
def webcam(request):
    user = User.objects.filter(username=request.user)
    if time.time() - 8 * 3600 - time.mktime(time.strptime(str(user[0].last_login)[0:19], "%Y-%m-%d %H:%M:%S")) > 1800:
        auth.logout(request)
        return redirect("login")

    alarms = Alarm.objects.order_by("-id").all()
    for alarm in alarms:
        alarm.alarm_class = "检测到人"
        alarm.time_begin = time.strftime(
            "%Y-%m-%d %H:%M:%S", time.localtime(alarm.time_begin))
        alarm.time_end = time.strftime(
            "%Y-%m-%d %H:%M:%S", time.localtime(alarm.time_end))
    return render(request, "index.html", {
        "alarms": alarms,
        "camera": "web",
    })


@login_required(login_url='login')
def new_alarms(request):
    alarms = Alarm.objects.order_by("-id").all()
    result = []
    for alarm in alarms:
        if alarm.visible == 1:
            result.append({
                "id": alarm.id,
                "alarm_class": "检测到人",
                "time_begin": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(alarm.time_begin)),
                "time_end": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(alarm.time_end)),
            })
    return HttpResponse(json.dumps(result))


@login_required(login_url='login')
def delete_alarm(request, alarm_id):
    alarm_id = request.POST.get('alarm_id')
    Alarm.objects.filter(id=alarm_id).update(visible=0)
    return HttpResponse(json.dumps({
        "status": 200
    }))


@login_required(login_url='login')
def new_frame(request):
    try:
        all_objects = MonitoringImage.objects.order_by("id")
        image_info = all_objects[len(all_objects) - 1]
        image_path = os.path.join(settings.BASE_DIR, "%s.%s" % (
            image_info.cooked_image_filename, image_info.cooked_image_format))
        with open(image_path, 'rb') as f:
            image_data = base64.b64encode(f.read())
        return HttpResponse(image_data, content_type="image/jpeg")
    except Exception as exception:
        print(exception)
        return HttpResponseServerError()


@login_required(login_url='login')
def current_status(request):
    try:
        all_objects = Record.objects.order_by("id")
        info = all_objects[len(all_objects) - 1].class_result
        return HttpResponse(info)
    except Exception as exception:
        print(exception)
        return HttpResponseServerError()
