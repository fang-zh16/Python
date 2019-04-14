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

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from utils import label_map_util
from utils import visualization_utils as vis_util


class WebcamStreamingAnalyzer:
    def __init__(self):
        # 数据库资源
        self.database_connect = sqlite3.connect(
            "..\\..\\intelligent_monitoring_platform\\db.sqlite3")
        print("成功连接数据库！")
        self.cursor = self.database_connect.cursor()

        # # Model preparation
        # Any model exported using the `export_inference_graph.py` tool can be loaded here simply by changing `self.PATH_TO_CKPT` to point to a new .pb file.
        # By default we use an "SSD with Mobilenet" model here. See the [detection model zoo](https://github.com/tensorflow/models/blob/master/object_detection/g3doc/detection_model_zoo.md) for a list of other models that can be run out-of-the-box with varying speeds and accuracies.

        # What model to download.
        self.MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
        self.MODEL_FILE = self.MODEL_NAME + '.tar.gz'
        self.DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

        # Path to frozen detection graph. This is the actual model that is used for the object detection.
        self.PATH_TO_CKPT = self.MODEL_NAME + '/frozen_inference_graph.pb'

        # List of the strings that is used to add correct label for each box.
        self.PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

        self.NUM_CLASSES = 90

        # ## Download Model
        if not os.path.exists(self.MODEL_NAME + '/frozen_inference_graph.pb'):
            print('Downloading the model')
            opener = urllib.request.URLopener()
            opener.retrieve(self.DOWNLOAD_BASE +
                            self.MODEL_FILE, self.MODEL_FILE)
            tar_file = tarfile.open(self.MODEL_FILE)
            for file in tar_file.getmembers():
                file_name = os.path.basename(file.name)
                if 'frozen_inference_graph.pb' in file_name:
                    tar_file.extract(file, os.getcwd())
            print('Download complete')
        else:
            print('Model already exists')

        # ## Load a (frozen) Tensorflow model into memory.
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        # ## Loading label map
        # Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine
        self.label_map = label_map_util.load_labelmap(self.PATH_TO_LABELS)
        self.categories = label_map_util.convert_label_map_to_categories(
            self.label_map, max_num_classes=self.NUM_CLASSES, use_display_name=True)
        self.category_index = label_map_util.create_category_index(
            self.categories)

        self.cap = cv2.VideoCapture(0)
        fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        self.video_writer = cv2.VideoWriter(
            'output.avi', fourcc, 4, (640, 480))
        self.frame_count = 0

    def handle_frames(self):
        # Running the tensorflow session
        with self.detection_graph.as_default():
            with tf.Session(graph=self.detection_graph) as sess:
                ret = True
                while (ret):
                    ret, image_np = self.cap.read()
                    raw_frame = image_np
                    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                    image_np_expanded = np.expand_dims(image_np, axis=0)
                    image_tensor = self.detection_graph.get_tensor_by_name(
                        'image_tensor:0')
                    # Each box represents a part of the image where a particular object was detected.
                    boxes = self.detection_graph.get_tensor_by_name(
                        'detection_boxes:0')
                    # Each score represent how level of confidence for each of the objects.
                    # Score is shown on the result image, together with the class label.
                    scores = self.detection_graph.get_tensor_by_name(
                        'detection_scores:0')
                    classes = self.detection_graph.get_tensor_by_name(
                        'detection_classes:0')
                    num_detections = self.detection_graph.get_tensor_by_name(
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
                        self.category_index,
                        use_normalized_coordinates=True,
                        line_thickness=8)

                    # cv2.imshow('image', cv2.resize(image_np, (640, 480)))
                    self.video_writer.write(cv2.resize(image_np, (640, 480)))
                    if cv2.waitKey(25) & 0xFF == ord('q'):
                        cv2.destroyAllWindows()
                        self.cap.release()
                        self.video_writer.release()
                        self.database_connect.close()
                        break

                    print(self.frame_count)
                    self.frame_count = self.frame_count + 1
                    if True: # self.frame_count % 5 == 0:
                        current_time_float = time.time()
                        current_time_int = int(current_time_float)
                        current_time = str(current_time_float)
                        raw_image_filename = "main/static/raw/" + current_time
                        raw_image_format = "jpg"
                        cooked_image_filename = "main/static/cooked/" + current_time
                        cooked_image_format = "jpg"
                        # cv2.imwrite("..\\..\\intelligent_monitoring_platform\\" + raw_image_filename + ".jpg", raw_frame)
                        cv2.imwrite("..\\..\\intelligent_monitoring_platform\\" +
                                    cooked_image_filename + ".jpg", image_np)
                        max_id_in_database = 1
                        all_database_rows = self.cursor.execute(
                            "SELECT * FROM main_monitoringimage").fetchall()
                        if len(all_database_rows) != 0:
                            max_id_in_database = all_database_rows[len(
                                all_database_rows) - 1][0]

                        boxes_list = np.squeeze(boxes).tolist()
                        scores_list = np.squeeze(scores).tolist()
                        classes_list = np.squeeze(classes).tolist()
                        results_length = len(scores_list)
                        for i in range(results_length):
                            if scores_list[i] > 0.5:
                                # print(scores_list[i], self.category_index[classes_list[i]]["name"])
                                # 置信度, 种类
                                self.cursor.execute("""
                                    INSERT INTO main_record (id, class_result, image_id, time, confidence)
                                    VALUES (NULL, '""" + self.category_index[classes_list[i]]["name"] + "'," + str(max_id_in_database + 1) + "," + current_time + "," + str(scores_list[i]) + """)
                                """)

                        self.cursor.execute("""
                            INSERT INTO main_monitoringimage (id, raw_image_filename, raw_image_format, cooked_image_filename, cooked_image_format)
                            VALUES (NULL, '""" + raw_image_filename + "','" + raw_image_format + "','" + cooked_image_filename + "','" + cooked_image_format + """')
                        """)
                        self.database_connect.commit()


if __name__ == "__main__":
    analyzer = WebcamStreamingAnalyzer()
    analyzer.handle_frames()
