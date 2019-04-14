import cv2
import time
import sqlite3
import os
from yolo import YOLO_TF


class ImageAnalyzer():
    def __init__(self):
        # 摄像头资源
        self.capture = cv2.VideoCapture(0)

        # 数据库资源
        self.database_connect = sqlite3.connect(
            "..\\..\\intelligent_monitoring_platform\\db.sqlite3")
        print("成功连接数据库！")
        self.cursor = self.database_connect.cursor()

        self.yolo = YOLO_TF(self.cursor)

    def __del__(self):
        self.capture.release()
        self.database_connect.commit()
        self.database_connect.close()

    def handle_frame(self):
        _, frame = self.capture.read()
        current_time_float = time.time()
        current_time_int = int(current_time_float)
        current_time = str(current_time_float)
        raw_image_filename = "main/static/raw/" + current_time
        raw_image_format = "jpg"
        cooked_image_filename = "main/static/cooked/" + current_time
        cooked_image_format = "jpg"
        cv2.imwrite("..\\..\\intelligent_monitoring_platform\\" +
                    raw_image_filename + ".jpg", frame)
        max_id_in_database = 1
        all_database_rows = self.cursor.execute(
            "SELECT * FROM main_monitoringimage").fetchall()
        if len(all_database_rows) != 0:
            max_id_in_database = all_database_rows[len(
                all_database_rows) - 1][0]
        self.yolo.analyze(max_id_in_database + 1, "..\\..\\intelligent_monitoring_platform\\" + raw_image_filename +
                          ".jpg", "..\\..\\intelligent_monitoring_platform\\" + cooked_image_filename + ".jpg", current_time_int)
        self.cursor.execute("""
            INSERT INTO main_monitoringimage (id, raw_image_filename, raw_image_format, cooked_image_filename, cooked_image_format)
            VALUES (NULL, '""" + raw_image_filename + "','" + raw_image_format + "','" + cooked_image_filename + "','" + cooked_image_format + """')
        """)
        self.database_connect.commit()


if __name__ == "__main__":
    count = 0
    analyzer = ImageAnalyzer()
    while True:
        count += 1
        print(count)
        analyzer.handle_frame()
