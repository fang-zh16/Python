import sqlite3
import time
import os

class RecordAnalyzer:
    def __init__(self):
        self.database_connect = sqlite3.connect("..\\..\\intelligent_monitoring_platform\\db.sqlite3")
        print("成功连接数据库！")
        self.cursor = self.database_connect.cursor()
        
    def __del__(self):
        self.database_connect.commit()
        self.database_connect.close()

    def handle_record(self):
        main_record_rows = self.cursor.execute(
            "SELECT * FROM main_record").fetchall()
        main_record_len = len(main_record_rows)
        main_alarm_rows = self.cursor.execute(
            "SELECT * FROM main_alarm").fetchall()
        main_alarm_len = len(main_alarm_rows)
        if main_record_len == 0:
            return
        begin_id = 0
        if main_alarm_len == 0:
            begin_id = main_record_rows[0][0]
        else:
            begin_id = main_alarm_rows[main_alarm_len - 1][3] + 1

        self.detect_person(begin_id)

    def detect_person(self, begin_id):
        rows = self.cursor.execute("SELECT * FROM main_record WHERE id >= " + str(begin_id))
        alarms = []
        for row in rows:
            if row[1] == "person":
                alarms.append(row)
        alarms_len = len(alarms)
        if alarms_len == 0:
            return
        
        alarm_range = []
        begin = 0
        end = 0
        for i in range(alarms_len - 1):
            if alarms[i + 1][8] - alarms[i][8] > 3: # 间隔 3 秒
                end = i
                alarm_range.append((begin, end))
                begin = i + 1

            # 对最后一个的检测
            if i + 1 == alarms_len - 1 and int(time.time()) - alarms[i + 1][8] > 3:
                end = i + 1
                alarm_range.append((begin, end))

        for alarm in alarm_range:
            image_id_begin = alarms[alarm[0]][0]
            image_id_end = alarms[alarm[1]][0]
            time_begin = alarms[alarm[0]][8]
            time_end = alarms[alarm[1]][8]
            self.cursor.execute("""
                INSERT INTO main_alarm (id, alarm_class, image_id_begin, image_id_end, time_begin, time_end, visible)
                VALUES (NULL, 'person', """ + str(image_id_begin) + "," + str(image_id_end) + "," + str(time_begin) + "," + str(time_end) + """, 1)
            """)
        
        self.database_connect.commit()


if __name__ == "__main__":
    record_analyzer = RecordAnalyzer()
    while True:
        print("唤醒开始处理...")
        record_analyzer.handle_record()
        print("处理完成...")
        time.sleep(3)
        