import glob
import os
import time

if __name__ == "__main__":
    while True:
        print("正在执行删除操作！")

        file_names = glob.iglob(
            r"../../intelligent_monitoring_platform/main/static/raw/*.jpg")
        name = []
        pre = ""

        for py in file_names:
            pyshort = os.path.basename(py)
            pynumber = pyshort[:-4]
            name.append(float(pynumber))
            loc = py.find(pyshort)
            pre = py[0:loc]

        name.sort()
        lengthsize = len(name)

        for i in range(lengthsize - 1):
            nameshort = str(name[i])
            pos = pre + nameshort + ".jpg"
            os.remove(pos)

        file_names = glob.iglob(
            r"../../intelligent_monitoring_platform/main/static/cooked/*.jpg")
        name = []
        pre = ""

        for py in file_names:
            pyshort = os.path.basename(py)
            pynumber = pyshort[:-4]
            name.append(float(pynumber))
            loc = py.find(pyshort)
            pre = py[0:loc]

        name.sort()
        lengthsize = len(name)

        for i in range(lengthsize - 1):
            nameshort = str(name[i])
            pos = pre + nameshort + ".jpg"
            os.remove(pos)

        print("删除结束！开始休眠...")
        time.sleep(60)
