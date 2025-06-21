#!/usr/bin/python
# coding-UTF-8
import cv2
import threading
from img_rec.img_rec import ImageProcessing
from robotic_arm.my_serial import MySerial
# 控制机械臂位置
CONTROL_ROBOTIC_ARM_POSITION_DATA = "3001070155a11131"
# 控制机械臂抓取
CONTROL_ROBOTIC_ARM_GRAB_DATA = "3002070155a131"
# 单片机的功能选择
SINGLE_CHIP_FUNCTION_DATA = "3001070155a1ff"


# 创建串口对象
my_serial = MySerial('/dev/ttyS4', baudrate=115200, timeout=1)
# 创建图像识别对象
image_processing = ImageProcessing()
# 创建串口接收线程
t_serial = threading.Thread(target=my_serial.receive_msg)
t_serial.start()
vs = cv2.VideoCapture(0)
current_state = 0
next_state = 0
is_grab = True
source_lacation = 0
# my_serial.send_msg(SINGLE_CHIP_FUNCTION_DATA + "31")

while True:
    current_state = next_state
    if current_state == 0:
        my_serial.send_msg(CONTROL_ROBOTIC_ARM_POSITION_DATA)
        print("正在控制机械臂移动到仓库一。")
        next_state = 1
    elif current_state == 1:
        if my_serial.recv_msg[12:16] == "2131":
            print("机械臂已到达仓库一。")
            my_serial.recv_msg = ""
            next_state = 2
    elif current_state == 2:
        print("开始拍照。")
        for i in range(30):
            ret, frame = vs.read()
        cv2.imwrite("./pic.jpg", frame)
        print("拍摄完成，保存在pic.jpg，开始识别。")
        image_thresh, cargo_location = image_processing.image_position(frame)
        cargo_location_sort = image_processing.image_sort(cargo_location)
        rec_result = image_processing.image_recognize(cargo_location, cargo_location_sort, frame)
        print("识别完成，结果为：", rec_result)
        if is_grab and rec_result != {}:
            sort_result = sorted(rec_result.items(), key=lambda kv: (kv[1], kv[0]), reverse=False)
            print("排序完成，结果为：", sort_result)
            source_lacation = sort_result[0][0] + 1
            next_state = 3
        else:
            break
    elif current_state == 3:
        my_serial.send_msg(CONTROL_ROBOTIC_ARM_GRAB_DATA + "1{}21".format(source_lacation))
        next_state = 4
    elif current_state == 4:
        if my_serial.recv_msg[12:16] == "4131":
            print("机械臂抓取完毕")
            break
    else:
        break
my_serial.THREAD_CONTROL = False
