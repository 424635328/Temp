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
cargo_list = []  # 存储所有待抓取物品
grabbed_count = 0  # 已抓取物品计数器

try:
    while True:
        current_state = next_state
        
        if current_state == 0:
            # 移动到仓库
            my_serial.send_msg(CONTROL_ROBOTIC_ARM_POSITION_DATA)
            print("正在控制机械臂移动到仓库。")
            next_state = 1
            
        elif current_state == 1:
            # 等待到达仓库
            if my_serial.recv_msg[12:16] == "2131":
                print("机械臂已到达仓库。")
                my_serial.recv_msg = ""
                next_state = 2
                
        elif current_state == 2:
            # 拍照并识别物品
            print("开始拍照。")
            for i in range(30):  # 预热摄像头
                ret, frame = vs.read()
            cv2.imwrite("./pic.jpg", frame)
            print("拍摄完成，保存在pic.jpg，开始识别。")
            
            # 图像处理和识别
            image_thresh, cargo_location = image_processing.image_position(frame)
            cargo_location_sort = image_processing.image_sort(cargo_location)
            rec_result = image_processing.image_recognize(cargo_location, cargo_location_sort, frame)
            
            print("识别完成，结果为：", rec_result)
            
            if is_grab and rec_result:
                # 排序物品，准备抓取
                cargo_list = sorted(rec_result.items(), key=lambda kv: (kv[1], kv[0]))
                if cargo_list:
                    next_state = 3
                else:
                    print("没有可抓取的物品")
                    break
            else:
                print("未识别到物品或抓取已取消")
                break
                
        elif current_state == 3:
            # 检查是否还有物品需要抓取
            if cargo_list:
                current_cargo = cargo_list.pop(0)  # 获取下一个要抓取的物品
                source_location = current_cargo[0] + 1  # 物品位置
                my_serial.send_msg(CONTROL_ROBOTIC_ARM_GRAB_DATA + "1{}21".format(source_location))
                print(f"正在抓取位置 {source_location} 的物品")
                next_state = 4
            else:
                print("所有物品抓取完成")
                next_state = 5  # 退出流程
                
        elif current_state == 4:
            # 等待抓取完成
            if my_serial.recv_msg[12:16] == "4131":
                grabbed_count += 1
                print(f"机械臂抓取完毕，已抓取 {grabbed_count} 个物品")
                my_serial.recv_msg = ""
                next_state = 0  # 回到初始状态，准备抓取下一个物品
                
        elif current_state == 5:
            # 所有物品抓取完成，退出循环
            break
            
        else:
            print("未知状态，退出程序")
            break
            
except KeyboardInterrupt:
    print("程序被用户中断")
    
finally:
    # 清理资源
    my_serial.THREAD_CONTROL = False
    vs.release()
    print(f"共抓取 {grabbed_count} 个物品")