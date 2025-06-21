#!/usr/bin/python
# coding=UTF-8
import cv2
import numpy as np
from .mnist_predict import img_input


class ImageProcessing:
    """
    图像处理的类，增强了强光环境下的识别鲁棒性
    """
    def __init__(self):
        """
        初始化
        """
        # 基础黄色HSV阈值
        self.base_lower_yellow = np.array([26, 43, 46])
        self.base_upper_yellow = np.array([40, 255, 255])
        # 强光环境下的阈值补偿系数
        self.brightness_factor = 1.0
        self.edge_img = np.ones((28, 28, 1), np.uint8) * 0
        self.edge_img[3:25, 6:23] = 255
        # 初始化CLAHE处理器
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def adjust_threshold_by_brightness(self, image):
        """
        根据图像亮度动态调整HSV阈值
        :param image: 输入图像
        :return: 调整后的HSV阈值
        """
        # 计算图像亮度均值
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        
        # 根据亮度调整阈值补偿系数
        if brightness > 200:  # 强光环境
            self.brightness_factor = 0.8 + (255 - brightness) / 100.0
            if self.brightness_factor < 0.5:
                self.brightness_factor = 0.5
        elif brightness < 100:  # 弱光环境
            self.brightness_factor = 1.0 + (100 - brightness) / 200.0
            if self.brightness_factor > 1.5:
                self.brightness_factor = 1.5
        else:  # 正常光照
            self.brightness_factor = 1.0
        
        # 动态调整HSV阈值
        lower_yellow = self.base_lower_yellow.copy()
        upper_yellow = self.base_upper_yellow.copy()
        
        # 对V通道(亮度)进行自适应调整
        lower_yellow[2] = int(lower_yellow[2] * self.brightness_factor)
        upper_yellow[2] = int(upper_yellow[2] * min(self.brightness_factor, 1.2))
        
        # 确保阈值在有效范围内
        lower_yellow = np.clip(lower_yellow, [0, 43, 46], [180, 255, 255])
        upper_yellow = np.clip(upper_yellow, [0, 43, 46], [180, 255, 255])
        
        return lower_yellow, upper_yellow

    def handle_overexposure(self, image):
        """
        过曝区域检测与处理
        :param image: 输入图像
        :return: 过曝处理后的图像
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # 检测过曝区域(V通道>230)
        overexposed = cv2.inRange(hsv, np.array([0, 0, 230]), np.array([180, 30, 255]))
        # 对过曝区域进行高斯模糊处理
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        # 替换过曝区域像素
        result = image.copy()
        result[overexposed > 0] = blurred[overexposed > 0]
        return result

    def enhanced_histogram_equalization(self, image):
        """
        增强型直方图均衡化，结合CLAHE处理强光场景
        :param image: 输入图像
        :return: 处理后的图像
        """
        # 先进行过曝处理
        image = self.handle_overexposure(image)
        # 转换到YCrCb颜色空间
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        channels = cv2.split(ycrcb)
        # 对亮度通道应用CLAHE
        channels[0] = self.clahe.apply(channels[0])
        # 合并通道
        cv2.merge(channels, ycrcb)
        he_image = cv2.cvtColor(ycrcb, cv2.COLOR_YCR_CB2BGR)
        return he_image

    def image_position(self, image):
        """
        图像定位：增强强光环境下的黄色色块提取
        :param image: 传入的图像
        :return: 黄色色块的位置：[[y1, y2, x1, x2], ...]
        """
        # 增强型直方图均衡化
        image = self.enhanced_histogram_equalization(image)
        # 获取动态调整后的HSV阈值
        lower_yellow, upper_yellow = self.adjust_threshold_by_brightness(image)
        # BGR转HSV
        image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # 颜色阈值分割
        image_thresh = cv2.inRange(image_hsv, lower_yellow, upper_yellow)
        # 多级噪声过滤
        image_thresh = cv2.medianBlur(image_thresh, 5)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        # 先进行闭操作连接断裂区域，再进行开操作去除噪声
        image_thresh = cv2.morphologyEx(image_thresh, cv2.MORPH_CLOSE, kernel)
        image_thresh = cv2.morphologyEx(image_thresh, cv2.MORPH_OPEN, kernel)
        
        # 查找轮廓
        binary, contours, hierarchy = cv2.findContours(
            image_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        cargo_location = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            # 动态调整面积过滤阈值，适应强光下可能的轮廓变化
            min_area = max(5000, int(10000 * self.brightness_factor))
            if area < min_area:
                continue
                
            # 计算最小外接矩形
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            box = np.maximum(box, 0)
            
            # 计算边界框坐标
            ys = [box[0, 1], box[1, 1], box[2, 1], box[3, 1]]
            xs = [box[0, 0], box[1, 0], box[2, 0], box[3, 0]]
            y1, y2 = min(ys), max(ys)
            x1, x2 = min(xs), max(xs)
            cargo_location.append([y1, y2, x1, x2])
        
        return image_thresh, cargo_location

    def edge_processing(self, img):
        """
        增强型边缘处理，优化强光下的二值化效果
        :param img: 预处理的图像
        :return: 处理后的图像
        """
        # 调整图像尺寸
        img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_CUBIC)
        # 转换为灰度图
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 自适应阈值处理，替代固定阈值
        img_thresh = cv2.adaptiveThreshold(
            img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        # 应用边缘掩码
        img_end = cv2.bitwise_and(img_thresh, self.edge_img)
        return img_end

    # 其他方法保持不变
    @staticmethod
    def image_sort(cargo_location):
        image_num = len(cargo_location)
        cargo_location_sort = [[], [], [], []]
        for i in range(image_num):
            if ((cargo_location[i][0] + cargo_location[i][1]) / 2 < 240) \
                    and ((cargo_location[i][2] + cargo_location[i][3]) / 2 < 320):
                cargo_location_sort[0] = cargo_location[i]
            elif ((cargo_location[i][0] + cargo_location[i][1]) / 2 < 240) \
                    and ((cargo_location[i][2] + cargo_location[i][3]) / 2 > 320):
                cargo_location_sort[1] = cargo_location[i]
            elif ((cargo_location[i][0] + cargo_location[i][1]) / 2 > 240) \
                    and ((cargo_location[i][2] + cargo_location[i][3]) / 2 < 320):
                cargo_location_sort[2] = cargo_location[i]
            elif ((cargo_location[i][0] + cargo_location[i][1]) / 2 > 240) \
                    and ((cargo_location[i][2] + cargo_location[i][3]) / 2 > 320):
                cargo_location_sort[3] = cargo_location[i]
        return cargo_location_sort

    def image_recognize(self, cargo_location, cargo_location_sort, image_thresh):
        location_result = {}
        for i in range(len(cargo_location)):
            if cargo_location_sort[i]:
                img = image_thresh[cargo_location_sort[i][0]: cargo_location_sort[i][1],
                      cargo_location_sort[i][2]: cargo_location_sort[i][3]]
                target_img = self.edge_processing(img)
                location_result[i] = img_input(target_img)
        return location_result