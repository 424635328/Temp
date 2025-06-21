#!/usr/bin/python
# coding=UTF-8
import cv2
import numpy as np
from .mnist_predict import img_input


class ImageProcessing:
    """
    图像处理的类
    """
    def __init__(self):
        """
        初始化
        """
        self.lower_yellow = np.array([26, 43, 46])
        self.upper_yellow = np.array([40, 255, 255])
        self.edge_img = np.ones((28, 28, 1), np.uint8) * 0
        self.edge_img[3:25, 6:23] = 255

    @staticmethod
    def histogram_equalization(image):
        """
        彩色图片的直方图均衡化,直方图均衡化是图像处理领域中利用图像直方图对对比度进行调整的方法。
        :param image: 传入的图片
        :return: 直方图均衡化后的图片
        """
        # YCbCr颜色空间, Y为颜色的亮度（luma）成分、而CB和CR则为蓝色和红色的浓度偏移量成分
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCR_CB)
        channels = cv2.split(ycrcb)
        cv2.equalizeHist(channels[0], channels[0])  # equalizeHist(in,out)
        cv2.merge(channels, ycrcb)
        he_image = cv2.cvtColor(ycrcb, cv2.COLOR_YCR_CB2BGR)
        return he_image

    def image_position(self, image):
        """
        图像定位：将黄色色块从整个图像中取出
        :param image: 传入的图像
        :return: 黄色色块的位置：[[y1, y2, x1, x2], ...]
        """
        # cv2.imshow('image_', image)
        # image = self.histogram_equalization(image)
        # cv2.imshow('image', image)
        image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)  # BGR转HSV
        # cv2.imshow('image_hsv', image_hsv)
        image_thresh = cv2.inRange(image_hsv, self.lower_yellow, self.upper_yellow)  # 区间内为白，其余为黑
        # cv2.imshow('image_thresh', image_thresh)
        image_thresh = cv2.medianBlur(image_thresh, 7)  # 中值滤波
        kenel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        open_image = cv2.morphologyEx(image_thresh, cv2.MORPH_OPEN, kenel, iterations=1)  # 开操作
        # cv2.imshow('open_image', open_image)
        binary, contours, hierarchy = cv2.findContours(open_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 查找轮廓
        cargo_location = []
        for i in range(len(contours)):
            cnt = contours[i]
            area = cv2.contourArea(cnt)  # 计算该轮廓的面积
            if area < 10000:  # 面积小的都筛选掉
                continue
            # print("area is: ", area)

            rect = cv2.minAreaRect(cnt)  # 找到最小的矩形

            box = cv2.boxPoints(rect)  # box是四个点的坐标
            box = np.int0(box)  # 其实就是numpy.int64,取整
            box = np.maximum(box, 0)  # 将小于0的调整为0
            ys = [box[0, 1], box[1, 1], box[2, 1], box[3, 1]]
            xs = [box[0, 0], box[1, 0], box[2, 0], box[3, 0]]
            ys_sorted_index = np.argsort(ys)
            xs_sorted_index = np.argsort(xs)
            x1 = box[xs_sorted_index[0], 0]
            x2 = box[xs_sorted_index[3], 0]
            y1 = box[ys_sorted_index[0], 1]
            y2 = box[ys_sorted_index[3], 1]
            location = [y1, y2, x1, x2]
            cargo_location.append(location)
        # print("cargo_location: ", cargo_location)
        return image_thresh, cargo_location

    @staticmethod
    def image_sort(cargo_location):
        """
        图像排序
        :param cargo_location:  数字图像位置
        :return:  识别结果
        """
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
        # print("cargo_location_sort: ", cargo_location_sort)
        return cargo_location_sort

    def edge_processing(self, img):
        """
        图像边框处理与缩放
        :param img: 预处理的图像
        :return: 处理后的图像
        """
        img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_CUBIC)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # ret, img_thresh = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY_INV)
        ret, img_thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        img_end = cv2.bitwise_and(img_thresh, self.edge_img, dst=None, mask=None)
        return img_end

    def image_recognize(self, cargo_location, cargo_location_sort, image_thresh):
        """
        图像识别
        :param cargo_location: 图像的位置信息
        :param cargo_location_sort:  排序后的图像位置信息
        :param image_thresh: 每一张处理后的图像，28*28
        :return:  识别结果，字典
        """
        location_result = {}
        for i in range(len(cargo_location)):
            if cargo_location_sort[i]:
                img = image_thresh[cargo_location_sort[i][0]: cargo_location_sort[i][1],
                      cargo_location_sort[i][2]: cargo_location_sort[i][3]]
                target_img = self.edge_processing(img)
                # cv2.imshow("target_img", target_img)
                # cv2.waitKey(0)
                location_result[i] = img_input(target_img)
        return location_result


# image_processing = ImageProcessing()
# # image = cv2.imread("./pic/sun.jpg")
# # image = cv2.imread("./pic/glare.jpg")
# image = cv2.imread("./pic.jpg")
# # image = cv2.imread("./pic/low_light.jpg")
# image_thresh, cargo_location = image_processing.image_position(image)
# cargo_location_sort = image_processing.image_sort(cargo_location)
# rec_result = image_processing.image_recognize(cargo_location, cargo_location_sort, image)
# print(rec_result)

