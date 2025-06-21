# -*- coding:utf-8 -*-
import os
import cv2
import tensorflow as tf
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 模型恢复
sess = tf.Session()
saver = tf.train.import_meta_graph('./img_rec/model_data/model.meta')
saver.restore(sess, './img_rec/model_data/model')
# saver = tf.train.import_meta_graph('./model_data/model.meta')
# saver.restore(sess, './model_data/model')
graph = tf.get_default_graph()


# 获取识别结果
def predict(img_thre):
    # 获取输入tensor,获取输出tensor
    input_x = sess.graph.get_tensor_by_name("Mul:0")
    y_conv2 = sess.graph.get_tensor_by_name("final_result:0")

    x_img = np.reshape(img_thre, [-1, 784])
    output = sess.run(y_conv2, feed_dict={input_x: x_img})
    return np.argmax(output)


# 图像
def img_input(img):
    recognize_result = predict(img)
    return recognize_result
