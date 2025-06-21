#!/usr/bin/python
# coding=UTF-8
import re
import serial
import binascii
import crcmod.predefined


class CRCGenerator:

    def __init__(self):
        self.module = 'crc-8'

    def crc_8(self, hexData):
        crc8 = crcmod.predefined.Crc(self.module)
        hex_data = binascii.unhexlify(hexData)
        crc8.update(hex_data)
        result = hex(crc8.crcValue)
        result = re.sub("0x", '', result)
        if len(result) < 2:
            result = "0" + result
        return result


class MySerial:
    recv_data = ""

    def __init__(self, port, baudrate, timeout):
        self.port = serial.Serial(port, baudrate)
        if self.port.is_open:
            print("open :", self.port.portstr)
        else:
            print("打开端口失败")
        self.recv_msg = ""
        # 是否做校验
        self.crc_send = True
        self.crc_recv = True
        self.THREAD_CONTROL = True

    @staticmethod
    def hex_show(data):
        hex_data = ''
        h_len = len(data)
        for i in range(h_len):
            h_hex = '%02x' % data[i]
            hex_data += h_hex
        return hex_data

    @staticmethod
    def str2hex(string_data):
        hex_data = bytes.fromhex(string_data)
        return hex_data

    def receive_msg(self):
        while self.THREAD_CONTROL:
            size = self.port.in_waiting
            if size:
                self.recv_data = self.port.read_all()
                if self.recv_data != "":
                    data = self.hex_show(self.recv_data)
                    if self.crc_recv and data[-2:] == crc.crc_8(data[:-2]):
                        self.recv_msg = str(self.hex_show(self.recv_data))
                        # print(self.recv_msg)
            self.recv_data = ""

    def send_msg(self, data):
        send_data = self.str2hex(data + crc.crc_8(data))
        self.port.write(send_data)


# 实例化
crc = CRCGenerator()

