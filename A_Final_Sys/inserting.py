# -*- coding: utf-8 -*-
'''
将事件插入数据库主程序

用法：

'''

import datetime
import argparse
import oldcare.utils.getfromURL as URLTool

# 传入参数
ap = argparse.ArgumentParser()
ap.add_argument("-img", "--image_path", required=False,
                default='', help="")
ap.add_argument("-en", "--eventName", required=False,
                default='', help="")
ap.add_argument("-r", "--room", required=False,
                default='', help="")
ap.add_argument("-type", "--renyuanType", required=False,
                default='', help="")
ap.add_argument("-ren", "--renyuan", required=False,
                default='', help="")

args = vars(ap.parse_args())

smile_image_path = args['image_path']
eventName = args['eventName']
room = args['room']
renyuanType = args['renyuanType']
renyuan = args['renyuan']

# event_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
key = URLTool.upload_img(smile_image_path)

inputdata = {}
inputdata["eventName"] = eventName
inputdata["room"] = room
inputdata["path"] = key
inputdata["renyuanType"] = renyuanType
inputdata["renyuan"] = renyuan

result = URLTool.HttpRequest("event", "eventinfo", "event", "post", inputdata)
print(result)
if result["msg"] == "success":
    print('插入成功')
