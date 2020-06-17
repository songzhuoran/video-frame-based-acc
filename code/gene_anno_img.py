import csv
import os,sys
import re
from PIL import Image, ImageDraw, ImageColor
import cv2
import numpy as np

classes_map = ['__background__',  # always index 0
            'n02691156', 'n02419796', 'n02131653', 'n02834778',
            'n01503061', 'n02924116', 'n02958343', 'n02402425',
            'n02084071', 'n02121808', 'n02503517', 'n02118333',
            'n02510455', 'n02342885', 'n02374451', 'n02129165',
            'n01674464', 'n02484322', 'n03790512', 'n02324045',
            'n02509815', 'n02411705', 'n01726692', 'n02355227',
            'n02129604', 'n04468005', 'n01662784', 'n04530566',
            'n02062744', 'n02391049']

Map_DIR="/home/songzhuoran/video/video-frame-based-acc/data/mapping_result_mthread/"
ANN_DIR="/home/songzhuoran/benchmark/imagenet-detection/Annotations/VID/val/"
Ben_DIR="/home/songzhuoran/video/video-frame-based-acc/data/benchmark_result/"

videofiles= os.listdir(ANN_DIR)
for videofile in videofiles:
    imagefiles = os.listdir(ANN_DIR+videofile)
    for imagefile in imagefiles:
        framenum = re.sub('[.xml]', '', imagefile)
        framenum1 = "%06d" % int(framenum)
        framenum2 = "%08d" % int(framenum)
        with open(ANN_DIR+videofile+"/"+framenum1+".xml","r") as gtfile: # ground truth file
            gtline = gtfile.readline()
            while gtline:
                for i in range(len(classes_map)):
                    if i !=0:
                        if gtline.find(classes_map[i]) != -1:
                            # print(classes_map[int(content[1])])
                            gtline = gtfile.readline()
                            gtline = gtfile.readline()
                            gtmaxx = int(gtline[(gtline.find(">")+1):gtline.rfind("<")])
                            gtline = gtfile.readline()
                            gtminx = int(gtline[(gtline.find(">")+1):gtline.rfind("<")])  
                            gtline = gtfile.readline()
                            gtmaxy = int(gtline[(gtline.find(">")+1):gtline.rfind("<")])
                            gtline = gtfile.readline()
                            gtminy = int(gtline[(gtline.find(">")+1):gtline.rfind("<")])
                            cur_img = np.zeros((800,1500))
                            len_x = int(gtmaxx - gtminx)
                            len_y = int(gtmaxy - gtminy)
                            for x in range(len_x):
                                for y in range(len_y):
                                    cur_img[int(gtminy+y)][int(gtminx+x)] = 255
                            imgstr = Ben_DIR + videofile+"/"+framenum2+"/"+str(classes_map[i])+".png"
                            cv2.imwrite(imgstr,cur_img)

                        gtline = gtfile.readline()


# videofiles= os.listdir(Map_DIR)
# for videofile in videofiles:
#     os.mkdir(Ben_DIR+videofile)
#     imagefiles = os.listdir(Map_DIR+videofile)
#     for imagefile in imagefiles:
#         os.mkdir(Ben_DIR+videofile+"/"+imagefile)

