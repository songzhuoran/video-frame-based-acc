import csv
import os,sys
import re
from PIL import Image, ImageDraw, ImageColor
import cv2
import numpy as np

f = open("/home/songzhuoran/video/video-frame-based-acc/VID_val_videos.txt","r")
video_names = f.readlines()
f.close()

# # make video directory
# for i in video_names:
#     video_list = i.split(' ') # i.e., ['ILSVRC2015_val_00000000', '1', '0', '464\n']
#     os.mkdir('/home/songzhuoran/video/video-frame-based-acc/data/baseline_result/' + video_list[0])

f = open("/home/songzhuoran/video/video-frame-based-acc/tmp_nms_result.txt","r") 
rectangles = f.readlines()
f.close()

for r in rectangles:
    rectangle_list = r.split(' ') # i.e., [214,27,0.9800,621.31,163.67,831.14,360.44\n]
    rectangle_list[6]=rectangle_list[6].replace('\n', '').replace('\r', '') # i.e., 360.44
    for i in video_names:
        video_list = i.split(' ') # i.e., ['ILSVRC2015_val_00000000', '1', '0', '464\n']
        video_list[3]=video_list[3].replace('\n', '').replace('\r', '') # i.e., 464
        start_file_num = int(video_list[1])-1 # i.e., 0
        final_file_num = start_file_num + int(video_list[3]) - 1 # i.e., 463
        if int(rectangle_list[0])>=start_file_num and int(rectangle_list[0])<=final_file_num :
            image_name = int(rectangle_list[0]) - start_file_num # start from 0???? need to justify, i.e., 213
            curstr = '%08d' % image_name # i.e., 00213
            imgfold = '/home/songzhuoran/video/video-frame-based-acc/data/baseline_result/' + video_list[0] + '/' + curstr
            imgstr = imgfold + '/' + rectangle_list[1] + '.png'
            cur_img = cv2.imread(imgstr,0) #open frame
            if rectangle_list[1] not in os.listdir(imgfold):
                cur_img = np.zeros((800,1500)) # init frame
            len_x = int(float(rectangle_list[5]) - float(rectangle_list[3]))
            len_y = int(float(rectangle_list[6]) - float(rectangle_list[4]))
            for x in range(len_x):
                for y in range(len_y):
                    cur_img[int(float(rectangle_list[4])+y)][int(float(rectangle_list[3])+x)] = 255

            cv2.imwrite(imgstr,cur_img)


            # if (curstr) not in os.listdir('/home/songzhuoran/video/video-frame-based-acc/data/baseline_result/' + video_list[0]):
            #     os.mkdir(imgfold) # i.e., /home/songzhuoran/video/video-frame-based-acc/data/baseline_result/ILSVRC2015_val_00000000/00213/

            


