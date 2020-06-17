import csv
import os,sys
import re
import cv2
import numpy as np

DATA_DIR="/home/songzhuoran/benchmark/"
DET_DIR="/home/songzhuoran/video/video-frame-based-acc/data/baseline_result/"
R_DIR="/home/songzhuoran/video/video-frame-based-acc/data/eup_result_2/"

interval = 3

filename = sys.argv[1]
f = open(filename,'r')
videofiles = f.readlines()
f.close()
for videofile in videofiles: # i.e., ILSVRC2015_val_00177000
    videofile = re.sub('\n', '', videofile)
    imagefiles = os.listdir(DATA_DIR+videofile)
    for imagefile in imagefiles:
        cur_num = int(re.sub('[.JPEG]', '', imagefile))
        if cur_num != 0:
            prevImg=cv2.imread(DATA_DIR+videofile+"/"+ '%06d' % (cur_num-1) + ".JPEG")
            # print(DATA_DIR+videofile+"/"+ '%06d' % (cur_num-1) + ".JPEG")
            prevImg = cv2.cvtColor(prevImg,cv2.COLOR_BGR2GRAY)
            nextImg=cv2.imread(DATA_DIR+videofile+"/"+imagefile)
            nextImg = cv2.cvtColor(nextImg,cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(prevImg,nextImg, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            # print(flow.shape[0])
            if cur_num % interval != 0:
                ref_num = cur_num - cur_num % interval
                if ('%08d' % ref_num) in os.listdir(DET_DIR+videofile):
                    classfiles = os.listdir(DET_DIR+videofile+"/"+'%08d' % ref_num)
                    for classfile in classfiles:
                        ref_det_img=cv2.imread(DET_DIR+videofile+"/"+'%08d' % ref_num+"/"+classfile)
                        # print(ref_det_img.shape)
                        cur_det_img=np.zeros((800,1500))
                        x_flow = 0
                        y_flow = 0
                        num_count = 0
                        for i in range(1500):
                            for j in range(800):
                                # print(ref_det_img[j][i])
                                if ref_det_img[j][i][0]!=0 and j<flow.shape[0] and i<flow.shape[1]:
                                    x_flow = x_flow+flow[j][i][0]
                                    y_flow = y_flow+flow[j][i][1]
                                    num_count = num_count + 1
                        if num_count!=0:
                            x_flow = int(x_flow/num_count)
                            y_flow = int(y_flow/num_count)
                        for i in range(1500):
                            for j in range(800):
                                if (j-y_flow)<800 and (i-x_flow)<1500:
                                    cur_det_img[j][i]=ref_det_img[j-y_flow][i-x_flow][0]
                                else:
                                    cur_det_img[j][i]=0
                        cv2.imwrite(R_DIR+videofile+"/"+'%08d' % ref_num+"/"+classfile,cur_det_img)
                        print(R_DIR+videofile+"/"+'%08d' % ref_num+"/"+classfile)
                        
                    








