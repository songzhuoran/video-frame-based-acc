import csv
import os,sys
import re
from PIL import Image, ImageDraw, ImageColor
import cv2
import numpy as np

## MV: [CurrentFrame, TargetFrame, BlockWidth, BlockHeight, CurrentBlockX, CurBlockY, TargetX, TargetY]

IDX_DIR="/home/songzhuoran/video/video-frame-based-acc/data/idx/"
B_OUT_DIR="/home/songzhuoran/video/video-frame-based-acc/data/mapping_result_save_mem/"
P_DIR="/home/songzhuoran/video/video-frame-based-acc/data/baseline_result/"
MVS_DIR="/home/songzhuoran/video/video-frame-based-acc/data/mvs/"


# f = open("/home/songzhuoran/video/video-frame-based-acc/VID_val_videos.txt","r")
# video_names = f.readlines()
# f.close()

# # make video directory
# for i in video_names:
#     video_list = i.split(' ') # i.e., ['ILSVRC2015_val_00000000', '1', '0', '464\n']
#     os.mkdir('/home/songzhuoran/video/video-frame-based-acc/data/mapping_result_save_mem/' + video_list[0])

mvsmat = []
vis = [False] * 3000
classname = "111"


class Image_mat:
    def __init__(self, classtype,img_data): # i.e., 23, [...]
        self.classtype = classtype
        self.img_data = img_data


def check_x_outside(point_x):
    if 0 <= point_x < 1500:
        return True
    else:
        return False

def check_y_outside(point_y):
    if 0 <= point_y < 800:
        return True
    else:
        return False


def bframe_gen_kernel(fcnt):
    # print(fcnt)
    bframe_img = np.zeros((35,800,1500),dtype="uint8")
    img_vis = np.zeros((35,800,1500))
    global classname
    global mvsmat

    print("frame mate:")
    print("fcnt:")
    print(fcnt)

    with open(MVS_DIR+classname+".csv","r") as file:
        datainfo = csv.reader(file)
        tmp_class_list=[]
        # print(datainfo)
        for row in datainfo:
            if int(row[0]) == fcnt:
                if fcnt == 322:
                    print("===== read mv =====")
                w = int(row[2])
                h = int(row[3])
                srcx = int(row[4])
                srcy = int(row[5])
                dstx = int(row[6])
                dsty = int(row[7])
                TargetFrame = '%08d' % int(row[1])
                if (TargetFrame) in os.listdir(B_OUT_DIR+classname):
                    if fcnt == 322:
                        print("B_OUT_DIR+classname + '/' + TargetFrame: ")
                        print(B_OUT_DIR+classname + '/' + TargetFrame)
                    class_lists = os.listdir(B_OUT_DIR+classname + '/' + TargetFrame)
                    for class_list in class_lists:
                        img_str = B_OUT_DIR+classname + '/' + TargetFrame + '/' + class_list # i.e., /home/songzhuoran/video/video-frame-based-acc/data/baseline_result/ILSVRC2015_val_00161002/00000297/3.png
                        cur_img = cv2.imread(img_str,0)
                        dst = cur_img
                        dst_class_type = re.sub('[.png]', '', class_list) # i.e., 25
                        # print(class_list)
                        # print(dst_class_type)
                        for i in range(w):
                            for j in range(h):
                                if check_x_outside(srcx+i) and check_x_outside(dstx+i) and check_y_outside(srcy+j) and check_y_outside(dsty+j):
                                    if img_vis[int(dst_class_type)][srcy+j][srcx+i] == 0:
                                        bframe_img[int(dst_class_type)][srcy+j][srcx+i] = dst[dsty+j][dstx+i]
                                    else :
                                        bframe_img[int(dst_class_type)][srcy+j][srcx+i] = (int(dst[dsty+j][dstx+i]) + int(bframe_img[int(dst_class_type)][srcy+j][srcx+i])) / 2
                                    img_vis[int(dst_class_type)][srcy+j][srcx+i] += 1
                        if len(tmp_class_list)==0:
                            tmp_class_list.append(int(dst_class_type))
                        else:
                            for tmp in tmp_class_list:
                                if tmp!=int(dst_class_type):
                                    tmp_class_list.append(int(dst_class_type))                        
                    


    if ('%08d' % fcnt) not in os.listdir(B_OUT_DIR+classname):
        os.mkdir(B_OUT_DIR+classname +'/'+'%08d' % fcnt)

    # print("tmp_class_list:")
    # print(len(tmp_class_list))
    # print(B_OUT_DIR+classname + '/' +'%08d' % fcnt)
    for i in tmp_class_list:
        print(B_OUT_DIR+classname + '/' +'%08d' % fcnt+'/'+str(i)+'.png')
        cv2.imwrite(B_OUT_DIR+classname + '/' +'%08d' % fcnt+'/'+str(i)+'.png',bframe_img[i])
    


def DFS(fcnt):
    global classname
    global mvsmat
    global vis
    # print(vis)
    if vis[fcnt]:
        return True
    else:
        for i in mvsmat[fcnt]:
            DFS(i)
        bframe_gen_kernel(fcnt)
        vis[fcnt] = True
        return True
    

def bframe_gen():
    
    bflist = []  # aka b frame list
    pflist = []  # aka b frame list
    global classname
    global mvsmat
    global vis
    with open(IDX_DIR+"b/"+classname, "r") as file:
        for row in file:
            bflist.append(int(row)-1)

    with open(IDX_DIR+"p/"+classname, "r") as file:
        for row in file:
            pflist.append(int(row)-1)

    framecnt = pflist[-1] + 1

    for i in pflist:
        vis[i] = True

    for i in range(framecnt):
        mvsmat.append(set())

    with open(MVS_DIR+classname+".csv","r") as file:
        datainfo = csv.reader(file)
        for row in datainfo:
            mvsmat[int(row[0])].add(int(row[1]))
    
    
    for i in bflist:
        # print(i)
        if not vis[i]:
            DFS(i)

    for i in range(framecnt):
        if not vis[i]:
            print("ERROR")




def main():
    mvsfiles= os.listdir(B_OUT_DIR)
    for filename in mvsfiles: # i.e., ILSVRC2015_val_00177000
        global classname
        global mvsmat
        global vis
        classname = filename
        vis = [False] * 3000
        mvsmat = []
        bframe_gen()




if __name__ == "__main__":
    main()
    

