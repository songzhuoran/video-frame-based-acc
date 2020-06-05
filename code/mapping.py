import csv
import os,sys
import re
from PIL import Image, ImageDraw, ImageColor
import cv2
import numpy as np
from copy import deepcopy

## MV: [CurrentFrame, TargetFrame, BlockWidth, BlockHeight, CurrentBlockX, CurBlockY, TargetX, TargetY]

IDX_DIR="/home/songzhuoran/video/video-frame-based-acc/data/idx/"
B_OUT_DIR="/home/songzhuoran/video/video-frame-based-acc/data/mapping_result/"
P_DIR="/home/songzhuoran/video/video-frame-based-acc/data/baseline_result/"
MVS_DIR="/home/songzhuoran/video/video-frame-based-acc/data/mvs/"


# f = open("/home/songzhuoran/video/video-frame-based-acc/VID_val_videos.txt","r")
# video_names = f.readlines()
# f.close()

# # make video directory
# for i in video_names:
#     video_list = i.split(' ') # i.e., ['ILSVRC2015_val_00000000', '1', '0', '464\n']
#     os.mkdir('/home/songzhuoran/video/video-frame-based-acc/data/mapping_result/' + video_list[0])

mvsmat = []
vis = [False] * 3000
classname = "111"
frame_mat = {}
bflist = []  # aka b frame list
pflist = []  # aka b frame list


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
    global frame_mat
    global classname
    global mvsmat
    global bflist
    global pflist

    print("frame mate:")
    print(len(frame_mat))
    print("fcnt:")
    print(fcnt)

    with open(MVS_DIR+classname+".csv","r") as file:
        datainfo = csv.reader(file)
        tmp_class_list=[]
        # print(datainfo)
        for row in datainfo:
            # if fcnt == 322:
            #     print("===== read large table =====")
            if int(row[0]) == fcnt:
                # if fcnt == 322:
                #     print("===== read mv =====")
                w = int(row[2])
                h = int(row[3])
                srcx = int(row[4])
                srcy = int(row[5])
                dstx = int(row[6])
                dsty = int(row[7])
                TargetFrame = '%08d' % int(row[1])
                if (TargetFrame) in os.listdir(B_OUT_DIR+classname):
                    # print("length of frame_mat target list")
                    # print(len(frame_mat[B_OUT_DIR+classname + '/' + TargetFrame]))
                    for class_list in frame_mat[B_OUT_DIR+classname + '/' + TargetFrame]:
                        # if fcnt == 322:
                        #     print("class_list:")
                        #     print(class_list)
                        dst_class_type = class_list.classtype
                        # print(dst_class_type)
                        dst = class_list.img_data
                        for i in range(w):
                            for j in range(h):
                                if check_x_outside(srcx+i) and check_x_outside(dstx+i) and check_y_outside(srcy+j) and check_y_outside(dsty+j):
                                    if img_vis[dst_class_type][srcy+j][srcx+i] == 0:
                                        bframe_img[dst_class_type][srcy+j][srcx+i] = dst[dsty+j][dstx+i]
                                    else :
                                        bframe_img[dst_class_type][srcy+j][srcx+i] = (int(dst[dsty+j][dstx+i]) + int(bframe_img[dst_class_type][srcy+j][srcx+i])) / 2
                                    img_vis[dst_class_type][srcy+j][srcx+i] += 1
                        if len(tmp_class_list)==0:
                            tmp_class_list.append(dst_class_type)
                        else:
                            for i in range(len(tmp_class_list)):
                                if tmp_class_list[i]==dst_class_type:
                                    break
                            if i==(len(tmp_class_list)-1) and tmp_class_list[i]!=dst_class_type:
                                tmp_class_list.append(dst_class_type)
                        
                    


    if ('%08d' % fcnt) not in os.listdir(B_OUT_DIR+classname):
        os.mkdir(B_OUT_DIR+classname +'/'+'%08d' % fcnt)

    tmp_list=[]
    # print("tmp_class_list:")
    # print(len(tmp_class_list))
    # print(B_OUT_DIR+classname + '/' +'%08d' % fcnt)
    for i in tmp_class_list:
        print(B_OUT_DIR+classname + '/' +'%08d' % fcnt+'/'+str(i)+'.png')
        cv2.imwrite(B_OUT_DIR+classname + '/' +'%08d' % fcnt+'/'+str(i)+'.png',bframe_img[i])
        tmp_list.append(Image_mat(i,bframe_img[i]))
        
    frame_mat[B_OUT_DIR+classname +'/'+'%08d' % fcnt] = tmp_list

    # if fcnt % 100 <=2 and fcnt>2:
    #     frame_mat_tmp = deepcopy(frame_mat)
    #     frame_mat.clear()
    #     for i in range(15):
    #         if (B_OUT_DIR+classname +'/'+'%08d' % (fcnt-i)) in frame_mat_tmp :
    #             frame_mat[B_OUT_DIR+classname +'/'+'%08d' % (fcnt-i)] = frame_mat_tmp[B_OUT_DIR+classname +'/'+'%08d' % (fcnt-i)]
    #     for i in range(15):
    #         if (B_OUT_DIR+classname +'/'+'%08d' % (fcnt+i+1)) in frame_mat_tmp :
    #             frame_mat[B_OUT_DIR+classname +'/'+'%08d' % (fcnt+i+1)] = frame_mat_tmp[B_OUT_DIR+classname +'/'+'%08d' % (fcnt+i+1)]
    #     frame_mat_tmp.clear()

    #     for i in pflist:
    #         if ('%08d' % i) not in os.listdir(B_OUT_DIR+classname):
    #             tmp_list=[]
    #             tmp_list.append(Image_mat(0,np.zeros((800,1500))))
    #             frame_mat[B_OUT_DIR+classname + '/' + '%08d' % i] = tmp_list
    #         else:
    #             tmp_list=[]
    #             class_list = os.listdir(B_OUT_DIR+classname + '/' + '%08d' % i+'/')
    #             for classtype in class_list:
    #                 img_str = B_OUT_DIR+classname + '/' + '%08d' % i + '/' + classtype # i.e., /home/songzhuoran/video/video-frame-based-acc/data/baseline_result/ILSVRC2015_val_00161002/00000297/3.png
    #                 cur_img = cv2.imread(img_str,0)
    #                 classtype = re.sub('[.png]', '', classtype) # i.e., 25
    #                 tmp_list.append(Image_mat(int(classtype),cur_img))
    #                     # print(classtype)
    #             frame_mat[B_OUT_DIR+classname + '/' + '%08d' % i] = tmp_list

    


def DFS(fcnt):
    global frame_mat
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
    
    global bflist
    global pflist
    global frame_mat
    global classname
    global mvsmat
    global vis
    bflist = []
    pflist = []
    with open(IDX_DIR+"b/"+classname, "r") as file:
        for row in file:
            bflist.append(int(row)-1)

    with open(IDX_DIR+"p/"+classname, "r") as file:
        for row in file:
            pflist.append(int(row)-1)

    framecnt = pflist[-1] + 1

    for i in pflist:
        vis[i] = True
        if ('%08d' % i) not in os.listdir(B_OUT_DIR+classname):
            tmp_list=[]
            tmp_list.append(Image_mat(0,np.zeros((800,1500))))
            frame_mat[B_OUT_DIR+classname + '/' + '%08d' % i] = tmp_list
        else:
            tmp_list=[]
            class_list = os.listdir(B_OUT_DIR+classname + '/' + '%08d' % i+'/')
            # print(B_OUT_DIR+classname + '/' + '%08d' % i)
            # print(class_list)
            for classtype in class_list:
                img_str = B_OUT_DIR+classname + '/' + '%08d' % i + '/' + classtype # i.e., /home/songzhuoran/video/video-frame-based-acc/data/baseline_result/ILSVRC2015_val_00161002/00000297/3.png
                cur_img = cv2.imread(img_str,0)
                classtype = re.sub('[.png]', '', classtype) # i.e., 25
                tmp_list.append(Image_mat(int(classtype),cur_img))
                    # print(classtype)
            frame_mat[B_OUT_DIR+classname + '/' + '%08d' % i] = tmp_list

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
        global frame_mat
        global classname
        global mvsmat
        global vis
        classname = filename
        vis = [False] * 3000
        mvsmat = []
        frame_mat.clear()
        bframe_gen()




if __name__ == "__main__":
    main()
    

