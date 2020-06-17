import os
import numpy as np
import cv2

Map_DIR="/home/songzhuoran/video/video-frame-based-acc/data/baseline_result/"
Bench_DIR="/home/songzhuoran/video/video-frame-based-acc/data/benchmark_result/"

record_file = open("/home/songzhuoran/video/video-frame-based-acc/result_baseline.csv", "w")

def main():
    videofiles= os.listdir(Map_DIR)
    for videofile in videofiles: # i.e., ILSVRC2015_val_00177000\
        iou = 0.0
        count = 0
        imagefiles= os.listdir(Map_DIR+videofile)
        for imagefile in imagefiles: # i.e., 00000014
            classtypes = os.listdir(Map_DIR+videofile+'/'+imagefile)
            for classtype in classtypes:
                count = count + 1
                map_img = cv2.imread(Map_DIR+videofile+'/'+imagefile+'/'+classtype,0)
                if imagefile in os.listdir(Bench_DIR+videofile):
                    if classtype not in os.listdir(Bench_DIR+videofile+'/'+imagefile):
                        bench_img = np.zeros([800,1500])
                    else:
                        bench_img = cv2.imread(Bench_DIR+videofile+'/'+imagefile+'/'+classtype,0)
                    num_overlap = 0
                    num_union = 0
                    for i in range(map_img.shape[0]):
                        for j in range(map_img.shape[1]):
                            if map_img[i][j]==bench_img[i][j] and map_img[i][j]!=0:
                                num_overlap = num_overlap + 1
                                num_union = num_union + 1
                            elif map_img[i][j]!=bench_img[i][j]:
                                num_union = num_union + 1
                    if num_union!=0:
                        iou = float(iou + float(num_overlap/num_union))
        iou = iou / count
        print(videofile)
        print(iou)
        record_file.write(videofile + "," + str(iou) + "\n")


if __name__ == "__main__":
    main()
    
