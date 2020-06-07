import os
import numpy as np

Map_DIR="/home/songzhuoran/video/video-frame-based-acc/data/mapping_result_mthread/"
Bench_DIR="/home/songzhuoran/video/video-frame-based-acc/data/benchmark_result/"

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
                if classtype not in os.listdir(Bench_DIR+videofile+'/'+imagefile):
                    bench_img = np.zeros([800,1500])
                else:
                    bench_img = cv2.imread(Bench_DIR+videofile+'/'+imagefile+'/'+classtype,0)
                num_overlap = 0
                num_union = 0
                for i in range(map_img.size):
                    if map_img[i]==bench_img[i] and map_img[i]!=0:
                        num_overlap = num_overlap + 1
                        num_union = num_union + 1
                    else if map_img[i]!=bench_img[i]:
                        num_union = num_union + 1
                iou = iou + num_overlap/num_union
        iou = iou / count
        print(videofile)
        print(iou)


if __name__ == "__main__":
    main()
    
