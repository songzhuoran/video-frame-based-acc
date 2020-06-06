import os

B_OUT_DIR="/home/songzhuoran/video/video-frame-based-acc/data/mapping_result/"

def main():
    mvsfiles= os.listdir(B_OUT_DIR)
    f1 = open('/home/songzhuoran/video/video-frame-based-acc/test.txt','w')
    for filename in mvsfiles: # i.e., ILSVRC2015_val_00177000
        print(filename)
        f1.write(filename)
        f1.write("\r\n")




if __name__ == "__main__":
    main()