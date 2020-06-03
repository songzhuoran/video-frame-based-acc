import os

f = open("/home/songzhuoran/video/video-frame-based-acc/final_nms_result.txt","r") 
rectangles = f.readlines()
f.close()

max_0 = -1.0
max_1 = -1.0
for r in rectangles:
    rectangle_list = r.split(' ') # i.e., [214,27,0.9800,621.31,163.67,831.14,360.44\n]
    rectangle_list[6]=rectangle_list[6].replace('\n', '').replace('\r', '') # i.e., 360.44
    if max_0 <= float(rectangle_list[5]):
        max_0 = float(rectangle_list[5])
    if max_1 <= float(rectangle_list[6]):
        max_1 = float(rectangle_list[6])

print(max_0)
print(max_1)

#1492.43
#774.08

