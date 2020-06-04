file_to_write="/home/songzhuoran/video/video-frame-based-acc/tmp_nms_result.txt"
file_to_read="/home/songzhuoran/video/video-frame-based-acc/det_VID_val_videos_all.txt"

fwrite = open(file_to_write,"w")
with open(file_to_read, "r") as file:
    line = file.readline()
    linetemp = ""
    while line:
        if linetemp == "":
            linetemp = line
            line = file.readline()
        else:
            contenttemp = linetemp.split()
            content = line.split()
            if contenttemp[0] == content[0] and contenttemp[1] == content[1]:
                if float(contenttemp[2]) < float(content[2]):
                    linetemp = line
                    line = file.readline()
                else:
                    line = file.readline()
            else:
                if float(contenttemp[2]) > 0.3:
                    fwrite.write(linetemp)
                # print(linetemp)
                linetemp = line
                line = file.readline()
fwrite.close()
