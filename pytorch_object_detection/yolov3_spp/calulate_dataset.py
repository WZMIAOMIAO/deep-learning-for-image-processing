import os
import shutil

dataset_dir = "../coco/labels/train2017/"
classes_label = "./data/my_classes.txt"
cfg_path = "./cfg/yolov3-spp.cfg"

assert os.path.exists(dataset_dir), "dataset_dir not exist!"
assert os.path.exists(classes_label), "classes_label not exist!"
assert os.path.exists(cfg_path), "cfg_path not exist!"

# create my_data.txt file that record image list
with open("./data/my_data.txt", "w") as w:
    for file_name in os.listdir(dataset_dir):
        if file_name == "classes.txt":
            continue

        line = os.path.join(dataset_dir, file_name.split(".")[0]) + "\n"
        w.write(line)

# create my_data.data file that record classes, train, valid and names info.
shutil.copyfile(classes_label, "./data/my_data_label.names")
classes_info = [line.strip() for line in open(classes_label, "r").readlines() if len(line.strip()) > 0]
with open("./data/my_data.data", "w") as w:
    w.write("classes={}".format(len(classes_info)) + "\n")
    w.write("train=data/my_data.txt" + "\n")
    w.write("valid=data/my_data.txt" + "\n")
    w.write("names=data/my_data_label.names" + "\n")

# create my_yolov3.cfg file changed predictor filters and yolo classes param.
# this operation only deal with yolov3-spp.cfg
filters_lines = [636, 722, 809]
classes_lines = [643, 729, 816]
cfg_lines = open(cfg_path, "r").readlines()

for i in filters_lines:
    assert "filters" in cfg_lines[i-1], "filters param is not in line:{}".format(i-1)
    output_num = (5 + len(classes_info)) * 3
    cfg_lines[i-1] = "filters={}\n".format(output_num)

for i in classes_lines:
    assert "classes" in cfg_lines[i-1], "classes param is not in line:{}".format(i-1)
    cfg_lines[i-1] = "classes={}\n".format(len(classes_info))

with open("./cfg/my_yolov3.cfg", "w") as w:
    w.writelines(cfg_lines)
