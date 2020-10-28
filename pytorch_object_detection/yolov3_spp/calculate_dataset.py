"""
该脚本有3个功能：
1.统计训练集和验证集的数据并生成相应.txt文件
2.创建data.data文件，记录classes个数, train以及val数据集文件(.txt)路径和label.names文件路径
3.根据yolov3-spp.cfg创建my_yolov3.cfg文件修改其中的predictor filters以及yolo classes参数(这两个参数是根据类别数改变的)
"""
import os

train_annotation_dir = "./my_yolo_dataset/train/labels"
val_annotation_dir = "./my_yolo_dataset/val/labels"
classes_label = "./data/my_data_label.names"
cfg_path = "./cfg/yolov3-spp.cfg"

assert os.path.exists(train_annotation_dir), "train_annotation_dir not exist!"
assert os.path.exists(val_annotation_dir), "val_annotation_dir not exist!"
assert os.path.exists(classes_label), "classes_label not exist!"
assert os.path.exists(cfg_path), "cfg_path not exist!"


def calculate_data_txt(txt_path, dataset_dir):
    # create my_data.txt file that record image list
    with open(txt_path, "w") as w:
        for file_name in os.listdir(dataset_dir):
            if file_name == "classes.txt":
                continue

            img_path = os.path.join(dataset_dir.replace("labels", "images"),
                                    file_name.split(".")[0]) + ".jpg"
            line = img_path + "\n"
            assert os.path.exists(img_path), "file:{} not exist!".format(img_path)
            w.write(line)


def create_data_data(create_data_path, label_path, train_path, val_path, classes_info):
    # create my_data.data file that record classes, train, valid and names info.
    # shutil.copyfile(label_path, "./data/my_data_label.names")
    with open(create_data_path, "w") as w:
        w.write("classes={}".format(len(classes_info)) + "\n")  # 记录类别个数
        w.write("train={}".format(train_path) + "\n")           # 记录训练集对应txt文件路径
        w.write("valid={}".format(val_path) + "\n")             # 记录验证集对应txt文件路径
        w.write("names=data/my_data_label.names" + "\n")        # 记录label.names文件路径


def change_and_create_cfg_file(classes_info, save_cfg_path="./cfg/my_yolov3.cfg"):
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

    with open(save_cfg_path, "w") as w:
        w.writelines(cfg_lines)


def main():
    # 统计训练集和验证集的数据并生成相应txt文件
    train_txt_path = "data/my_train_data.txt"
    val_txt_path = "data/my_val_data.txt"
    calculate_data_txt(train_txt_path, train_annotation_dir)
    calculate_data_txt(val_txt_path, val_annotation_dir)

    classes_info = [line.strip() for line in open(classes_label, "r").readlines() if len(line.strip()) > 0]
    # 创建data.data文件，记录classes个数, train以及val数据集文件(.txt)路径和label.names文件路径
    create_data_data("./data/my_data.data", classes_label, train_txt_path, val_txt_path, classes_info)

    # 根据yolov3-spp.cfg创建my_yolov3.cfg文件修改其中的predictor filters以及yolo classes参数(这两个参数是根据类别数改变的)
    change_and_create_cfg_file(classes_info)


if __name__ == '__main__':
    main()
