"""
本脚本有两个功能：
1.将voc数据集标注信息(.xml)转为yolo标注格式(.txt)，并将图像文件复制到相应文件夹
2.根据json标签文件，生成对应names标签(my_data_label.names)
"""
import os
from tqdm import tqdm
from lxml import etree
import json
import shutil


# voc数据集根目录以及版本
voc_root = "/data/VOCdevkit"
voc_version = "VOC2012"

# 转换的训练集以及验证集对应txt文件
train_txt = "train.txt"
val_txt = "val.txt"

# 转换后的文件保存目录
save_file_root = "./my_yolo_dataset"

# label标签对应json文件
label_json_path = './data/pascal_voc_classes.json'

# 拼接出voc的images目录，xml目录，txt目录
voc_images_path = os.path.join(voc_root, voc_version, "JPEGImages")
voc_xml_path = os.path.join(voc_root, voc_version, "Annotations")
train_txt_path = os.path.join(voc_root, voc_version, "ImageSets", "Main", train_txt)
val_txt_path = os.path.join(voc_root, voc_version, "ImageSets", "Main", val_txt)

# 检查文件/文件夹都是否存在
assert os.path.exists(voc_images_path), "VOC images path not exist..."
assert os.path.exists(voc_xml_path), "VOC xml path not exist..."
assert os.path.exists(train_txt_path), "VOC train txt file not exist..."
assert os.path.exists(val_txt_path), "VOC val txt file not exist..."
assert os.path.exists(label_json_path), "label_json_path does not exist..."
if os.path.exists(save_file_root) is False:
    os.makedirs(save_file_root)


def parse_xml_to_dict(xml):
    """
    将xml文件解析成字典形式，参考tensorflow的recursive_parse_xml_to_dict
    Args：
        xml: xml tree obtained by parsing XML file contents using lxml.etree

    Returns:
        Python dictionary holding XML contents.
    """

    if len(xml) == 0:  # 遍历到底层，直接返回tag对应的信息
        return {xml.tag: xml.text}

    result = {}
    for child in xml:
        child_result = parse_xml_to_dict(child)  # 递归遍历标签信息
        if child.tag != 'object':
            result[child.tag] = child_result[child.tag]
        else:
            if child.tag not in result:  # 因为object可能有多个，所以需要放入列表里
                result[child.tag] = []
            result[child.tag].append(child_result[child.tag])
    return {xml.tag: result}


def translate_info(file_names: list, save_root: str, class_dict: dict, train_val='train'):
    """
    将对应xml文件信息转为yolo中使用的txt文件信息
    :param file_names:
    :param save_root:
    :param class_dict:
    :param train_val:
    :return:
    """
    save_txt_path = os.path.join(save_root, train_val, "labels")
    if os.path.exists(save_txt_path) is False:
        os.makedirs(save_txt_path)
    save_images_path = os.path.join(save_root, train_val, "images")
    if os.path.exists(save_images_path) is False:
        os.makedirs(save_images_path)

    for file in tqdm(file_names, desc="translate {} file...".format(train_val)):
        # 检查下图像文件是否存在
        img_path = os.path.join(voc_images_path, file + ".jpg")
        assert os.path.exists(img_path), "file:{} not exist...".format(img_path)

        # 检查xml文件是否存在
        xml_path = os.path.join(voc_xml_path, file + ".xml")
        assert os.path.exists(xml_path), "file:{} not exist...".format(xml_path)

        # read xml
        with open(xml_path) as fid:
            xml_str = fid.read()
        xml = etree.fromstring(xml_str)
        data = parse_xml_to_dict(xml)["annotation"]
        img_height = int(data["size"]["height"])
        img_width = int(data["size"]["width"])

        # write object info into txt
        assert "object" in data.keys(), "file: '{}' lack of object key.".format(xml_path)
        if len(data["object"]) == 0:
            # 如果xml文件中没有目标就直接忽略该样本
            print("Warning: in '{}' xml, there are no objects.".format(xml_path))
            continue

        with open(os.path.join(save_txt_path, file + ".txt"), "w") as f:
            for index, obj in enumerate(data["object"]):
                # 获取每个object的box信息
                xmin = float(obj["bndbox"]["xmin"])
                xmax = float(obj["bndbox"]["xmax"])
                ymin = float(obj["bndbox"]["ymin"])
                ymax = float(obj["bndbox"]["ymax"])
                class_name = obj["name"]
                class_index = class_dict[class_name] - 1  # 目标id从0开始

                # 进一步检查数据，有的标注信息中可能有w或h为0的情况，这样的数据会导致计算回归loss为nan
                if xmax <= xmin or ymax <= ymin:
                    print("Warning: in '{}' xml, there are some bbox w/h <=0".format(xml_path))
                    continue

                # 将box信息转换到yolo格式
                xcenter = xmin + (xmax - xmin) / 2
                ycenter = ymin + (ymax - ymin) / 2
                w = xmax - xmin
                h = ymax - ymin

                # 绝对坐标转相对坐标，保存6位小数
                xcenter = round(xcenter / img_width, 6)
                ycenter = round(ycenter / img_height, 6)
                w = round(w / img_width, 6)
                h = round(h / img_height, 6)

                info = [str(i) for i in [class_index, xcenter, ycenter, w, h]]

                if index == 0:
                    f.write(" ".join(info))
                else:
                    f.write("\n" + " ".join(info))

        # copy image into save_images_path
        path_copy_to = os.path.join(save_images_path, img_path.split(os.sep)[-1])
        if os.path.exists(path_copy_to) is False:
            shutil.copyfile(img_path, path_copy_to)


def create_class_names(class_dict: dict):
    keys = class_dict.keys()
    with open("./data/my_data_label.names", "w") as w:
        for index, k in enumerate(keys):
            if index + 1 == len(keys):
                w.write(k)
            else:
                w.write(k + "\n")


def main():
    # read class_indict
    json_file = open(label_json_path, 'r')
    class_dict = json.load(json_file)

    # 读取train.txt中的所有行信息，删除空行
    with open(train_txt_path, "r") as r:
        train_file_names = [i for i in r.read().splitlines() if len(i.strip()) > 0]
    # voc信息转yolo，并将图像文件复制到相应文件夹
    translate_info(train_file_names, save_file_root, class_dict, "train")

    # 读取val.txt中的所有行信息，删除空行
    with open(val_txt_path, "r") as r:
        val_file_names = [i for i in r.read().splitlines() if len(i.strip()) > 0]
    # voc信息转yolo，并将图像文件复制到相应文件夹
    translate_info(val_file_names, save_file_root, class_dict, "val")

    # 创建my_data_label.names文件
    create_class_names(class_dict)


if __name__ == "__main__":
    main()
