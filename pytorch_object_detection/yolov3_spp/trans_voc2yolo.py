import os
from tqdm import tqdm
from lxml import etree
import json


voc_root = "/home/w180662/my_project/my_github/pytorch_object_detection/VOCdevkit"
voc_version = "VOC2012"

train_txt = "train.txt"
val_txt = "val.txt"

train_save_path = "/home/w180662/my_project/yolo_annotation_train"
val_save_path = "/home/w180662/my_project/yolo_annotation_val"

voc_images_path = os.path.join(voc_root, voc_version, "JPEGImages")
voc_xml_path = os.path.join(voc_root, voc_version, "Annotations")
train_txt_path = os.path.join(voc_root, voc_version, "ImageSets", "Main", train_txt)
val_txt_path = os.path.join(voc_root, voc_version, "ImageSets", "Main", val_txt)

assert os.path.exists(voc_images_path), "VOC images path not exist..."
assert os.path.exists(voc_xml_path), "VOC xml path not exist..."
assert os.path.exists(train_txt_path), "VOC train txt file not exist..."
assert os.path.exists(val_txt_path), "VOC val txt file not exist..."
assert os.path.exists(train_save_path), "path saving yolo annotation not exist..."
assert os.path.exists(val_save_path), "path saving yolo annotation not exist..."


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


def translate_info(file_names: list, save_path: str, class_dict: dict):
    for file in tqdm(file_names, desc="translate train file..."):
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
        with open(os.path.join(save_path, file + ".txt"), "w") as f:
            for index, obj in enumerate(data["object"]):
                # 获取每个object的box信息
                xmin = float(obj["bndbox"]["xmin"])
                xmax = float(obj["bndbox"]["xmax"])
                ymin = float(obj["bndbox"]["ymin"])
                ymax = float(obj["bndbox"]["ymax"])
                class_name = obj["name"]
                class_index = class_dict[class_name] - 1

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


def main():
    # read class_indict
    try:
        json_file = open('./pascal_voc_classes.json', 'r')
        class_dict = json.load(json_file)
    except Exception as e:
        print(e)
        exit(-1)

    # 读取train.txt中的所有行信息，删除空行
    with open(train_txt_path, "r") as r:
        train_file_names = [i for i in r.read().splitlines() if len(i.strip()) > 0]
    # voc信息转yolo
    translate_info(train_file_names, train_save_path, class_dict)

    # 读取val.txt中的所有行信息，删除空行
    with open(val_txt_path, "r") as r:
        val_file_names = [i for i in r.read().splitlines() if len(i.strip()) > 0]
    # voc信息转yolo
    translate_info(val_file_names, val_save_path, class_dict)


if __name__ == "__main__":
    main()
