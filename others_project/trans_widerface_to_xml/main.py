import os

from tqdm import tqdm
import cv2
from create_xml import create_pascal_voc_xml


def create_xml(labels: list, img_root: str, img_path: str, save_root: str) -> bool:
    source_dict = {'database': 'The WIDERFACE2017 Database',
                   'annotation': 'WIDERFACE 2017',
                   'image': 'WIDERFACE'}

    img_full_path = os.path.join(img_root, img_path)
    if os.path.exists(img_full_path):
        im = cv2.imread(img_full_path)
        im_shape = im.shape
    else:
        print(f"Warning: {img_path} does not exist, can't read image shape.")
        im_shape = (0, 0, 0)

    ob_list = []
    for ob in labels:
        if ob[7] == '1':
            # invalid face image, skip
            continue

        if int(ob[2]) <= 0 or int(ob[3]) <= 0:
            print(f"Warning: find bbox w or h <= 0, in {img_path}, skip.")
            continue

        ob_dict = {'name': 'face',
                   'truncated': '0' if ob[8] == '0' else '1',
                   'difficult': '1' if ob[4] == '2' or ob[8] == '2' else '0',
                   'xmin': ob[0], 'ymin': ob[1],
                   'xmax': str(int(ob[0]) + int(ob[2])),
                   'ymax': str(int(ob[1]) + int(ob[3])),
                   'blur': ob[4], 'expression': ob[5],
                   'illumination': ob[6], 'invalid': ob[7],
                   'occlusion': ob[8], 'pose': ob[9]}

        # if ob[7] == '1':
        #     cv2.rectangle(im, (int(ob_dict['xmin']), int(ob_dict['ymin'])),
        #                   (int(ob_dict['xmax']), int(ob_dict['ymax'])),
        #                   (0, 0, 255))
        #     cv2.imshow("s", im)
        #     cv2.waitKey(0)

        ob_list.append(ob_dict)
    
    if len(ob_list) == 0: 
        print(f"in {img_path}, no object, skip.")
        return False

    create_pascal_voc_xml(filename=img_path,
                          years="WIDERFACE2017",
                          source_dict=source_dict,
                          objects_list=ob_list,
                          im_shape=im_shape,
                          save_root=save_root)

    return True


def parse_wider_txt(data_root: str, split: str, save_root: str):
    """
    refer to: torchvision.dataset.widerface.py
    :param data_root:
    :param split:
    :param save_root:
    :return:
    """
    assert split in ['train', 'val'], f"split must be in ['train', 'val'], got {split}"

    if os.path.exists(save_root) is False:
        os.makedirs(save_root)

    txt_path = os.path.join(data_root, 'wider_face_split', f'wider_face_{split}_bbx_gt.txt')
    img_root = os.path.join(data_root, f'WIDER_{split}', 'images')
    with open(txt_path, "r") as f:
        lines = f.readlines()
        file_name_line, num_boxes_line, box_annotation_line = True, False, False
        num_boxes, box_counter, idx = 0, 0, 0
        labels = []
        xml_list = []
        progress_bar = tqdm(lines)
        for line in progress_bar:
            line = line.rstrip()
            if file_name_line:
                img_path = line
                file_name_line = False
                num_boxes_line = True
            elif num_boxes_line:
                num_boxes = int(line)
                num_boxes_line = False
                box_annotation_line = True
            elif box_annotation_line:
                box_counter += 1
                line_split = line.split(" ")
                line_values = [x for x in line_split]
                labels.append(line_values)
                if box_counter >= num_boxes:
                    box_annotation_line = False
                    file_name_line = True

                    if num_boxes == 0:
                        print(f"in {img_path}, no object, skip.")
                    else:
                        if create_xml(labels, img_root, img_path, save_root):
                            # 只记录有目标的xml文件
                            xml_list.append(img_path.split("/")[-1].split(".")[0])

                    box_counter = 0
                    labels.clear()
                    idx += 1
                    progress_bar.set_description(f"{idx} images")
            else:
                raise RuntimeError("Error parsing annotation file {}".format(txt_path))

        with open(split+'.txt', 'w') as w:
            w.write("\n".join(xml_list))


parse_wider_txt("/data/wider_face/",
                "val",
                "./annotation/")
