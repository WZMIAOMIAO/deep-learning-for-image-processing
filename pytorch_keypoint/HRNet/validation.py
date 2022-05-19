"""
该脚本用于调用训练好的模型权重去计算验证集/测试集的COCO指标
"""

import os
import json

import torch
from tqdm import tqdm
import numpy as np

from model import HighResolutionNet
from train_utils import EvalCOCOMetric
from my_dataset_coco import CocoKeypoint
import transforms


def summarize(self, catId=None):
    """
    Compute and display summary metrics for evaluation results.
    Note this functin can *only* be applied on the default parameter setting
    """

    def _summarize(ap=1, iouThr=None, areaRng='all', maxDets=100):
        p = self.params
        iStr = ' {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}'
        titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
        typeStr = '(AP)' if ap == 1 else '(AR)'
        iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) \
            if iouThr is None else '{:0.2f}'.format(iouThr)

        aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
        mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]

        if ap == 1:
            # dimension of precision: [TxRxKxAxM]
            s = self.eval['precision']
            # IoU
            if iouThr is not None:
                t = np.where(iouThr == p.iouThrs)[0]
                s = s[t]

            if isinstance(catId, int):
                s = s[:, :, catId, aind, mind]
            else:
                s = s[:, :, :, aind, mind]

        else:
            # dimension of recall: [TxKxAxM]
            s = self.eval['recall']
            if iouThr is not None:
                t = np.where(iouThr == p.iouThrs)[0]
                s = s[t]

            if isinstance(catId, int):
                s = s[:, catId, aind, mind]
            else:
                s = s[:, :, aind, mind]

        if len(s[s > -1]) == 0:
            mean_s = -1
        else:
            mean_s = np.mean(s[s > -1])

        print_string = iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s)
        return mean_s, print_string

    stats, print_list = [0] * 10, [""] * 10
    stats[0], print_list[0] = _summarize(1, maxDets=20)
    stats[1], print_list[1] = _summarize(1, maxDets=20, iouThr=.5)
    stats[2], print_list[2] = _summarize(1, maxDets=20, iouThr=.75)
    stats[3], print_list[3] = _summarize(1, maxDets=20, areaRng='medium')
    stats[4], print_list[4] = _summarize(1, maxDets=20, areaRng='large')
    stats[5], print_list[5] = _summarize(0, maxDets=20)
    stats[6], print_list[6] = _summarize(0, maxDets=20, iouThr=.5)
    stats[7], print_list[7] = _summarize(0, maxDets=20, iouThr=.75)
    stats[8], print_list[8] = _summarize(0, maxDets=20, areaRng='medium')
    stats[9], print_list[9] = _summarize(0, maxDets=20, areaRng='large')

    print_info = "\n".join(print_list)

    if not self.eval:
        raise Exception('Please run accumulate() first')

    return stats, print_info


def save_info(coco_evaluator,
              save_name: str = "record_mAP.txt"):
    # calculate COCO info for all keypoints
    coco_stats, print_coco = summarize(coco_evaluator)

    # 将验证结果保存至txt文件中
    with open(save_name, "w") as f:
        record_lines = ["COCO results:", print_coco]
        f.write("\n".join(record_lines))


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("Using {} device training.".format(device.type))

    data_transform = {
        "val": transforms.Compose([
            transforms.AffineTransform(scale=(1.25, 1.25), fixed_size=args.resize_hw),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    }

    # read class_indict
    label_json_path = args.label_json_path
    assert os.path.exists(label_json_path), "json file {} dose not exist.".format(label_json_path)
    with open(label_json_path, 'r') as f:
        person_coco_info = json.load(f)

    data_root = args.data_path

    # 注意这里的collate_fn是自定义的，因为读取的数据包括image和targets，不能直接使用默认的方法合成batch
    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using %g dataloader workers' % nw)

    # load validation data set
    val_dataset = CocoKeypoint(data_root, "val", transforms=data_transform["val"], det_json_path=None)
    # VOCdevkit -> VOC2012 -> ImageSets -> Main -> val.txt
    # val_dataset = VOCInstances(data_root, year="2012", txt_name="val.txt", transforms=data_transform["val"])
    val_dataset_loader = torch.utils.data.DataLoader(val_dataset,
                                                     batch_size=batch_size,
                                                     shuffle=False,
                                                     pin_memory=True,
                                                     num_workers=nw,
                                                     collate_fn=val_dataset.collate_fn)

    # create model
    model = HighResolutionNet()

    # 载入你自己训练好的模型权重
    weights_path = args.weights_path
    assert os.path.exists(weights_path), "not found {} file.".format(weights_path)
    model.load_state_dict(torch.load(weights_path, map_location='cpu'))
    # print(model)
    model.to(device)

    # evaluate on the val dataset
    key_metric = EvalCOCOMetric(val_dataset.coco, "keypoints", "key_results.json")
    model.eval()
    with torch.no_grad():
        for images, targets in tqdm(val_dataset_loader, desc="validation..."):
            # 将图片传入指定设备device
            images = images.to(device)

            # inference
            outputs = model(images)
            if args.flip:
                flipped_images = transforms.flip_images(images)
                flipped_outputs = model(flipped_images)
                flipped_outputs = transforms.flip_back(flipped_outputs, person_coco_info["flip_pairs"])
                # feature is not aligned, shift flipped heatmap for higher accuracy
                # https://github.com/leoxiaobin/deep-high-resolution-net.pytorch/issues/22
                flipped_outputs[..., 1:] = flipped_outputs.clone()[..., 0:-1]
                outputs = (outputs + flipped_outputs) * 0.5

            # decode keypoint
            reverse_trans = [t["reverse_trans"] for t in targets]
            outputs = transforms.get_final_preds(outputs, reverse_trans, post_processing=True)

            key_metric.update(targets, outputs)

    key_metric.synchronize_results()
    key_metric.evaluate()

    save_info(key_metric.coco_evaluator, "keypoint_record_mAP.txt")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__)

    # 使用设备类型
    parser.add_argument('--device', default='cuda:0', help='device')

    parser.add_argument('--resize-hw', type=list, default=[256, 192], help="resize for predict")
    # 是否开启图像翻转
    parser.add_argument('--flip', type=bool, default=True, help='whether using flipped images')

    # 数据集的根目录
    parser.add_argument('--data-path', default='/data/coco2017', help='dataset root')

    # 训练好的权重文件
    parser.add_argument('--weights-path', default='./pose_hrnet_w32_256x192.pth', type=str, help='training weights')

    # batch size
    parser.add_argument('--batch-size', default=1, type=int, metavar='N',
                        help='batch size when validation.')
    # 类别索引和类别名称对应关系
    parser.add_argument('--label-json-path', type=str, default="person_keypoints.json")
    # 原项目提供的验证集person检测信息，如果要使用GT信息，直接将该参数置为None
    parser.add_argument('--person-det', type=str, default="./COCO_val2017_detections_AP_H_56_person.json")

    args = parser.parse_args()

    main(args)
