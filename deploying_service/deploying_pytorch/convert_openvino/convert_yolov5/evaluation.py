from tqdm import tqdm
import torch
from openvino.runtime import Core
from utils import MyDataLoader, EvalCOCOMetric, non_max_suppression


def main():
    data_path = "/data/coco2017"
    ir_model_xml = "quant_ir_output/quantized_yolov5s.xml"
    img_size = (640, 640)  # h, w

    data_loader = MyDataLoader(data_path, "val", size=img_size)
    coco80_to_91 = data_loader.coco_id80_to_id91
    metrics = EvalCOCOMetric(coco=data_loader.coco, classes_mapping=coco80_to_91)

    # Load the network in Inference Engine
    ie = Core()
    model_ir = ie.read_model(model=ir_model_xml)
    compiled_model = ie.compile_model(model=model_ir, device_name="CPU")
    inputs_names = compiled_model.inputs
    outputs_names = compiled_model.outputs

    # inference
    request = compiled_model.create_infer_request()
    for i in tqdm(range(len(data_loader))):
        data = data_loader[i]
        ann, img, info = data
        ann = ann + (info,)

        request.infer(inputs={inputs_names[0]: img})
        result = request.get_output_tensor(outputs_names[0].index).data

        # post-process
        result = non_max_suppression(torch.Tensor(result), conf_thres=0.001, iou_thres=0.6, multi_label=True)[0]
        boxes = result[:, :4].numpy()
        scores = result[:, 4].numpy()
        cls = result[:, 5].numpy().astype(int)
        metrics.update(ann, [boxes, cls, scores])

    metrics.evaluate()


if __name__ == '__main__':
    main()
