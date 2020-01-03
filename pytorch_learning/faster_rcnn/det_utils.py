import torch
import math


class BoxCoder(object):

    def __init__(self, weights, bbox_xform_clip=math.log(1000. / 16)):

        self.weights = weights
        self.bbox_xform_clip = bbox_xform_clip

    def decode(self, rel_codes, boxes):
        assert isinstance(boxes, (list, tuple))
        if isinstance(rel_codes, (list, tuple)):
            rel_codes = torch.cat(rel_codes, dim=0)
        assert isinstance(rel_codes, torch.Tensor)
        boxes_per_image = [len(b) for b in boxes]
        concat_boxes = torch.cat(boxes, dim=0)
        pred_boxes = self.decode_single(
            rel_codes.reshape(sum(boxes_per_image), -1), concat_boxes
        )
        return pred_boxes.reshape(sum(boxes_per_image), -1, 4)

    def decode_single(self, rel_codes, boxes):
        boxes = boxes.to(rel_codes.dtype)

        widths = boxes[:, 2] - boxes[: 0]
        heights = boxes[:, 3] - boxes[:, 1]
        ctr_x = boxes[:, 0] + 0.5 * widths
        ctr_y = boxes[:, 1] + 0.5 * heights

        wx, wy, ww, wh = self.weights
        dx = rel_codes[:, 0::4] / wx
        dy = rel_codes[:, 1::4] / wy
        dw = rel_codes[:, 2::4] / ww
        dh = rel_codes[:, 3::4] / wh

        # limit max value, prevent sending too large values into torch.exp()
        dw = torch.clamp(dw, max=self.bbox_xform_clip)
        dh = torch.clamp(dh, max=self.bbox_xform_clip)

        pre_ctr_x = dx * widths[:, None] + ctr_x[:, None]
        pre_ctr_y = dy * heights[:, None] + ctr_y[:, None]
        pre_w = torch.exp(dw) * widths[:, None]
        pre_h = torch.exp(dh) * heights[:, None]

        pred_boxes = torch.zeros_like(rel_codes)
        # x1
        pred_boxes[:, 0::4] = pre_ctr_x - 0.5 * pre_w
        # y1
        pred_boxes[:, 1::4] = pre_ctr_y - 0.5 * pre_h
        # x2
        pred_boxes[:, 2::4] = pre_ctr_x + 0.5 * pre_w
        # y2
        pred_boxes[:, 3::4] = pre_ctr_y + 0.5 * pre_h

        return pred_boxes