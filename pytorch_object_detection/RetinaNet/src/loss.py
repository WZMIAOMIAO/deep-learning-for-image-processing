from torch import nn, Tensor
import torch


class Loss(nn.Module):
    """
        Implements the loss as the sum of the followings:
        1. Confidence Loss: All labels, with hard negative mining
        2. Localization Loss: Only on positive labels
        Suppose input dboxes has the shape 76725x4
    """
    def __init__(self, dboxes):
        super(Loss, self).__init__()
        self.scale_xy = 1.0 / dboxes.scale_xy
        self.scale_wh = 1.0 / dboxes.scale_wh

        self.location_loss = nn.SmoothL1Loss(reduction='none')
        # self.location_loss = nn.SmoothL1Loss(reduce=False)
        self.dboxes = nn.Parameter(dboxes(order="xywh").transpose(0, 1).unsqueeze(dim=0),
                                   requires_grad=False)

        # Two factor are from following links
        # http://jany.st/post/2017-11-05-single-shot-detector-ssd-from-scratch-in-tensorflow.html
        self.confidence_loss = nn.CrossEntropyLoss(reduction='none')
        # self.confidence_loss = nn.CrossEntropyLoss(reduce=False)

    def _location_vec(self, loc):
        # type: (Tensor)
        """
        Generate Location Vectors
        计算ground truth相对anchors的回归参数
        :param loc:
        :return:
        """
        gxy = self.scale_xy * (loc[:, :2, :] - self.dboxes[:, :2, :]) / self.dboxes[:, 2:, :]
        gwh = self.scale_wh * (loc[:, 2:, :] / self.dboxes[:, 2:, :]).log()
        return torch.cat((gxy, gwh), dim=1).contiguous()

    def forward(self, ploc, plabel, gloc, glabel):
        # type: (Tensor, Tensor, Tensor, Tensor)
        """
            ploc, plabel: Nx4x76725, Nxlabel_numx76725
                predicted location and labels

            gloc, glabel: Nx4x76725, Nx76725
                ground truth location and labels
        """
        # 获取正样本的mask  Tensor: [N, 76725]
        mask = glabel > 0
        # mask1 = torch.nonzero(glabel)
        # 计算一个batch中的每张图片的正样本个数 Tensor: [N]
        pos_num = mask.sum(dim=1)

        # 计算gt的location回归参数 Tensor: [N, 4, 76725]
        vec_gd = self._location_vec(gloc)

        # sum on four coordinates, and mask
        # 计算定位损失(只有正样本)
        loc_loss = self.location_loss(ploc, vec_gd).sum(dim=1)  # Tensor: [N, 76725]
        loc_loss = (mask.float() * loc_loss).sum(dim=1)  # Tenosr: [N]

        # hard negative mining Tenosr: [N, 76725]
        con = self.confidence_loss(plabel, glabel)

        # positive mask will never selected
        # 获取负样本
        con_neg = con.clone()
        con_neg[mask] = torch.tensor(0.0)
        # 按照confidence_loss降序排列 con_idx(Tensor: [N, 76725])
        _, con_idx = con_neg.sort(dim=1, descending=True)
        _, con_rank = con_idx.sort(dim=1)  # 这个步骤比较巧妙

        # number of negative three times positive
        # 用于损失计算的负样本数是正样本的3倍（在原论文Hard negative mining部分），
        # 但不能超过总样本数
        neg_num = torch.clamp(3 * pos_num, max=mask.size(1)).unsqueeze(-1)
        neg_mask = con_rank < neg_num  # Tensor [N, 76725]

        # confidence最终loss使用选取的正样本loss+选取的负样本loss
        con_loss = (con * (mask.float() + neg_mask.float())).sum(dim=1)  # Tensor [N]

        # avoid no object detected
        # 避免出现图像中没有GTBOX的情况
        total_loss = loc_loss + con_loss
        num_mask = (pos_num > 0).float()  # 统计一个batch中的每张图像中是否存在GTBOX
        pos_num = pos_num.float().clamp(min=1e-6)  # 防止出现分母为零的情况
        ret = (total_loss * num_mask / pos_num).mean(dim=0)  # 只计算存在GTBOX的图像损失
        return ret


class FocalLoss(nn.Module):
    """
        Implements the loss as the sum of the followings:
        1. Confidence Loss: All labels, with hard negative mining
        2. Localization Loss: Only on positive labels
        Suppose input dboxes has the shape 76725x4
    """
    def __init__(self, dboxes):
        super(FocalLoss, self).__init__()
        self.scale_xy = 1.0 / dboxes.scale_xy
        self.scale_wh = 1.0 / dboxes.scale_wh

        self.location_loss = nn.SmoothL1Loss(reduction='none')
        # self.location_loss = nn.SmoothL1Loss(reduce=False)
        self.dboxes = nn.Parameter(dboxes(order="xywh").transpose(0, 1).unsqueeze(dim=0),
                                   requires_grad=False)

        # Two factor are from following links
        # http://jany.st/post/2017-11-05-single-shot-detector-ssd-from-scratch-in-tensorflow.html
        self.confidence_loss = nn.CrossEntropyLoss(reduction='none')
        # self.confidence_loss = nn.CrossEntropyLoss(reduce=False)

    def _location_vec(self, loc):
        # type: (Tensor)
        """
        Generate Location Vectors
        计算ground truth相对anchors的回归参数
        :param loc:
        :return:
        """
        gxy = self.scale_xy * (loc[:, :2, :] - self.dboxes[:, :2, :]) / self.dboxes[:, 2:, :]
        gwh = self.scale_wh * (loc[:, 2:, :] / self.dboxes[:, 2:, :]).log()
        return torch.cat((gxy, gwh), dim=1).contiguous()

    def forward(self, ploc, plabel, gloc, glabel):
        # type: (Tensor, Tensor, Tensor, Tensor)
        """
            ploc, plabel: Nx4x76725, Nxlabel_numx76725
                predicted location and labels

            gloc, glabel: Nx4x76725, Nx76725
                ground truth location and labels
        """
        # 获取正样本的mask  Tensor: [N, 76725]
        mask = glabel > 0
        # mask1 = torch.nonzero(glabel)
        # 计算一个batch中的每张图片的正样本个数 Tensor: [N]
        pos_num = mask.sum(dim=1)

        # 计算gt的location回归参数 Tensor: [N, 4, 76725]
        vec_gd = self._location_vec(gloc)

        # sum on four coordinates, and mask
        # 计算定位损失(只有正样本)
        loc_loss = self.location_loss(ploc, vec_gd).sum(dim=1)  # Tensor: [N, 76725]
        loc_loss = (mask.float() * loc_loss).sum(dim=1)  # Tenosr: [N]

        # hard negative mining Tenosr: [N, 76725]
        con = self.confidence_loss(plabel, glabel)

        # positive mask will never selected
        # 获取负样本
        con_neg = con.clone()
        con_neg[mask] = torch.tensor(0.0)
        # 按照confidence_loss降序排列 con_idx(Tensor: [N, 76725])
        _, con_idx = con_neg.sort(dim=1, descending=True)
        _, con_rank = con_idx.sort(dim=1)  # 这个步骤比较巧妙

        # number of negative three times positive
        # 用于损失计算的负样本数是正样本的3倍（在原论文Hard negative mining部分），
        # 但不能超过总样本数
        neg_num = torch.clamp(3 * pos_num, max=mask.size(1)).unsqueeze(-1)
        neg_mask = con_rank < neg_num  # Tensor [N, 76725]

        # confidence最终loss使用选取的正样本loss+选取的负样本loss
        con_loss = (con * (mask.float() + neg_mask.float())).sum(dim=1)  # Tensor [N]

        # avoid no object detected
        # 避免出现图像中没有GTBOX的情况
        total_loss = loc_loss + con_loss
        num_mask = (pos_num > 0).float()  # 统计一个batch中的每张图像中是否存在GTBOX
        pos_num = pos_num.float().clamp(min=1e-6)  # 防止出现分母为零的情况
        ret = (total_loss * num_mask / pos_num).mean(dim=0)  # 只计算存在GTBOX的图像损失
        return ret