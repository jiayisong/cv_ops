import torch
import numpy as np
from . import iou3d_nms_cuda


def check_numpy_to_torch(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float(), True
    return x, False


def boxes_iou_3d_gpu(boxes_a, boxes_b, aligned=False):
    """
    Args:
        boxes_a: (B, N, 7) [x, y, z, dx, dy, dz, heading]
        boxes_b: (B, M, 7) [x, y, z, dx, dy, dz, heading]

    Returns:
        ans_iou: (1, N,) or (B, N, M)
    """
    assert boxes_a.shape[0] == boxes_b.shape[0]

    boxes_a = boxes_a.contiguous()
    boxes_b = boxes_b.contiguous()
    if aligned:
        assert boxes_a.shape[1] == boxes_b.shape[1] == 7
        assert len(boxes_a.shape) == 2
        overlaps_bev = boxes_a.new_zeros((boxes_a.shape[0],))
        # overlaps_bev = torch.cuda.FloatTensor(torch.Size((boxes_a.shape[0], boxes_b.shape[0]))).zero_()  # (N, M)
        iou3d_nms_cuda.boxes_iou_3d_aligned_gpu(boxes_a, boxes_b, overlaps_bev, boxes_a.device.index)
    else:
        assert boxes_a.shape[2] == boxes_b.shape[2] == 7
        assert len(boxes_a.shape) == 3
        overlaps_bev = boxes_a.new_zeros((boxes_a.shape[0], boxes_a.shape[1], boxes_b.shape[1]))
        # overlaps_bev = torch.cuda.FloatTensor(torch.Size((boxes_a.shape[0], boxes_b.shape[0]))).zero_()  # (N, M)
        iou3d_nms_cuda.boxes_iou_3d_gpu(boxes_a, boxes_b, overlaps_bev, boxes_a.device.index)

    return overlaps_bev


def boxes_iou_bev_gpu(boxes_a, boxes_b, aligned=False):
    """
    Args:
        boxes_a: (N, 7) [x, y, z, dx, dy, dz, heading]
        boxes_b: (M, 7) [x, y, z, dx, dy, dz, heading]

    Returns:
        ans_iou: (N, M)
    """
    assert boxes_a.shape[1] == boxes_b.shape[1] == 7
    if aligned:
        raise NotImplementedError
    else:

        overlaps_bev = boxes_a.new_zeros((boxes_a.shape[0], boxes_b.shape[0]))
        # overlaps_bev = torch.cuda.FloatTensor(torch.Size((boxes_a.shape[0], boxes_b.shape[0]))).zero_()  # (N, M)
        iou3d_nms_cuda.boxes_iou_bev_gpu(boxes_a.contiguous(), boxes_b.contiguous(), overlaps_bev, boxes_a.device.index)

    return overlaps_bev


def nms_bev_gpu(boxes, scores, labels, thresh, class_agnostic=False, nms_rescale_factor=None, pre_maxsize=0,
                fol_maxsize=0):
    """
    :param boxes: (N, 7) [x, y, z, dx, dy, dz, heading]
    :param scores: (N)
    :param thresh:
    :return:
    """
    assert boxes.shape[1] == 7
    order = scores.sort(0, descending=True)[1]
    if pre_maxsize > 0:
        order = order[:pre_maxsize]
    boxes = boxes[order]
    labels = labels[order].contiguous()
    if nms_rescale_factor is not None:
        nms_rescale_factor = boxes.new_tensor(nms_rescale_factor)
        boxes[:, 3:6] *= nms_rescale_factor[labels.long()].unsqueeze(-1)
    if not class_agnostic:
        max_dim = boxes[:, 3].max() + boxes[:, 5].max()
        min_coordinate = boxes[:, 0].min()
        max_coordinate = boxes[:, 0].max()
        offsets = labels * (max_coordinate - min_coordinate + max_dim)
        boxes[:, 0] = boxes[:, 0] + offsets
    boxes = boxes.contiguous()
    keep = torch.zeros(boxes.size(0), dtype=torch.long)
    num_out = iou3d_nms_cuda.nms_bev_gpu(boxes, keep, thresh, boxes.device.index)
    if fol_maxsize > 0:
        num_out = min(num_out, fol_maxsize)
    return order[keep[:num_out].cuda()].contiguous()


def nms_3d_gpu(boxes, scores, labels, thresh, class_agnostic=False, nms_rescale_factor=None, pre_maxsize=0,
               fol_maxsize=0):
    """
    :param boxes: (N, 7) [x, y, z, dx, dy, dz, heading]
    :param scores: (N)
    :param labels: (N)
    :param thresh:
    :return:
    """
    assert boxes.shape[1] == 7
    order = scores.sort(0, descending=True)[1]
    if pre_maxsize > 0:
        order = order[:pre_maxsize]
    boxes = boxes[order]
    labels = labels[order].contiguous()
    if nms_rescale_factor is not None:
        nms_rescale_factor = boxes.new_tensor(nms_rescale_factor)
        boxes[:, 3:6] *= nms_rescale_factor[labels.long()].unsqueeze(-1)
    if not class_agnostic:
        min_coordinate = (boxes[:, 1] - boxes[:, 4] / 2).min()
        max_coordinate = (boxes[:, 1] + boxes[:, 4] / 2).max()
        offsets = labels * (max_coordinate - min_coordinate + 1)
        boxes[:, 1] = boxes[:, 1] + offsets
    keep = torch.zeros(boxes.size(0), dtype=torch.long)
    boxes = boxes.contiguous()
    num_out = iou3d_nms_cuda.nms_3d_gpu(boxes, keep, thresh, boxes.device.index)
    if fol_maxsize > 0:
        num_out = min(num_out, fol_maxsize)
    return order[keep[:num_out].cuda()].contiguous()


def nms_dist_gpu(boxes, scores, labels, thresh, class_agnostic=False, nms_rescale_factor=None, pre_maxsize=0,
                 fol_maxsize=0):
    """
    :param boxes: (N, 7) [x, y, z, dx, dy, dz, heading]
    :param scores: (N)
    :param labels: (N)
    :param thresh:
    :return:
    """
    assert boxes.shape[1] == 7
    order = scores.sort(0, descending=True)[1]
    if pre_maxsize > 0:
        order = order[:pre_maxsize]
    boxes = boxes[order]
    labels = labels[order].contiguous()
    if nms_rescale_factor is not None:
        nms_rescale_factor = boxes.new_tensor(nms_rescale_factor)
        boxes[:, :3] /= nms_rescale_factor[labels.long()].unsqueeze(-1)
    if not class_agnostic:
        min_coordinate = boxes[:, 0].min()
        max_coordinate = boxes[:, 0].max()
        offsets = labels * (max_coordinate - min_coordinate + thresh + 1)
        boxes[:, 0] = boxes[:, 0] + offsets
    keep = torch.zeros(boxes.size(0), dtype=torch.long)
    boxes = boxes.contiguous()
    num_out = iou3d_nms_cuda.nms_dist_gpu(boxes, keep, thresh, boxes.device.index)
    if fol_maxsize > 0:
        num_out = min(num_out, fol_maxsize)
    return order[keep[:num_out].cuda()].contiguous()
