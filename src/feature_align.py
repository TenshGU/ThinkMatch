import torch
from torch import Tensor


def feature_align(raw_feature: Tensor, P: Tensor, ns_t: Tensor, ori_size: tuple, device=None) -> Tensor:
    r"""
    Perform feature align on the image feature map.

    Feature align performs bi-linear interpolation on the image feature map. This operation is inspired by "ROIAlign"
    in `Mask R-CNN <https://arxiv.org/abs/1703.06870>`_.

    :param raw_feature: :math:`(b\times c \times w \times h)` raw feature map. :math:`b`: batch size, :math:`c`: number
     of feature channels, :math:`w`: feature map width, :math:`h`: feature map height
    :param P: :math:`(b\times n \times 2)` point set containing point coordinates. The coordinates are at the scale of
     the original image size. :math:`n`: number of points
    :param ns_t: :math:`(b)` number of exact points. We support batched instances with different number of nodes, and
     ``ns_t`` is required to specify the exact number of nodes of each instance in the batch.
    :param ori_size: size of the original image. Since the point coordinates are in the scale of the original image
     size, this parameter is required.
    :param device: output device. If not specified, it will be the same as the input
    :return: :math:`(b\times c \times n)` extracted feature vectors
    """

    #  U_src = feature_align(src_node, P_src, ns_src, self.rescale)

    if device is None:
        device = raw_feature.device

    batch_num = raw_feature.shape[0]  # batch size
    channel_num = raw_feature.shape[1]  # feature channel
    n_max = P.shape[1]  # number of points

    ori_size = torch.tensor(ori_size, dtype=torch.float32, device=device)  # self.rescale
    F = torch.zeros(batch_num, channel_num, n_max, dtype=torch.float32, device=device)
    for idx, feature in enumerate(raw_feature):
        n = ns_t[idx]
        feat_size = torch.as_tensor(feature.shape[1:3], dtype=torch.float32, device=device)  # w * h
        _P = P[idx, 0:n]  # batch_size(idx) * point_number * 2(w*h) -> 这里取idx批次的所有点（每个点是有w*h的坐标）
        interp_2d(feature, _P, ori_size, feat_size, out=F[idx, :, 0:n])  # idx批次，所有channel，所有n个点
    return F


def interp_2d(z: Tensor, P: Tensor, ori_size: Tensor, feat_size: Tensor, out=None, device=None) -> Tensor:
    r"""
    Interpolate in 2d grid space. z can be 3-dimensional where the first dimension is feature dimension.

    :param z: :math:`(c\times w\times h)` feature map. :math:`c`: number of feature channels, :math:`w`: feature map
     width, :math:`h`: feature map height
    :param P: :math:`(n\times 2)` point set containing point coordinates. The coordinates are at the scale of
     the original image size. :math:`n`: number of points
    :param ori_size: :math:`(2)` size of the original image
    :param feat_size: :math:`(2)` size of the feature map
    :param out: optional output tensor
    :param device: output device. If not specified, it will be the same as the input
    :return: :math:`(c \times n)` extracted feature vectors
    """

    # p - step / 2：这部分是对原始坐标 `p` 进行偏移。偏移 `step / 2` 是为了确保坐标点能够正确地对齐到特征图的中心。
    # 这是因为特征图通常是通过卷积或池化等操作从原始图像中生成的，这些操作可能会导致空间位置的轻微偏移。
    # (p - step / 2) / ori_size：这部分是将偏移后的坐标从原始图像空间缩放到 [0, 1] 范围内。`ori_size` 是原始图像的大小，所以除以 `ori_size` 可以将坐标归一化。
    # (p - step / 2) / ori_size * feat_size：最后，将归一化后的坐标乘以 `feat_size`，将其从 [0, 1] 范围缩放到特征图空间的大小。这样，你就得到了在特征图空间中对应的坐标。

    if device is None:
        device = z.device

    step = ori_size / feat_size  # self.rescale / w * h
    if out is None:
        out = torch.zeros(z.shape[0], P.shape[0], dtype=torch.float32, device=device)  # channel * point_num
    for i, p in enumerate(P):  # 每个点
        p = (p - step / 2) / ori_size * feat_size
        out[:, i] = bilinear_interpolate(z, p[0], p[1])

    return out


def bilinear_interpolate(im: Tensor, x: Tensor, y: Tensor, device=None):
    r"""
    Bi-linear interpolate 3d feature map to 2d coordinate (x, y).
    The coordinates are at the same scale of :math:`w\times h`.

    :param im: :math:`(c\times w\times h)` feature map
    :param x: :math:`(1)` x coordinate
    :param y: :math:`(1)` y coordinate
    :param device: output device. If not specified, it will be the same as the input
    :return: :math:`(c)` interpolated feature vector
    """
    if device is None:
        device = im.device
    x = x.to(torch.float32).to(device)
    y = y.to(torch.float32).to(device)

    x0 = torch.floor(x)
    x1 = x0 + 1
    y0 = torch.floor(y)
    y1 = y0 + 1

    # 用w和h构建x和y
    x0 = torch.clamp(x0, 0, im.shape[2] - 1)
    x1 = torch.clamp(x1, 0, im.shape[2] - 1)
    y0 = torch.clamp(y0, 0, im.shape[1] - 1)
    y1 = torch.clamp(y1, 0, im.shape[1] - 1)

    x0 = x0.to(torch.int32).to(device)
    x1 = x1.to(torch.int32).to(device)
    y0 = y0.to(torch.int32).to(device)
    y1 = y1.to(torch.int32).to(device)

    Ia = im[:, y0, x0]
    Ib = im[:, y1, x0]
    Ic = im[:, y0, x1]
    Id = im[:, y1, x1]

    # to perform nearest neighbor interpolation if out of bounds
    if x0 == x1:
        if x0 == 0:
            x0 -= 1
        else:
            x1 += 1
    if y0 == y1:
        if y0 == 0:
            y0 -= 1
        else:
            y1 += 1

    x0 = x0.to(torch.float32).to(device)
    x1 = x1.to(torch.float32).to(device)
    y0 = y0.to(torch.float32).to(device)
    y1 = y1.to(torch.float32).to(device)

    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)

    out = Ia * wa + Ib * wb + Ic * wc + Id * wd
    return out
