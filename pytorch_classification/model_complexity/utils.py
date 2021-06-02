"""
these code refers to:
https://github.com/facebookresearch/pycls/blob/master/pycls/models/blocks.py
"""


def conv2d_cx(cx, in_c, out_c, k, *, stride=1, groups=1, bias=False, trainable=True):
    """Accumulates complexity of conv2d into cx = (h, w, flops, params, acts)."""
    assert k % 2 == 1, "Only odd size kernels supported to avoid padding issues."
    h, w, c = cx["h"], cx["w"], cx["c"]
    assert c == in_c
    h, w = (h - 1) // stride + 1, (w - 1) // stride + 1
    cx["h"] = h
    cx["w"] = w
    cx["c"] = out_c
    cx["flops"] += k * k * in_c * out_c * h * w // groups + (out_c if bias else 0)
    cx["params"] += k * k * in_c * out_c // groups + (out_c if bias else 0)
    cx["acts"] += out_c * h * w
    if trainable is False:
        cx["freeze"] += k * k * in_c * out_c // groups + (out_c if bias else 0)
    return cx


def pool2d_cx(cx, in_c, k, *, stride=1):
    """Accumulates complexity of pool2d into cx = (h, w, flops, params, acts)."""
    assert k % 2 == 1, "Only odd size kernels supported to avoid padding issues."
    h, w, c = cx["h"], cx["w"], cx["c"]
    assert c == in_c
    h, w = (h - 1) // stride + 1, (w - 1) // stride + 1
    cx["h"] = h
    cx["w"] = w
    cx["acts"] += in_c * h * w
    return cx


def norm2d_cx(cx, in_c, trainable=True):
    """Accumulates complexity of norm2d into cx = (h, w, flops, params, acts)."""
    c, params = cx["c"], cx["params"]
    assert c == in_c
    cx["params"] += 4 * c
    cx["freeze"] += 2 * c  # moving_mean, variance
    if trainable is False:
        cx["freeze"] += 2 * c  # beta, gamma
    return cx


def gap2d_cx(cx):
    """Accumulates complexity of gap2d into cx = (h, w, flops, params, acts)."""
    cx["h"] = 1
    cx["w"] = 1
    return cx


def linear_cx(cx, in_units, out_units, *, bias=False, trainable=True):
    """Accumulates complexity of linear into cx = (h, w, flops, params, acts)."""
    c = cx["c"]
    assert c == in_units
    cx["c"] = out_units
    cx["flops"] += in_units * out_units + (out_units if bias else 0)
    cx["params"] += in_units * out_units + (out_units if bias else 0)
    cx["acts"] += out_units
    if trainable is False:
        cx["freeze"] += in_units * out_units + (out_units if bias else 0)
    return cx
