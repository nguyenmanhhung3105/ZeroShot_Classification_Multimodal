"""Các phép gộp xác suất cố định dùng trong diagnostic một tầng."""
import numpy as np

from .diagnostic import mix, probability


def candidates(clip, stata, calip):
    """Các ứng viên cố định, không nhận nhãn; dùng được với 2 hoặc nhiều lớp.

    StatA rất sắc ở pilot nên thử giảm trọng số hoặc làm mềm posterior.
    Không có cổng binary, phân cấp hay bước chọn lớp trung gian.
    """
    branches = {k: probability(v) for k, v in
                [('clip', clip), ('stata', stata), ('calip', calip)]}
    if len({v.shape for v in branches.values()}) != 1:
        raise ValueError('Các nhánh phải cùng shape và thứ tự lớp')
    result = dict(branches)
    result['fusion_clip_calip'] = mix(branches, {'clip': .5, 'calip': .5})
    result['fusion_three'] = mix(branches, {'clip': 1/3, 'stata': 1/3, 'calip': 1/3})
    for weight in [.1, .25]:
        result[f'clip_stata_w{weight:g}'] = mix(branches, {'clip': 1-weight, 'stata': weight})
        result[f'three_stata_w{weight:g}'] = mix(
            branches, {'clip': (1-weight)/2, 'calip': (1-weight)/2, 'stata': weight})
    for temperature in [10., 100.]:
        softened = {**branches, 'stata': probability(branches['stata'], temperature)}
        result[f'three_stata_t{temperature:g}'] = mix(
            softened, {'clip': 1/3, 'stata': 1/3, 'calip': 1/3})
    return result

