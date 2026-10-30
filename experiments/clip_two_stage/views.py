"""Hai view cố định: preprocess gốc và toàn khung pad đen trước preprocess."""
import numpy as np
from PIL import Image


def full_frame(image):
    """Giữ đủ pixel/aspect ratio của ảnh; không kéo giãn, không crop nội dung."""
    side=max(image.size)
    canvas=Image.new('RGB',(side,side),(0,0,0))
    canvas.paste(image,((side-image.width)//2,(side-image.height)//2))
    return canvas


def classify_views(original,whole,info_labels,labels):
    from experiments.clip_two_stage.run import decide
    if len(original)!=2 or len(whole)!=2:raise ValueError('Cần score của cả hai tầng')
    for a,b in zip(original,whole):
        if a.shape!=b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError('Score hai view không khớp')
    averaged=[(a+b)/2 for a,b in zip(original,whole)]
    return {'full_frame_two_stage':decide(*whole,info_labels,labels)[0],
            'mean_view_two_stage':decide(*averaged,info_labels,labels)[0]}
