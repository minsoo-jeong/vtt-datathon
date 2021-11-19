from pycocotools.coco import COCO
import os
import shutil
import cv2

img_root = '/workspace/vtt-data/train/datas'
label_root = '/workspace/vtt-data/train/labels'
dst_root = '/workspace/dataset/vtt-coco/train'

for n, img in enumerate(os.listdir(img_root)):
    basename, _ = os.path.splitext(img)
    label = os.path.join(label_root, f'{basename}.txt')
    w, h = 1024, 768
    valid_anns = []
    anns = open(label, 'r').readlines()
    for ann in anns:
        cat, x1, y1, x2, y2 = map(int, ann.strip().split())
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(x2, w)
        y2 = min(y2, h)
        if cat != 0:
            bw, bh = x2 - x1, y2 - y1
            yolo_bbox = list(map(str, [(x1 + bw / 2) / w, (y1 + bh / 2) / h, bw / w, bh / h]))

            yolo_format = f'{cat - 1} {" ".join(yolo_bbox)}\n'
            valid_anns.append(yolo_format)

    if len(valid_anns):
        dst_label = os.path.join(dst_root, f'{basename}.txt')
        src_img = os.path.join(img_root, img)
        dst_img = os.path.join(dst_root, img)
        print(src_img, dst_img, dst_label, valid_anns)

        # os.symlink(src_img, dst_img)
        dst_label_file = open(dst_label, 'w')
        for ann in valid_anns:
            dst_label_file.write(ann)
        dst_label_file.close()
