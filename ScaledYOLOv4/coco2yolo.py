from pycocotools.coco import COCO
import os
import shutil

phase = 'train'  # val
dataDir = '/nfs_shared_/mscoco/2017'
prefix = 'instances'
dataType = f'{phase}2017'
annFile = '%s/annotations/%s_%s.json' % (dataDir, prefix, dataType)

img_root = f'/nfs_shared_/mscoco/2017/{phase}2017'
dst_root = f'/workspace/dataset/coco2017/{phase}/'
print(annFile)


coco_class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
                    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
                    'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
                    'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
                    'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
                    'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog',
                    'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
                    'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
                    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

coco = COCO(annFile)
coco_category = {v['id']: v['name'] for k, v in coco.cats.items()}
print(coco_category)
coco_names = [v['name'] for k, v in coco.cats.items()]
print(coco_names)
print(len(coco_names))

for n, (iid, img) in enumerate(coco.imgs.items()):
    aids = coco.getAnnIds(imgIds=[iid])
    name = img['file_name']
    w, h = img['width'], img['height']

    dst_label = os.path.join(dst_root, f"{name.split('.')[0]}.txt")
    dst_img = os.path.join(dst_root, name)

    src_img = os.path.join(img_root, name)
    os.symlink(src_img, dst_img)

    dst_label_file = open(dst_label, 'w')

    for aid in aids:
        ann = coco.anns[aid]
        cid = ann['category_id']
        cat = coco.cats[cid]['name']
        label = coco_class_names.index(cat)

        left, top, width, height = ann['bbox']  # x,y,w,h
        yolo_bbox = list(map(str, [(left + width / 2) / w, (top + height / 2) / h, width / w, height / h]))
        yolo_format = f'{label} {" ".join(yolo_bbox)}\n'
        dst_label_file.write(yolo_format)
    dst_label_file.close()

    print(f'[{n}/{len(coco.imgs.keys())}]', src_img, os.path.exists(src_img), dst_img, dst_label)
