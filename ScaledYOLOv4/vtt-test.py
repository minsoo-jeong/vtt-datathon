import sys
import argparse
from detect_custom import detect


def test(opt):
    detect(opt)

    # test code


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_folder', type=str, required=False, dest='source',default='/workspace/sample/images')
    parser.add_argument('--save_path', type=str, required=False,default='/workspace/sample/output')
    parser.add_argument('--weights', nargs='+', type=str, default='/workspace/ScaledYOLOv4/best.pt', help='model.pt path(s)')
    parser.add_argument('--img-size', type=int, default=896, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.2, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')

    return parser.parse_args(argv)


if __name__ == '__main__':
    opt = parse_arguments(sys.argv[1:])
    # opt.source = 'ScaledYOLOv4/inference/images'
    # opt.save_path='./save'

    test(opt)
