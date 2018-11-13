import argparse
import glob
import os
from functools import partial
from multiprocessing import Pool

import cv2
import dlib
import numpy as np
from FaceDetection.python.face2vector import FaceModel, face_preprocess
from PIL import Image
from colorama import init, Fore, Style
from tqdm import tqdm

init(autoreset=True)


class BBoxWrapper(object):
    def __init__(self, *coor):
        assert len(coor) == 4
        self.coor = coor

    def left(self): return self.coor[0]

    def top(self): return self.coor[1]

    def right(self): return self.coor[2]

    def bottom(self): return self.coor[3]

    def height(self): return self.coor[3] - self.coor[1] + 1

    def width(self): return self.coor[2] - self.coor[0] + 1


def detect(fpath, save_dir, resize_factors=None):
    if resize_factors is None:
        resize_factors = []
    save_path = os.path.join(save_dir, fpath[-35:])
    if not os.path.isfile(save_path):
        try:
            with Image.open(save_path) as image:
                image.verify()
            with Image.open(save_path) as image:
                image.load()
            return Fore.CYAN + Style.DIM + fpath
        except (IOError, AttributeError):
            pass
    message = Fore.RED + fpath
    image = Image.open(fpath)

    if 1 not in resize_factors:
        resize_factors.append(1)

    for factor in resize_factors:
        if factor != 1:
            _save_path = '{}#{}x.jpg'.format(save_path.rstrip('.jpg'), factor)
            _image = image.resize((int(image.width * factor), int(image.height * factor)))
        else:
            _save_path = save_path
            _image = image
        image_gray = _image.convert('L')
        boxes = detector.detect(image_gray)
        if boxes:
            box = max(boxes, key=lambda b: (b.right - b.left) * (b.bottom - b.top))
            face = _image.crop((box.left, box.top, box.right, box.bottom))
            try:
                if not os.path.isdir(os.path.dirname(_save_path)):
                    os.makedirs(os.path.dirname(_save_path), exist_ok=True)
                face.save(_save_path)
                if factor == 1:
                    message = Fore.CYAN + fpath
            except IOError:
                pass
    return message


def detect_and_align_by_dlib(fpath, save_dir, resize_factors=None):
    if resize_factors is None:
        resize_factors = []
    s = fpath.split('/')
    save_path = os.path.join(save_dir, s[-2], s[-1])
    if not os.path.isfile(save_path):
        try:
            with Image.open(save_path) as image:
                image.verify()
            with Image.open(save_path) as image:
                image.load()
            return Fore.CYAN + Style.DIM + fpath
        except (IOError, AttributeError):
            pass
    message = Fore.RED + fpath
    image = cv2.imread(fpath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if 1 not in resize_factors:
        resize_factors.append(1)

    for factor in resize_factors:
        if factor != 1:
            _save_path = '{}#{}x.jpg'.format(save_path.rstrip('.jpg'), factor)
            _image = cv2.resize(image, None, fx=factor, fy=factor)
        else:
            _save_path = save_path
            _image = image
        dets = dlib_detector(_image, 1)
        if dets:
            try:
                det = max(dets, key=lambda d: d.area())
                face = sp(_image, det)
                if not os.path.isdir(os.path.dirname(_save_path)):
                    os.makedirs(os.path.dirname(_save_path), exist_ok=True)
                dlib.save_face_chip(_image, face, _save_path[:-4], size=224, padding=0.25)
                if factor == 1:
                    # add mask square to original images
                    _save_path = '{}#maskx.jpg'.format(save_path.rstrip('.jpg'))
                    _image = add_occlusion(_image, det)
                    dlib.save_face_chip(_image, face, _save_path[:-4], size=224, padding=0.25)
                if factor == 1:
                    message = Fore.CYAN + fpath
            except IOError:
                pass
    return message


def detect_and_align_by_mtcnn(fpath, save_dir, resize_factors=None, **kwargs):
    if resize_factors is None:
        resize_factors = []
    s = fpath.split('/')
    save_path = os.path.join(save_dir, s[-2], s[-1])
    if not os.path.isfile(save_path):
        try:
            with Image.open(save_path) as image:
                image.verify()
            with Image.open(save_path) as image:
                image.load()
            return Fore.CYAN + Style.DIM + fpath
        except (IOError, AttributeError):
            pass
    message = Fore.RED + fpath
    image = cv2.imread(fpath)  # BGR Format

    if 1 not in resize_factors:
        resize_factors.append(1)

    def save_aligned_face(img, bbox, landmark, save_path):
        result = face_preprocess.preprocess(img, bbox, landmark, image_size=(112, 112), **kwargs)
        # result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        cv2.imwrite(save_path, result)

    for factor in resize_factors:
        if factor != 1:
            _save_path = '{}#{}x.jpg'.format(save_path.rstrip('.jpg'), factor)
            _image = cv2.resize(image, None, fx=factor, fy=factor)
        else:
            _save_path = save_path
            _image = image
        if os.path.exists(_save_path):
            if factor == 1:
                message = Fore.CYAN + fpath
            continue
        dets = face_model.get_aligned_face_image(_image)
        if dets:
            try:
                bbox, landmark = dets
                if not os.path.isdir(os.path.dirname(_save_path)):
                    os.makedirs(os.path.dirname(_save_path), exist_ok=True)
                save_aligned_face(_image, bbox, landmark, _save_path)
                if factor != 1:
                    # add mask square to original images
                    _save_path = '{}#maskx.jpg'.format(save_path.rstrip('.jpg'))
                    _image = add_occlusion(_image, BBoxWrapper(*bbox))
                    save_aligned_face(_image, bbox, landmark, _save_path)
                if factor == 1:
                    message = Fore.CYAN + fpath
            except IOError:
                pass
    return message


def add_occlusion(image, bbox, square_size_ratio=0.2):
    h, w = bbox.height(), bbox.width()
    square_size = int(w * square_size_ratio)
    square_max_left = w - square_size
    square_max_top = h - square_size
    left = np.random.randint(square_max_left) + bbox.left()
    top = np.random.randint(square_max_top) + bbox.top()
    right, bottom = left + square_size, top + square_size
    image[top:bottom, left:right, :] = 0
    return image


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-dir', '-i', required=True)
    parser.add_argument('--output-dir', '-o', required=True)
    parser.add_argument('--face-model', '-model', type=str, default='mtcnn', choices=['mtcnn', 'dlib'])
    parser.add_argument('--min-face-size', '-m', type=int, default=50)
    parser.add_argument('--n-jobs', '-j', type=int, default=4)
    parser.add_argument('--align', '-a', action='store_true', default=True)
    parser.add_argument('--align-template', '-at', type=str, default='normal', choices=['normal', 'insightface'])
    parser.add_argument('--align-padding', '-ap', type=float, default=0.15)
    parser.add_argument('--resize-factors', '-r', type=str, default='1')
    parser.add_argument('--gpu-device', '-gpu', type=int)
    parser.add_argument('--verbose', '-v', action='store_true', help='if true, it will print skipping logs')
    args = parser.parse_args()

    image_dir = os.path.abspath(os.path.expanduser(args.image_dir))
    output_dir = os.path.abspath(os.path.expanduser(args.output_dir))

    if not os.path.isdir(image_dir):
        raise IOError('image_dir "{}" not exist!'.format(image_dir))

    np.random.seed(0)

    tqdm.write('Arguments:')
    for k, v in vars(args).items():
        tqdm.write(' - {}: {}'.format(k, v))

    os.makedirs(output_dir, exist_ok=True)
    factors = [float(i) for i in args.resize_factors.split(',')]

    if args.face_model == 'mtcnn':
        '''mtcnn face detector'''
        ctx = 'gpu,{}'.format(args.gpu_device) if args.gpu_device else 'cpu,0'
        face_model = FaceModel(ctx=ctx, model='.', det_threshold=[0.6, 0.7, 0.8], det_minsize=50,
                               det_factor=[0.709, 0.850], det_accurate_landmark=True)
    elif args.face_model == 'dlib':
        '''dlib face detector'''
        dlib_detector = dlib.get_frontal_face_detector()
        sp = dlib.shape_predictor(os.path.join(os.path.dirname(__file__), 'shape_predictor_68_face_landmarks.dat'))
    else:
        raise AttributeError('Only support mtcnn and dlib face_model')

    if args.align:
        tqdm.write(Fore.CYAN + 'Do alignment after detecting')
        detect_worker = partial(detect_and_align_by_mtcnn, save_dir=output_dir, resize_factors=factors,
                                template=args.align_template, padding=args.align_padding)
    else:
        from pyseeta import Detector

        detector = Detector()
        detector.set_min_face_size(args.min_face_size)
        detect_worker = partial(detect, save_dir=output_dir, resize_factors=factors)
    image_paths = glob.glob(os.path.join(args.image_dir, '*', '*.jpg'))
    n_missing = 0
    with Pool(processes=args.n_jobs) as p:
        with tqdm(total=len(image_paths), desc='Detecting') as pbar:
            for msg in p.imap(detect_worker, image_paths):
                pbar.update()
                if not args.verbose and Fore.CYAN in msg:
                    pass
                else:
                    pbar.write(msg)
                    n_missing += 1
    tqdm.write(Fore.CYAN + f'{n_missing} images fail to be detected')  # 183

    if not args.align:
        detector.release()
