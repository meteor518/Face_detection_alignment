# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from collections import Iterable

import cv2
import mxnet as mx
import numpy as np
import tensorflow as tf

from ..face_detection import MtcnnDetector


# import sys
# sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src', 'common'))


def do_flip(data):
    """
    flip images horizontally
    :param data: image numpy data
    :return:
    """
    for idx in range(data.shape[0]):
        data[idx, :, :] = np.fliplr(data[idx, :, :])


# def normalize(X):
#     """Scale input vectors individually to unit norm (vector length)"""
#     _norm = np.linalg.norm(X, axis=1, keepdims=True)
#     X = np.multiply(X, 1. / _norm)
#     X[np.isnan(X)] = 0
#     return X


class FaceModel:
    def __init__(self, ctx='gpu,0', model=None, image_size=(112, 112), flip=False,
                 det_minsize=20, det_threshold=None, det_factor=0.709, det_accurate_landmark=True,
                 load_shaking_head_model=False):
        """mtcnn+angular margin loss model
        :param ctx:string
                   cpu id or gpu id
        :param model:string
                   path for the models
        :param image_size:tuple or iterable
                   the size of clipped image
        :param flip:bool
                   whether do lr flip aug
        :param det_minsize:int
                   minimal face to detect
        :param det_threshold:float
                   detect threshold for 3 stages in mtcnn
        :param det_factor:float
                   scale factor for image pyramid in mtcnn
        :param det_accurate_landmark:float
                   use accurate landmark localization or not in mtcnn
        """
        assert len(image_size) == 2
        assert ctx is not None
        assert model is not None
        dev, dev_id = ctx.split(',')
        ctx_ = mx.gpu(int(dev_id)) if dev == 'gpu' else mx.cpu(int(dev_id))
        self.image_size = image_size
        self.flip = flip
        # _vec = model.split(',')
        # assert len(_vec) == 2
        # prefix = _vec[0]  # model path
        # epoch = int(_vec[1])
        # print('loading', prefix, epoch)
        # sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
        # all_layers = sym.get_internals()
        # sym = all_layers['fc1_output']  # use embedding layer
        # model = mx.mod.Module(symbol=sym, context=ctx_, label_names=None)
        # model.bind(data_shapes=[('data', (1, 3, image_size[0], image_size[1]))])  # BGR
        # model.set_params(arg_params, aux_params)  # load weights
        self.model = model
        mtcnn_path = os.path.join(os.path.dirname(__file__), '..', 'face_detection', 'mtcnn-model')
        det_threshold = [0.6, 0.7, 0.8] if det_threshold is None else det_threshold
        det_factor = det_factor if isinstance(det_factor, Iterable) else [det_factor]
        detectors = []
        for factor in det_factor:
            detector = MtcnnDetector(model_folder=mtcnn_path,
                                     minsize=det_minsize,
                                     threshold=det_threshold,
                                     factor=factor,
                                     accurate_landmark=det_accurate_landmark,
                                     ctx=ctx_)
            detectors.append(detector)
        self.detectors = detectors
        # whether load head-shaking detection model
        if load_shaking_head_model:
            model_name = 'shake_head_detection_tf'
            with tf.gfile.FastGFile('../../AntiSpoof/SDK/AU改/model/tf/%s.pb' % model_name, 'rb') as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
            with tf.Graph().as_default() as graph:
                tf.import_graph_def(graph_def, name="")
            self.hs_sess = tf.InteractiveSession(graph=graph)
            # 加载输入输出节点
            self.hs_inputs = graph.get_tensor_by_name('input_1:0')
            self.hs_preds = graph.get_tensor_by_name('dense_1/Softmax:0')

    def get_aligned_face_image(self, img):
        if isinstance(img, str):
            img = cv2.imread(img)  # img is a bgr image
        ret = None
        for detector in self.detectors:
            try:
                ret = detector.detect_face(img)
            except Exception:
                continue
            if ret:
                break
        if ret is None:
            return None
        bbox, points = ret
        num_faces = bbox.shape[0]
        if num_faces == 0:
            return None
        bindex = 0
        if num_faces > 0:
            det = bbox[:, 0:4]
            # img_size = np.asarray(img.shape)[0:2]
            if num_faces > 1:
                # 选取最终BBox时暂不考虑其与图像中心的距离
                bounding_box_size = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
                # img_center = img_size / 2
                # offsets = np.vstack(
                #     [(det[:, 0] + det[:, 2]) / 2 - img_center[1], (det[:, 1] + det[:, 3]) / 2 - img_center[0]])
                # offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
                # bindex = np.argmax(bounding_box_size - offset_dist_squared * 2)  # some extra weight on the centering
                bindex = np.argmax(bounding_box_size)
        _bbox = bbox[bindex, 0:4]
        _bbox = [int(i) if i > 0 else 0 for i in _bbox]  # 避免坐标为负问题
        _landmark = points[bindex, :].reshape((2, 5)).T

        # img = preprocess(img.copy()) if preprocess else img
        # img_warped = face_preprocess.preprocess(img, _bbox, _landmark, image_size=self.image_size)
        # img_warped = cv2.cvtColor(img_warped, cv2.COLOR_BGR2RGB)
        return _bbox, _landmark  # todo
