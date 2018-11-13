import cv2
import numpy as np
from skimage import transform as trans


def read_image(img_path, **kwargs):
    mode = kwargs.get('mode', 'rgb')
    layout = kwargs.get('layout', 'HWC')
    if mode == 'gray':
        img = cv2.imread(img_path, cv2.CV_LOAD_IMAGE_GRAYSCALE)
    else:
        img = cv2.imread(img_path, cv2.CV_LOAD_IMAGE_COLOR)
        if mode == 'rgb':
            # print('to rgb')
            img = img[..., ::-1]
        if layout == 'CHW':
            img = np.transpose(img, (2, 0, 1))
    return img


def preprocess(img, bbox=None, landmark=None, **kwargs):
    """
    align face
    :param img:numpy array, bgr order of shape (1, 3, n, m)
    :param bbox:face bounding box
    :param landmark:face key points
    :param kwargs:
    :return:
    """
    template = kwargs.get('template', 'insightface')
    padding = kwargs.get('padding', 0)
    if isinstance(img, str):
        img = read_image(img, **kwargs)
    M = None
    image_size = kwargs.get('image_size', (112, 112))
    if landmark is not None:
        assert len(image_size) == 2
        if template == 'insightface':
            src = np.array([
                [30.2946, 51.6963],
                [65.5318, 51.5014],
                [48.0252, 71.7366],
                [33.5493, 92.3655],
                [62.7299, 92.2041]], dtype=np.float32)
            if image_size[1] == 112:
                src[:, 0] += 8.0
        else:
            mean_face_shape_x = np.array([0.224152, 0.75610125, 0.490127, 0.254149, 0.726104])
            mean_face_shape_y = np.array([0.2119465, 0.2119465, 0.628106, 0.780233, 0.780233])
            xs = (mean_face_shape_x + padding) / (2 * padding + 1) * image_size[1]
            ys = (mean_face_shape_y + padding) / (2 * padding + 1) * image_size[0]
            src = np.vstack([xs, ys]).T
        dst = landmark.astype(np.float32)

        tform = trans.SimilarityTransform()
        tform.estimate(dst, src)
        M = tform.params[0:2, :]
        # M = cv2.estimateRigidTransform( dst.reshape(1,5,2), src.reshape(1,5,2), False)

    if M is None:
        if bbox is None:  # use center crop
            det = np.zeros(4, dtype=np.int32)
            det[0] = int(img.shape[1] * 0.0625)
            det[1] = int(img.shape[0] * 0.0625)
            det[2] = img.shape[1] - det[0]
            det[3] = img.shape[0] - det[1]
        else:
            det = bbox
        margin = kwargs.get('margin', 44)
        bb = np.zeros(4, dtype=np.int32)
        bb[0] = np.maximum(det[0] - margin / 2, 0)
        bb[1] = np.maximum(det[1] - margin / 2, 0)
        bb[2] = np.minimum(det[2] + margin / 2, img.shape[1])
        bb[3] = np.minimum(det[3] + margin / 2, img.shape[0])
        ret = img[bb[1]:bb[3], bb[0]:bb[2], :]
        if len(image_size) > 0:
            ret = cv2.resize(ret, (image_size[1], image_size[0]))
        return ret
    else:  # do align using landmark
        assert len(image_size) == 2

        # src = src[0:3,:]
        # dst = dst[0:3,:]

        # print(src.shape, dst.shape)
        # print(src)
        # print(dst)
        # print(M)
        warped = cv2.warpAffine(img, M, (image_size[1], image_size[0]), borderValue=0.0)

        # tform3 = trans.ProjectiveTransform()
        # tform3.estimate(src, dst)
        # warped = trans.warp(img, tform3, output_shape=_shape)
        return warped
