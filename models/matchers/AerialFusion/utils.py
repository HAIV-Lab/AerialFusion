from pathlib import Path
import time
from collections import OrderedDict
from threading import Thread
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')

class AverageTimer:
    """ Class to help manage printing simple timing of code execution. """

    def __init__(self, smoothing=0.3, newline=False):
        self.smoothing = smoothing
        self.newline = newline
        self.times = OrderedDict()
        self.will_print = OrderedDict()
        self.reset()

    def reset(self):
        now = time.time()
        self.start = now
        self.last_time = now
        for name in self.will_print:
            self.will_print[name] = False

    def update(self, name='default'):
        now = time.time()
        dt = now - self.last_time
        if name in self.times:
            dt = self.smoothing * dt + (1 - self.smoothing) * self.times[name]
        self.times[name] = dt
        self.will_print[name] = True
        self.last_time = now

    def print(self, text='Timer'):
        total = 0.
        print('[{}]'.format(text), end=' ')
        for key in self.times:
            val = self.times[key]
            if self.will_print[key]:
                print('%s=%.3f' % (key, val), end=' ')
                total += val
        print('total=%.3f sec {%.1f FPS}' % (total, 1./total), end=' ')
        if self.newline:
            print(flush=True)
        else:
            print(end='\r', flush=True)
        self.reset()


class VideoStreamer:
    """ Class to help process image streams. Four types of possible inputs:"
        1.) USB Webcam.
        2.) An IP camera
        3.) A directory of images (files in directory matching 'image_glob').
        4.) A video file, such as an .mp4 or .avi file.
    """
    def __init__(self, basedir, resize, skip, image_glob, max_length=1000000):
        self._ip_grabbed = False
        self._ip_running = False
        self._ip_camera = False
        self._ip_image = None
        self._ip_index = 0
        self.cap = []
        self.camera = True
        self.video_file = False
        self.listing = []
        self.resize = resize
        self.interp = cv2.INTER_AREA
        self.i = 0
        self.skip = skip
        self.max_length = max_length
        if isinstance(basedir, int) or basedir.isdigit():
            print('==> Processing USB webcam input: {}'.format(basedir))
            self.cap = cv2.VideoCapture(int(basedir))
            self.listing = range(0, self.max_length)
        elif basedir.startswith(('http', 'rtsp')):
            print('==> Processing IP camera input: {}'.format(basedir))
            self.cap = cv2.VideoCapture(basedir)
            self.start_ip_camera_thread()
            self._ip_camera = True
            self.listing = range(0, self.max_length)
        elif Path(basedir).is_dir():
            print('==> Processing image directory input: {}'.format(basedir))
            self.listing = list(Path(basedir).glob(image_glob[0]))
            for j in range(1, len(image_glob)):
                image_path = list(Path(basedir).glob(image_glob[j]))
                self.listing = self.listing + image_path
            self.listing.sort()
            self.listing = self.listing[::self.skip]
            self.max_length = np.min([self.max_length, len(self.listing)])
            if self.max_length == 0:
                raise IOError('No images found (maybe bad \'image_glob\' ?)')
            self.listing = self.listing[:self.max_length]
            self.camera = False
        elif Path(basedir).exists():
            print('==> Processing video input: {}'.format(basedir))
            self.cap = cv2.VideoCapture(basedir)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            num_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.listing = range(0, num_frames)
            self.listing = self.listing[::self.skip]
            self.video_file = True
            self.max_length = np.min([self.max_length, len(self.listing)])
            self.listing = self.listing[:self.max_length]
        else:
            raise ValueError('VideoStreamer input \"{}\" not recognized.'.format(basedir))
        if self.camera and not self.cap.isOpened():
            raise IOError('Could not read camera')

    def load_image(self, impath):
        """ Read image as grayscale and resize to img_size.
        Inputs
            impath: Path to input image.
        Returns
            grayim: uint8 numpy array sized H x W.
        """
        grayim = cv2.imread(impath, 0)
        if grayim is None:
            raise Exception('Error reading image %s' % impath)
        w, h = grayim.shape[1], grayim.shape[0]
        w_new, h_new = process_resize(w, h, self.resize)
        grayim = cv2.resize(
            grayim, (w_new, h_new), interpolation=self.interp)
        return grayim

    def next_frame(self):
        """ Return the next frame, and increment internal counter.
        Returns
             image: Next H x W image.
             status: True or False depending whether image was loaded.

        返回下一帧，并增加内部计数器。
        返回
            图片:下一张高x宽的图片。
            status: True或False，取决于图像是否加载。
        """

        if self.i == self.max_length:
            return (None, False)

        if self.camera:
            if self._ip_camera:
                while self._ip_grabbed is False and self._ip_exited is False:
                    time.sleep(.001)

                ret, image = self._ip_grabbed, self._ip_image.copy()
                if ret is False:
                    self._ip_running = False
            else:
                ret, image = self.cap.read()
            if ret is False:
                print('VideoStreamer: Cannot get image from camera')
                return (None, False)
            w, h = image.shape[1], image.shape[0]
            if self.video_file:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.listing[self.i])

            w_new, h_new = process_resize(w, h, self.resize)
            image = cv2.resize(image, (w_new, h_new), interpolation=self.interp)
            image_color = image                                                         # 原三通道图像
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)                             # 转换为灰度图

        else:
            image_file = str(self.listing[self.i])
            image = self.load_image(image_file)
        self.i = self.i + 1
        return (image, image_color, True)

    def start_ip_camera_thread(self):
        self._ip_thread = Thread(target=self.update_ip_camera, args=())
        self._ip_running = True
        self._ip_thread.start()
        self._ip_exited = False
        return self

    def update_ip_camera(self):
        while self._ip_running:
            ret, img = self.cap.read()
            if ret is False:
                self._ip_running = False
                self._ip_exited = True
                self._ip_grabbed = False
                return

            self._ip_image = img
            self._ip_grabbed = ret
            self._ip_index += 1


    def cleanup(self):
        self._ip_running = False


def process_resize(w, h, resize):
    assert(len(resize) > 0 and len(resize) <= 2)
    if len(resize) == 1 and resize[0] > -1:
        scale = resize[0] / max(h, w)
        w_new, h_new = int(round(w*scale)), int(round(h*scale))
    elif len(resize) == 1 and resize[0] == -1:
        w_new, h_new = w, h
    else:  # len(resize) == 2:
        w_new, h_new = resize[0], resize[1]

    if max(w_new, h_new) < 160:
        print('Warning: input resolution is very small, results may vary')
    elif max(w_new, h_new) > 2000:
        print('Warning: input resolution is very large, results may vary')

    return w_new, h_new


def frame2tensor(frame, device):
    return torch.from_numpy(frame/255.).float()[None, None].to(device)


def read_image(path, device, resize, rotation, resize_float):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None, None, None
    w, h = image.shape[1], image.shape[0]
    w_new, h_new = process_resize(w, h, resize)
    scales = (float(w) / float(w_new), float(h) / float(h_new))

    if resize_float:
        image = cv2.resize(image.astype('float32'), (w_new, h_new))
    else:
        image = cv2.resize(image, (w_new, h_new)).astype('float32')

    if rotation != 0:
        image = np.rot90(image, k=rotation)
        if rotation % 2:
            scales = scales[::-1]

    inp = frame2tensor(image, device)
    return image, inp, scales




def estimate_pose(kpts0, kpts1, K0, K1, thresh, conf=0.99999):
    if len(kpts0) < 5:
        return None

    f_mean = np.mean([K0[0, 0], K1[1, 1], K0[0, 0], K1[1, 1]])
    norm_thresh = thresh / f_mean

    kpts0 = (kpts0 - K0[[0, 1], [2, 2]][None]) / K0[[0, 1], [0, 1]][None]
    kpts1 = (kpts1 - K1[[0, 1], [2, 2]][None]) / K1[[0, 1], [0, 1]][None]

    E, mask = cv2.findEssentialMat(
        kpts0, kpts1, np.eye(3), threshold=norm_thresh, prob=conf,
        method=cv2.RANSAC)

    assert E is not None

    best_num_inliers = 0
    ret = None
    for _E in np.split(E, len(E) / 3):
        n, R, t, _ = cv2.recoverPose(
            _E, kpts0, kpts1, np.eye(3), 1e9, mask=mask)
        if n > best_num_inliers:
            best_num_inliers = n
            ret = (R, t[:, 0], mask.ravel() > 0)
    return ret


def rotate_intrinsics(K, image_shape, rot):
    """image_shape is the shape of the image after rotation"""
    assert rot <= 3
    h, w = image_shape[:2][::-1 if (rot % 2) else 1]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    rot = rot % 4
    if rot == 1:
        return np.array([[fy, 0., cy],
                         [0., fx, w-1-cx],
                         [0., 0., 1.]], dtype=K.dtype)
    elif rot == 2:
        return np.array([[fx, 0., w-1-cx],
                         [0., fy, h-1-cy],
                         [0., 0., 1.]], dtype=K.dtype)
    else:  # if rot == 3:
        return np.array([[fy, 0., h-1-cy],
                         [0., fx, cx],
                         [0., 0., 1.]], dtype=K.dtype)


def rotate_pose_inplane(i_T_w, rot):
    rotation_matrices = [
        np.array([[np.cos(r), -np.sin(r), 0., 0.],
                  [np.sin(r), np.cos(r), 0., 0.],
                  [0., 0., 1., 0.],
                  [0., 0., 0., 1.]], dtype=np.float32)
        for r in [np.deg2rad(d) for d in (0, 270, 180, 90)]
    ]
    return np.dot(rotation_matrices[rot], i_T_w)


def scale_intrinsics(K, scales):
    scales = np.diag([1./scales[0], 1./scales[1], 1.])
    return np.dot(scales, K)


def to_homogeneous(points):
    return np.concatenate([points, np.ones_like(points[:, :1])], axis=-1)


def compute_epipolar_error(kpts0, kpts1, T_0to1, K0, K1):
    kpts0 = (kpts0 - K0[[0, 1], [2, 2]][None]) / K0[[0, 1], [0, 1]][None]
    kpts1 = (kpts1 - K1[[0, 1], [2, 2]][None]) / K1[[0, 1], [0, 1]][None]
    kpts0 = to_homogeneous(kpts0)
    kpts1 = to_homogeneous(kpts1)

    t0, t1, t2 = T_0to1[:3, 3]
    t_skew = np.array([
        [0, -t2, t1],
        [t2, 0, -t0],
        [-t1, t0, 0]
    ])
    E = t_skew @ T_0to1[:3, :3]

    Ep0 = kpts0 @ E.T  # N x 3
    p1Ep0 = np.sum(kpts1 * Ep0, -1)  # N
    Etp1 = kpts1 @ E  # N x 3
    d = p1Ep0**2 * (1.0 / (Ep0[:, 0]**2 + Ep0[:, 1]**2)
                    + 1.0 / (Etp1[:, 0]**2 + Etp1[:, 1]**2))
    return d


def angle_error_mat(R1, R2):
    cos = (np.trace(np.dot(R1.T, R2)) - 1) / 2
    cos = np.clip(cos, -1., 1.)  # numercial errors can make it out of bounds
    return np.rad2deg(np.abs(np.arccos(cos)))


def angle_error_vec(v1, v2):
    n = np.linalg.norm(v1) * np.linalg.norm(v2)
    return np.rad2deg(np.arccos(np.clip(np.dot(v1, v2) / n, -1.0, 1.0)))


def compute_pose_error(T_0to1, R, t):
    R_gt = T_0to1[:3, :3]
    t_gt = T_0to1[:3, 3]
    error_t = angle_error_vec(t, t_gt)
    error_t = np.minimum(error_t, 180 - error_t)  # ambiguity of E estimation
    error_R = angle_error_mat(R, R_gt)
    return error_t, error_R


def pose_auc(errors, thresholds):
    sort_idx = np.argsort(errors)
    errors = np.array(errors.copy())[sort_idx]
    recall = (np.arange(len(errors)) + 1) / len(errors)
    errors = np.r_[0., errors]
    recall = np.r_[0., recall]
    aucs = []
    for t in thresholds:
        last_index = np.searchsorted(errors, t)
        r = np.r_[recall[:last_index], recall[last_index-1]]
        e = np.r_[errors[:last_index], t]
        aucs.append(np.trapz(r, x=e)/t)
    return aucs




def plot_image_pair(imgs, dpi=100, size=6, pad=.5):
    n = len(imgs)
    assert n == 2, 'number of images must be two'
    figsize = (size*n, size*3/4) if size is not None else None
    _, ax = plt.subplots(1, n, figsize=figsize, dpi=dpi)
    for i in range(n):
        ax[i].imshow(imgs[i], cmap=plt.get_cmap('gray'), vmin=0, vmax=255)
        ax[i].get_yaxis().set_ticks([])
        ax[i].get_xaxis().set_ticks([])
        for spine in ax[i].spines.values():  # remove frame
            spine.set_visible(False)
    plt.tight_layout(pad=pad)


def plot_keypoints(kpts0, kpts1, color='w', ps=2):
    ax = plt.gcf().axes
    ax[0].scatter(kpts0[:, 0], kpts0[:, 1], c=color, s=ps)
    ax[1].scatter(kpts1[:, 0], kpts1[:, 1], c=color, s=ps)


def plot_matches(kpts0, kpts1, color, lw=1.5, ps=4):
    fig = plt.gcf()
    ax = fig.axes
    fig.canvas.draw()

    transFigure = fig.transFigure.inverted()
    fkpts0 = transFigure.transform(ax[0].transData.transform(kpts0))
    fkpts1 = transFigure.transform(ax[1].transData.transform(kpts1))

    fig.lines = [matplotlib.lines.Line2D(
        (fkpts0[i, 0], fkpts1[i, 0]), (fkpts0[i, 1], fkpts1[i, 1]), zorder=1,
        transform=fig.transFigure, c=color[i], linewidth=lw)
                 for i in range(len(kpts0))]
    ax[0].scatter(kpts0[:, 0], kpts0[:, 1], c=color, s=ps)
    ax[1].scatter(kpts1[:, 0], kpts1[:, 1], c=color, s=ps)


def make_matching_plot(image0, image1, kpts0, kpts1, mkpts0, mkpts1,
                       color, text, path, show_keypoints=False,
                       fast_viz=True, opencv_display=False,
                       opencv_title='matches', small_text=[]):


    out_1, out_2, out_3, out_4 = make_matching_plot_fast(image0, image1, kpts0, kpts1, mkpts0, mkpts1,
                            color, text, path, show_keypoints, 10,
                            opencv_display, opencv_title, small_text)



    return out_1, out_2, out_3, out_4

def make_matching_plot_fast(image0, image1, kpts0, kpts1, mkpts0,
                            mkpts1, color, text, path=None,
                            show_keypoints=False, margin=5,
                            opencv_display=False, opencv_title='',
                            small_text=[], mask = None):
    H0, W0, c_0 = image0.shape
    H1, W1, c_1 = image1.shape

    H, W = max(H0, H1), W0 + W1 + margin      # 横向构建
    H_1, W_1 = H0 + H1 + margin, max(W0, W1)  # 纵向构建

    out = 255 * np.ones((H, W, c_0), np.uint8)
    out[:H0, :W0] = image0
    out[:H1, W0 + margin:] = image1

    out_1 = 255 * np.ones((H_1, W_1, c_0), np.uint8)
    out_2 = 255 * np.ones((H_1, W_1, c_0), np.uint8)
    out_3 = 255 * np.ones((H1, W1, c_0), np.uint8)
    out_4 = 255 * np.ones((H0, W0, c_0), np.uint8)
    out_1[:H0, :W0] = image0
    out_1[H0 + margin:, :W1] = image1

    out_4[:H0, :W0] = image0


    border = (0, 0, 0)
    aqua = (0, 255, 255)

    cv2.line(out_2, (0, 0), (0 + W0 - 1, 0), color=border, thickness=1, lineType=cv2.LINE_AA)
    cv2.line(out_2, (0, 0), (0, 0 + H0), color=border, thickness=1, lineType=cv2.LINE_AA)
    cv2.line(out_2, (0 + W0 - 1, 0), (0 + W0 - 1, 0 + H0 - 1), color=border, thickness=1, lineType=cv2.LINE_AA)
    cv2.line(out_2, (0, 0 + H0 - 1), (0 + W0 - 1, 0 + H0 - 1), color=border, thickness=1, lineType=cv2.LINE_AA)
    cv2.line(out_2, (0, 0 + margin + H0), (0 + W0 - 1, 0 + margin + H0), color=border, thickness=1, lineType=cv2.LINE_AA)
    cv2.line(out_2, (0, 0 + margin + H0), (0, 0 + margin + H0 + H1), color=border, thickness=1, lineType=cv2.LINE_AA)
    cv2.line(out_2, (0, 0 + margin + H0 + H1 - 1), (0 + W0 - 1, 0 + margin + H0 + H1 - 1), color=border, thickness=1, lineType=cv2.LINE_AA)
    cv2.line(out_2, (0 + W0 - 1, 0 + margin + H0 - 1), (0 + W0 - 1, 0 + margin + H0 + H1 - 1), color=border, thickness=1, lineType=cv2.LINE_AA)

    cv2.line(out_3, (0, 0), (0 + W0 - 1, 0), color=aqua, thickness=2, lineType=cv2.LINE_AA)
    cv2.line(out_3, (0, 0), (0, 0 + H0), color=aqua, thickness=2, lineType=cv2.LINE_AA)
    cv2.line(out_3, (0 + W0 - 1, 0), (0 + W0 - 1, 0 + H0 - 1), color=aqua, thickness=2, lineType=cv2.LINE_AA)
    cv2.line(out_3, (0, 0 + H0 - 1), (0 + W0 - 1, 0 + H0 - 1), color=aqua, thickness=2, lineType=cv2.LINE_AA)

    n = 16
    m = 16

    W_grid = W0 / n
    H_grid = H0 / m

    W0_base = 0
    H0_base = 0
    W1_base = 0
    H1_base = 0

    while W1_base < W0 or H0_base < H0:
        cv2.line(out_2, (int(W0_base + W_grid), int(0)), (int(W0_base + W_grid), int(H0)), color=border, thickness=1, lineType=cv2.LINE_AA)
        cv2.line(out_2, (int(0), int(H0_base + H_grid)), (int(W0), int(H0_base + H_grid)), color=border, thickness=1, lineType=cv2.LINE_AA)

        cv2.line(out_2, (int(W1_base + W_grid), int(margin + H0)), (int(W1_base + W_grid), int(H1 + margin + H0 - 1)), color=border, thickness=1, lineType=cv2.LINE_AA)
        cv2.line(out_2, (int(0), int(margin + H0 + H1_base + H_grid)), (int(W1), int(margin + H0 + H1_base + H_grid)), color=border, thickness=1, lineType=cv2.LINE_AA)

        cv2.line(out_3, (int(W0_base + W_grid), int(0)), (int(W0_base + W_grid), int(H0)), color=aqua, thickness=2, lineType=cv2.LINE_AA)
        cv2.line(out_3, (int(0), int(H0_base + H_grid)), (int(W0), int(H0_base + H_grid)), color=aqua, thickness=2, lineType=cv2.LINE_AA)

        W0_base = W0_base + W_grid
        H0_base = H0_base + H_grid
        W1_base = W1_base + W_grid
        H1_base = H1_base + H_grid

    if show_keypoints:
        kpts0, kpts1 = np.round(kpts0.cpu().numpy()).astype(int), np.round(kpts1.cpu().numpy()).astype(int)
        mkpts0, mkpts1 = np.round(mkpts0.cpu().numpy()).astype(int), np.round(mkpts1.cpu().numpy()).astype(int)
        red = (255, 0, 0)
        green = (0, 255, 0)
        white = (255, 255, 255)
        black = (0, 0, 0)

        for x, y in kpts0:
            cv2.circle(out, (x, y), 1, green, -1, lineType=cv2.LINE_AA)
            cv2.circle(out, (x, y), 1, green, -1, lineType=cv2.LINE_AA)
            
        for x, y in kpts1:
            cv2.circle(out, (x + margin + W0, y), 1, green, -1, lineType=cv2.LINE_AA)
            cv2.circle(out, (x + margin + W0, y), 1, green, -1, lineType=cv2.LINE_AA)

        for x, y in kpts0:
            cv2.circle(out_1, (x, y), 2, red, -1, lineType=cv2.LINE_AA)
            cv2.circle(out_1, (x, y), 1, red, -1, lineType=cv2.LINE_AA)

        for x, y in kpts1:
            cv2.circle(out_1, (x, y + margin + H0), 2, red, -1, lineType=cv2.LINE_AA)
            cv2.circle(out_1, (x, y + margin + H0), 1, red, -1, lineType=cv2.LINE_AA)

        for x, y in mkpts0:
            cv2.circle(out_2, (x, y), 6, red, -1, lineType=cv2.LINE_AA)
            cv2.circle(out_2, (x, y), 3, red, -1, lineType=cv2.LINE_AA)

        for x, y in mkpts1:
            cv2.circle(out_2, (x, y + margin + H0), 6, red, -1, lineType=cv2.LINE_AA)
            cv2.circle(out_2, (x, y + margin + H0), 3, red, -1, lineType=cv2.LINE_AA)

        for x, y in mkpts0:
            cv2.circle(out_3, (x, y), 3, red, -1, lineType=cv2.LINE_AA)
            cv2.circle(out_3, (x, y), 2, red, -1, lineType=cv2.LINE_AA)

        for x, y in mkpts1:
            cv2.circle(out_3, (x, y), 3, red, -1, lineType=cv2.LINE_AA)
            cv2.circle(out_3, (x, y), 2, red, -1, lineType=cv2.LINE_AA)

        for x, y in kpts0:
            cv2.circle(out_4, (x, y), 1, green, -1, lineType=cv2.LINE_AA)
            cv2.circle(out_4, (x, y), 1, green, -1, lineType=cv2.LINE_AA)

    mkpts0, mkpts1 = np.array(np.round(mkpts0)).astype(int), np.array(np.round(mkpts1)).astype(int)

    for (x0, y0), (x1, y1) in zip(mkpts0, mkpts1):
        c = (0, 255, 0)
        c_1 = (0, 0, 255)

        cv2.line(out, (x0, y0), (x1 + margin + W0, y1), color=c, thickness=1, lineType=cv2.LINE_AA)

        cv2.circle(out, (x0, y0), 2, c, -1, lineType=cv2.LINE_AA)
        cv2.circle(out, (x1 + margin + W0, y1), 2, c, -1, lineType=cv2.LINE_AA)

        cv2.line(out_1, (x0, y0), (x1, y1 + margin + H0), color=c, thickness=1, lineType=cv2.LINE_AA)
        cv2.circle(out_1, (x0, y0), 2, c, -1, lineType=cv2.LINE_AA)
        cv2.circle(out_1, (x1, y1 + margin + H0), 2, c, -1, lineType=cv2.LINE_AA)

        cv2.line(out_2, (x0, y0), (x1, y1 + margin + H0), color=c_1, thickness=1, lineType=cv2.LINE_AA)


    c_r = (0, 0, 255)
    if mask is not None:
        for (x0, y0), (x1, y1) in zip(mkpts0[mask == 0], mkpts1[mask == 0]):
            cv2.line(out, (x0, y0), (x1 + margin + W0, y1), color=c_r, thickness=1, lineType=cv2.LINE_AA)






    return out, out_1, out_3, out_4


def error_colormap(x):
    return np.clip(
        np.stack([2-x*2, x*2, np.zeros_like(x), np.ones_like(x)], -1), 0, 1)
