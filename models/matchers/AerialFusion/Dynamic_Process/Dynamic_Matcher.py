import numpy as np
import torch
from scipy.stats import norm
import matplotlib.pyplot as plt
import matplotlib
import cv2
from sklearn.neighbors import NearestNeighbors
import numba as nb
from numba import cuda, njit, float32
from numba import jit
import torch.nn as nn
import math
import torch.nn.functional as F

class SkewedGaussianGPU:
    def __init__(self, width = 640, height = 480, grid_stride = 2):
        """
        初始化GPU计算器
        :param width: 输出图像宽度
        :param height: 输出图像高度
        :param grid_stride: 降采样步长
        """
        self.width = width
        self.height = height
        self.grid_stride = grid_stride
        self.dtype = np.float64
        self.scales = [0.7500, 1.3333]

        self.threads_per_block = (16, 16)
        self.output_shape = (height // self.grid_stride, width // self.grid_stride)
        self.blocks_per_grid = (math.ceil(self.output_shape[1] / self.threads_per_block[0]),
                                math.ceil(self.output_shape[0] / self.threads_per_block[1]))


    def build_gaussian_lut(self, size, sigma = 3):
        """生成高斯查找表"""
        x = np.linspace(-3 * sigma, 3 * sigma, size)
        x = np.exp(-0.5 * (x ** 2) / (sigma ** 2))
        return x

    @staticmethod
    @cuda.jit
    def gaussian_kernel(peaks, sigmas, x, output):
        i = cuda.grid(1)                                      # 分布索引
        j = cuda.threadIdx.x                                  # 横坐标索引
        if i < peaks.shape[0] and j < x.shape[0]:
            sigma = sigmas[i]
            x_val = x[j]
            output[i, j] = peaks[i] * math.exp(-0.5 * (x_val ** 2) / (sigma ** 2))

    def generate_gaussians_cuda(self, peaks, size=256):
        peaks = torch.abs(torch.tensor(peaks, dtype=torch.float64)).cuda()

        sigmas = 1 / (peaks * math.sqrt(2 * math.pi))
        x = torch.linspace(-3 * sigmas.mean(), 3 * sigmas.mean(), size).cuda()

        output = torch.empty((len(peaks), size), device='cuda')
        threads_per_block = 1
        blocks_per_grid = (len(peaks) + threads_per_block - 1) // threads_per_block
        self.gaussian_kernel[blocks_per_grid, threads_per_block](peaks, sigmas, x, output)

        return output

    @staticmethod
    @cuda.jit
    def _skewed_gaussian_kernel(grid_stride, mu, alpha, peaks, lut, output):
        x, y = cuda.grid(2)                                                                                             # 每个线程处理一个维度坐标
        if x >= output.shape[1] or y >= output.shape[0]:
            return

        total = 0.0
        px = x * grid_stride
        py = y * grid_stride

        for i in range(mu.shape[0]):
            dx = px - mu[i, 0]
            dy = py - mu[i, 1]

            idx_x = min(255, max(0, int(dx * 10 + 128)))                                                                # 0.5-->10 √
            idx_y = min(255, max(0, int(dy * 10 + 128)))

            gauss = (peaks[i] * lut[idx_x]) * (peaks[i] * lut[idx_y])

            skew = alpha[i, 0] * dx + alpha[i, 1] * dy
            erf_approx = skew / math.sqrt(1.0 + skew**2)
            total += gauss * (1.0 + erf_approx)

        output[y, x] = total

    def compute(self, mu, alpha, peaks, lut):
        """
        计算偏态高斯分布
        :param mu:    均值数组 (n_points, 2)
        :param alpha: 偏态参数数组 (n_points, 2)
        :param lut:   高斯查找表 (256,)
        :return:      计算结果数组 (height//stride, width//stride)
        """


        mu_gpu = cuda.to_device(mu)
        alpha_gpu = cuda.to_device(alpha.astype(self.dtype))
        peaks_gpu = cuda.to_device(peaks.astype(self.dtype))
        lut_gpu = cuda.to_device(lut.astype(self.dtype))
        output_gpu = cuda.device_array(self.output_shape, dtype=self.dtype)

        self._skewed_gaussian_kernel[self.blocks_per_grid, self.threads_per_block](self.grid_stride, mu_gpu, alpha_gpu, peaks_gpu, lut_gpu, output_gpu)

        return output_gpu.copy_to_host()


class FeaturePointPredictor:
    def __init__(self, initial_pos, dt = 1.0 / 30.0, accel_noise = 0.01):
        """
        支持位置预测的卡尔曼滤波器（匀加速模型）
        :param initial_pos: 初始位置[x,y]
        :param dt: 时间步长（帧间隔）
        :param accel_noise: 加速度噪声强度
        """
        self.kf = cv2.KalmanFilter(6, 2)  # 状态[x,y,vx,vy,ax,ay]，观测[x,y]

        self.kf.transitionMatrix = np.array([
            [1, 0, dt, 0, 0.5 * dt ** 2, 0],
            [0, 1, 0, dt, 0, 0.5 * dt ** 2],
            [0, 0, 1, 0, dt, 0],
            [0, 0, 0, 1, 0, dt],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ], dtype=np.float32)

        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0]
        ], dtype=np.float32)

        self.kf.processNoiseCov = np.diag([0, 0, 0.5, 0.5, accel_noise, accel_noise]).astype(np.float32)
        self.kf.measurementNoiseCov = np.eye(2) * 1.0  # 观测噪声

        self.kf.statePost = np.array([initial_pos[0], initial_pos[1], 0, 0, 0, 0], dtype=np.float32)

        self.predicted_pos = None                                                                              # 显式保存预测结果

    def update(self, measured_pos):
        """ 更新观测值 """

        self.predicted_pos = self.kf.predict()[:2]                      # 保存预测值


        return self.predicted_pos

    def predict_next(self, steps = 1):
        """ 预测未来steps帧后的位置 """
        state = self.kf.statePre if hasattr(self.kf, 'statePre') else self.kf.statePost
        for _ in range(steps):
            state = self.kf.transitionMatrix.dot(state)
        return state[:2]                                                                                       # 返回[x,y]


def extract_circular_patches_gpu(
        coords,     # [m, 2]
        image,      # [H, W]
        patch_size: int = 16,
        device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    GPU加速的圆形区域特征提取（最快）
    Returns:
        features: [m, patch_size * patch_size]
    """
    image_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).to(device)           # [1, 1, H, W]
    coords_tensor = coords.to(device)                                                     # [m, 2]

    mask = torch.zeros((patch_size, patch_size), device=device)
    y, x = torch.meshgrid(
        torch.arange(patch_size, device=device),
        torch.arange(patch_size, device=device))
    center = patch_size // 2
    mask[(x - center) ** 2 + (y - center) ** 2 <= center ** 2] = 1

    m = coords_tensor.shape[0]
    radius = patch_size // 2
    grid = torch.zeros((m, patch_size, patch_size, 2), device=device)

    for i in range(m):
        x, y = coords_tensor[i]
        grid[i, :, :, 0] = (torch.linspace(x - radius, x + radius, patch_size, device=device) / image.shape[1] * 2 - 1)
        grid[i, :, :, 1] = (torch.linspace(y - radius, y + radius, patch_size, device=device) / image.shape[0] * 2 - 1)

    patches = F.grid_sample(
        image_tensor.expand(m, -1, -1, -1).float(),
        grid.float(),
        mode='bilinear',
        padding_mode='zeros',
        align_corners=False
    )  # [m, 1, patch_size, patch_size]

    features = (patches.squeeze(1) * mask).view(m, -1)
    return features
