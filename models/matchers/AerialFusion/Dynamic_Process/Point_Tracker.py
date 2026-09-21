import numpy
import numpy as np
import torch
import cv2

myjet = np.array([[0.        , 0.        , 0.5       ],
                  [0.        , 0.        , 0.99910873],
                  [0.        , 0.37843137, 1.        ],
                  [0.        , 0.83333333, 1.        ],
                  [0.30044276, 1.        , 0.66729918],
                  [0.66729918, 1.        , 0.30044276],
                  [1.        , 0.90123457, 0.        ],
                  [1.        , 0.48002905, 0.        ],
                  [0.99910873, 0.07334786, 0.        ],
                  [0.5       , 0.        , 0.        ]])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class PointTracker(object):
  """
  Class to manage a fixed memory of points and descriptors that enables
  sparse optical flow point tracking.

  Internally, the tracker stores a 'tracks' matrix sized M x (2+L), of M
  tracks with maximum length L, where each row corresponds to:
  row_m = [track_id_m, avg_desc_score_m, point_id_0_m, ..., point_id_L-1_m].
  """

  def __init__(self, max_length, width, height, matcher, desc_dim):
    if max_length < 2:
      raise ValueError('max_length must be greater than or equal to 2.')
    self.maxl = max_length
    self.all_frame = []                                                     # 图像 序列
    self.all_pts = []                                                       # 特征点 序列
    self.all_desc = []                                                      # 描述符 序列
    self.all_scores = []                                                      # 描述符 序列

    b0, c0, H0, W0 = 1, 3, height, width
    frame_null = 255 * np.ones((c0, H0, W0), np.uint8)
    frame_null[:H0, :W0] = 0

    for n in range(self.maxl):
      self.all_frame.append(torch.from_numpy(frame_null))
      self.all_pts.append(torch.from_numpy(np.zeros((2, 1))))                           # 不断在尾部添加最新帧特征点位置
      self.all_desc.append(torch.from_numpy(np.zeros((1, desc_dim))))
      self.all_scores.append(torch.from_numpy(np.zeros((1, 0))))

    self.last_desc = None
    self.tracks = np.zeros((0, self.maxl + 2))                                          # 构建轨迹
    self.track_count = 0
    self.max_score = 9999

    self.matcher = matcher                                                              # 调用 自身匹配方式

  def get_offsets(self):
    """ Iterate through list of points and accumulate an offset value. Used to
    index the global point IDs into the list of points.
    迭代点列表并累积偏移值。用于将全局点id索引到点列表中。

    Returns
      offsets - N length array with integer offset locations.
    """
    offsets = []
    offsets.append(0)
    for i in range(len(self.all_pts) - 1):                             # Skip last camera size, not needed.
      offsets.append(self.all_pts[i].shape[1])
    offsets = np.array(offsets)
    offsets = np.cumsum(offsets)
    return offsets

  def update(self, frame, pts, desc, scores):
    """ Add a new set of point and descriptor observations to the tracker.
    Inputs
      frame : frame
      pts - 3xN numpy array of 2D point observations.
      desc - DxN numpy array of corresponding D dimensional descriptors.
    """
    if pts is None or desc is None:
      print('PointTracker: Warning, no points were added to tracker.')
      return
    assert pts.shape[0] == desc.shape[0]
    if self.last_desc is None:
      self.last_desc = torch.from_numpy(np.zeros((1, desc.shape[1])))                      # 构建
    
    self.all_frame.pop(0)                                                       # 移除原表最老帧, [0]位置
    self.all_frame.append(frame)                                                # 在all_frame尾部添加新帧，实现队列运算

    remove_size = self.all_pts[0].shape[1]
    self.all_pts.pop(0)                                                         # 移除原表最老帧特征点, [0]位置
    self.all_pts.append(pts.T)                                                  # 在all_pts尾部添加新帧特征点，实现队列运算

    self.all_desc.pop(0)                                                        # 移除原表最老帧描述子, [0]位置
    self.all_desc.append(desc)                                                  # 在all_desc尾部添加新帧描述符，实现队列运算

    self.all_scores.pop(0)                                                      # 移除原表最老帧描述子, [0]位置
    self.all_scores.append(scores)                                              # 在all_desc尾部添加新帧描述符，实现队列运算

    self.tracks = np.delete(self.tracks, 2, axis=1)

    for i in range(2, self.tracks.shape[1]):
      self.tracks[:, i] -= remove_size
    self.tracks[:, 2:][self.tracks[:, 2:] < -1] = -1
    offsets = self.get_offsets()
    self.tracks = np.hstack((self.tracks, -1 * np.ones((self.tracks.shape[0], 1))))         # 拼接
    matched = np.zeros((pts.shape[0])).astype(bool)

    """根据描述子匹配特征点"""
    if self.last_desc.shape[0] == 0 or desc.shape[0] == 0:                                      # 若存在一帧为空特征点和描述
     matches = np.zeros((3, 0))
    else:
     data_track = {'img0': torch.unsqueeze(self.all_frame[-2], dim=0),
                   'img1': torch.unsqueeze(frame, dim=0),
                   'keypoints0': torch.unsqueeze(self.all_pts[-2].T, dim=0),
                   'keypoints1': torch.unsqueeze(self.all_pts[-1].T, dim=0),
                   'descriptors0': torch.unsqueeze(self.last_desc, dim=0),
                   'descriptors1': torch.unsqueeze(desc, dim=0),
                   'scores0': self.all_scores[-2], 'scores1': scores,
                   'modal': 'same'}                                                             # 构建 特征对

     matches_lg = self.matcher(data_track)                                                      # 使用本匹配算法进行特征描述子匹配
     matches_mpts1 = matches_lg['return_or']['matches0'][0].cpu().numpy()                                    #
     valid_tr = matches_mpts1 > -1                                                              # 有效匹配

     dmat = np.dot(self.last_desc, desc.T)
     dmat = np.sqrt(2 - 2 * np.clip(dmat, -1, 1))
     idx = np.argmin(dmat, axis=1)
     m_idx1 = np.arange(self.last_desc.T.shape[1])[valid_tr]
     m_idx2 = idx[valid_tr]

     matches = np.zeros((3, int(valid_tr.sum())))
     matches[0, :] = m_idx1
     matches[1, :] = m_idx2
     matches[2, :] = (matches_lg['return_or']['matching_scores0'][0].cpu().detach().numpy())[valid_tr]

    for match in matches.T:
      id1 = int(match[0]) + offsets[-2]
      id2 = int(match[1]) + offsets[-1]

      found = np.argwhere(self.tracks[:, -2] == id1)[0]

      if found.shape[0] > 0:
        matched[int(match[1])] = True
        row = int(found)
        self.tracks[row, -1] = id2
        if self.tracks[row, 1] == self.max_score:
          self.tracks[row, 1] = match[2]
        else:
          track_len = (self.tracks[row, 2:] != -1).sum() - 1.
          frac = 1. / float(track_len)
          self.tracks[row, 1] = (1.-frac) * self.tracks[row, 1] + frac*match[2]

    new_ids = np.arange(pts.shape[0]) + offsets[-1]
    new_ids = new_ids[~matched]
    new_tracks = -1*np.ones((new_ids.shape[0], self.maxl + 2))                        # 利用最短跟踪帧数
    new_tracks[:, -1] = new_ids
    new_num = new_ids.shape[0]
    new_trackids = self.track_count + np.arange(new_num)
    new_tracks[:, 0] = new_trackids
    new_tracks[:, 1] = self.max_score*np.ones(new_ids.shape[0])
    self.tracks = np.vstack((self.tracks, new_tracks))
    self.track_count += new_num                                                       # Update the track count. 更新曲目计数。
    keep_rows = np.any(self.tracks[:, 2:] >= 0, axis=1)
    self.tracks = self.tracks[keep_rows, :]

    self.last_desc = desc.clone()                                                     # self.last_desc = desc.copy()
    return

  def get_tracks(self, min_length):
    """ Retrieve point tracks of a given minimum length.
    Input
      min_length - integer >= 1 with minimum track length
    Output
      returned_tracks - M x (2+L) sized matrix storing track indices, where
        M is the number of tracks and L is the maximum track length.
    """
    if min_length < 1:
      raise ValueError('\'min_length\' too small.')
    valid = np.ones((self.tracks.shape[0])).astype(bool)
    good_len = np.sum(self.tracks[:, 2:] != -1, axis=1) >= min_length                            # good_len > min_length
    not_headless = (self.tracks[:, -1] != -1)
    keepers = np.logical_and.reduce((valid, good_len, not_headless))
    returned_tracks = self.tracks[keepers, :].copy()
    return returned_tracks

  def draw_tracks(self, out, tracks, kpt_t_track, desc_t_track, scores_t_track, scales):
    """ Visualize tracks all overlayed on a single image.
    Inputs
      out - numpy uint8 image sized HxWx3 upon which tracks are overlayed.
      tracks - M x (2+L) sized matrix storing track info.
    """
    pts_mem = self.all_pts
    desc_mem = self.all_desc
    scores_mem = self.all_scores
    
    N = len(pts_mem)                                                        # Number of cameras/images.
    N_d = len(desc_mem)
    assert N == N_d
    offsets = self.get_offsets()
    stroke = 1
    k = 0
    for track in tracks:
      clr = (0, 0, 255)

      kpt_t_track.append([])
      desc_t_track.append([])
      scores_t_track.append([])

      for i in range(N-1):
        """对每个跟踪序列进行遍历"""
        if track[i+2] == -1 or track[i+3] == -1:
          continue

        offset1 = offsets[i]
        offset2 = offsets[i+1]
        idx1 = int(track[i+2] - offset1)                                       # 前一帧 特征点 跟踪 索引
        idx2 = int(track[i+3] - offset2)                                       # 当前帧 特征点 跟踪 索引

        """前后两帧特征点位置"""
        pt1 = pts_mem[i][:2, idx1].numpy()                                           # 前一帧 特征点
        pt2 = pts_mem[i+1][:2, idx2].numpy()                                         # 当前帧 特征点对应
        
        desc2 = desc_mem[i + 1][idx2]                                            # 当前帧 描述子

        scores2 = scores_mem[i + 1][idx2]                                          # 当前帧 置信

        assert desc2.shape[0] == self.last_desc.shape[1]

        kpt_t_track[k].append(pt2)                                                     # 将当前帧被 跟踪特征点 添加入 帧特征点 跟踪队列
        desc_t_track[k].append(np.array(desc2))                                        # 将当前帧被 跟踪描述子 添加入 帧描述子 跟踪队列
        scores_t_track[k].append(scores2)

        """画出特征点及其跟踪轨迹（可忽略）"""
        p1 = (int(round(pt1[0])), int(round(pt1[1])))                              # 四舍五入至int类型
        p2 = (int(round(pt2[0])), int(round(pt2[1])))

        if scales is not None:
            p1 = (np.array(p1) + 0.5) / scales.numpy() - 0.5
            p2 = (np.array(p2) + 0.5) / scales.numpy() - 0.5
            p1 = (int(round(p1[0])), int(round(p1[1])))
            p2 = (int(round(p2[0])), int(round(p2[1])))
        else:
            p1 = p1
            p2 = p2



      k = k + 1

    return kpt_t_track, desc_t_track, scores_t_track, out                        # 返回本帧图片所有符合跟踪条件，特征点跟踪序列
