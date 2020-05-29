import cv2

import math
import random

import numpy as np

import torch
from torch import nn
import torch.nn.functional as F


class FeaturesMap(nn.Module):
    def __init__(self, train_dummy, f_channels=512, max_height=70,
                 max_width=40, f_size=1, dummy='zero', mask_out=False):
        super().__init__()

        self.f_size = f_size
        self.max_height = max_height * self.f_size
        self.max_width = max_width * self.f_size
        self.f_channels = f_channels
        self.dummy = dummy
        self.mask_out = mask_out

        if dummy not in ['self_mean', 'zero', 'zero_f', 'mean_f']:
            raise Exception(f"Unknown dummy backend {dummy}")

        self.backend_feature = None
        if self.dummy == 'zero':
            self.backend_feature = torch.full((self.f_channels, 1, 1), 0)
        elif self.dummy == 'zero_f':
            print("Load zero features")
            self.backend_feature = torch.load("../../zero_512x8x8.pth").\
                mean(-1).mean(-1).view(self.f_channels, 1, 1)
        elif self.dummy == 'mean_f':
            print("Load mean features")
            self.backend_feature = torch.load("../../train_mean_512x8x8.pth").\
                view(self.f_channels, 1, 1)

        if self.backend_feature is not None:
            if train_dummy:
                self.backend_feature = nn.Parameter(self.backend_feature)
            else:
                self.register_buffer('backend_feature', self.backend_feature)

    def forward(self, features, ys, xs, validation=None):
        if validation is None:
            validation = not self.training

        b_sz = features.shape[0]

        if self.backend_feature is not None:
            f_map = self.backend_feature.expand(self.f_channels,
                                                self.max_height,
                                                self.max_width)

        res_f_maps = []
        if self.mask_out:
            res_real_masks = []
        for b in range(b_sz):
            real_mask = ys[b] > -1

            min_y, max_y = (ys[b, real_mask].min().item(),
                            ys[b, real_mask].max().item())
            min_x, max_x = (xs[b, real_mask].min().item(),
                            xs[b, real_mask].max().item())

            height = (max_y - min_y + 1) * self.f_size
            width = (max_x - min_x + 1) * self.f_size

            if self.dummy == 'self_mean':
                f_map = features[b, :, real_mask].mean(dim=1)[:, None, None].\
                    expand(self.f_channels, self.max_height, self.max_width)

            tmp_f_map = torch.full((self.f_channels, height, width), -1,
                                   dtype=features.dtype,
                                   device=features.device)

            if self.f_size == 1:
                tmp_f_map[:, ys[b, real_mask] - min_y,
                          xs[b, real_mask] - min_x] = features[b, :, real_mask]
            else:
                for n in range(real_mask.sum().item()):
                    t = features[b, n]
                    _y = ys[b, n] - min_y
                    _x = xs[b, n] - min_x
                    tmp_f_map[:, _y*self.f_size:(_y+1)*self.f_size,
                              _x*self.f_size:(_x+1)*self.f_size] = t

            if width > height:
                tmp_f_map = tmp_f_map.transpose(-1, -2)

            _, height, width = tmp_f_map.shape
            # print(height, width)

            h_dif = height - self.max_height
            w_dif = width - self.max_width

            if h_dif > 0:
                cut_top = (math.ceil(h_dif / 2) if validation else
                           random.randint(0, h_dif))
                cut_bottom = -(h_dif - cut_top)
                if cut_bottom >= 0:
                    cut_bottom = None
                pad_top, pad_bottom = 0, 0
            else:
                pad_top = (math.ceil(-h_dif / 2) if validation else
                           random.randint(0, -h_dif))
                pad_bottom = -h_dif - pad_top
                cut_top, cut_bottom = 0, None

            if w_dif > 0:
                cut_left = (math.ceil(w_dif / 2) if validation else
                            random.randint(0, w_dif))
                cut_right = -(w_dif - cut_left)
                if cut_right >= 0:
                    cut_right = None
                pad_right, pad_left = 0, 0
            else:
                pad_right = (math.ceil(-w_dif / 2) if validation else
                             random.randint(0, -w_dif))
                pad_left = -w_dif - pad_right
                cut_left, cut_right = 0, None

            if not validation:
                if random.random() > 0.5:
                    tmp_f_map = torch.flip(tmp_f_map, [-1])

                if random.random() > 0.5:
                    tmp_f_map = torch.flip(tmp_f_map, [-2])

            tmp_f_map = F.pad(tmp_f_map[:, cut_top:cut_bottom,
                                        cut_left:cut_right],
                              (pad_right, pad_left,
                               pad_top, pad_bottom), value=-1)

            real_2d_mask = (tmp_f_map != -1).all(dim=0)

            # print(real_2d_mask.shape, f_map.shape, tmp_f_map.shape)

            res_f_map = (~real_2d_mask) * f_map + real_2d_mask * tmp_f_map
            res_f_maps.append(res_f_map)
            if self.mask_out:
                res_real_masks.append(real_2d_mask)

        return ((torch.stack(res_f_maps), torch.stack(res_real_masks))
                if self.mask_out else torch.stack(res_f_maps))


class TiledFeaturesMap(nn.Module):
    def __init__(self, f_channels=512, f_size=1,
                 t_sz=9, t_step=6, t_cut=2):
        super().__init__()

        self.f_size = f_size
        self.f_channels = f_channels
        self.t_sz = t_sz
        self.t_step = t_step
        self.t_cut = t_cut

    def forward(self, features, ys, xs, validation=None):
        if validation is None:
            validation = not self.training

        f_tiles = []
        f_ns = []

        for b in range(features.shape[0]):
            r_mask = ys[b] > -1

            y_min, x_min = ys[b, r_mask].min(), xs[b, r_mask].min()
            y_max, x_max = ys[b, r_mask].max(), xs[b, r_mask].max()

            if not validation:
                y_rnd = random.randint(0, self.t_step)
                x_rnd = random.randint(0, self.t_step)
            else:
                y_rnd, x_rnd = 0, 0

            x_map = torch.zeros((y_max-y_min+1+self.t_sz+y_rnd,
                                 x_max-x_min+1+self.t_sz+x_rnd,
                                 self.f_channels, self.f_size, self.f_size),
                                dtype=features.dtype, device=features.device)

            x_map[ys[b, r_mask]-y_min+y_rnd, xs[b, r_mask]-x_min+x_rnd] =\
                features[b, r_mask]

            x_tiles = x_map.unfold(0, self.t_sz, self.t_step).\
                unfold(1, self.t_sz, self.t_step)

            f_t_idxs = x_tiles[..., self.t_cut:-self.t_cut,
                               self.t_cut:-self.t_cut].\
                reshape(x_tiles.shape[:2]+(-1,)).sum(-1)

            if (f_t_idxs > 0).sum() == 0:  # if no tiles at all
                f_t_idxs = x_tiles.reshape(x_tiles.shape[:2]+(-1,)).sum(-1)

            f_tiles.append(x_tiles[f_t_idxs > 0])
            f_ns.extend([b, ]*(f_t_idxs > 0).sum().item())

        f_tiles = torch.cat(f_tiles, dim=0)
        f_ns = torch.tensor(f_ns)

        f_tiles = f_tiles.permute(0, 1, 4, 2, 5, 3).\
            reshape(f_tiles.shape[:2] +
                    (self.t_sz*self.f_size, self.t_sz*self.f_size))

        if not validation:
            for n in range(len(f_tiles)):
                f_tile = f_tiles[n]
                if random.random() > 0.5:
                    f_tile = torch.flip(f_tile, [-1])

                if random.random() > 0.5:
                    f_tile = torch.flip(f_tile, [-2])

                if random.random() > 0.5:
                    f_tile = f_tile.transpose(-1, -2)
                f_tiles[n] = f_tile

        return f_ns, f_tiles


def d8_transform(m, d8):
    if d8 & 0b001:
        m = m.transpose(1, 0, 2)
    if d8 & 0b010:
        m = m[:, ::-1]
    if d8 & 0b100:
        m = m[::-1, :]

    return m


d8_rev = [0, 1, 2, 5, 4, 3, 6, 7]


def torch_d8_transform(m, d8):
    if d8 & 0b001:
        m = m.transpose(-2, -1)
    if d8 & 0b010:
        m = torch.flip(m, [-1])
    if d8 & 0b100:
        m = torch.flip(m, [-2])

    return m


def get_fg_score(img, p=2):
    return (np.unique(img[..., 2], return_counts=True)[1][1:]**p).sum()


def get_bg_score(img, p=2):
    tmp = (img[..., 0] == -1).astype(np.uint8)
    dist_sum = (cv2.distanceTransform(tmp, cv2.DIST_L1, 5)**p).sum()
    return dist_sum


class RearrangedFeaturesMap(nn.Module):
    def __init__(self, use_dummy_feature, f_channels=512, f_size=1,
                 h=20, w=20,
                 p0=1.4, p1=1, p2=1, a0=1, a1=0.1, a2=1000,
                 n0=1, n1=200, rotations_only=False):
        super().__init__()

        self.f_size = f_size
        self.f_channels = f_channels
        self.t_h = h
        self.t_w = w
        assert self.t_h == self.t_w, "Not implemented yet for not square"

        self.rotations_only = rotations_only

        self.p0, self.p1, self.p2 = p0, p1, p2
        self.a0, self.a1, self.a2 = a0, a1, a2
        self.n0, self.n1 = n0, n1

        self.use_dummy_feature = use_dummy_feature

        if self.use_dummy_feature:
            self.backend_feature =\
                nn.Parameter(torch.full((1, self.f_channels, 1, 1), 0))

    def forward(self, features, ys, xs, validation=None):
        ys, xs = ys.cpu(), xs.cpu()

        if validation is None:
            validation = not self.training

        if self.use_dummy_feature:
            f_map = self.backend_feature.expand(features.shape[0],
                                                self.f_channels,
                                                self.t_h*self.f_size,
                                                self.t_w*self.f_size)
        else:
            f_map = torch.full((features.shape[0], self.f_channels,
                                self.t_h*self.f_size, self.t_w*self.f_size), 0,
                               dtype=features.dtype, device=features.device)

        # t_imgs = []

        for b in range(features.shape[0]):
            r_mask = ys[b] > - 1

            y_min, x_min = ys[b, r_mask].min(), xs[b, r_mask].min()

            n_ys = ys[b, r_mask] - y_min
            n_xs = xs[b, r_mask] - x_min
            _, t_img = self.rearrange(n_ys, n_xs)
            self.fill_features(f_map[b], features[b, r_mask],
                               t_img, n_ys, n_xs)
            # t_imgs.append(t_img)

        return f_map

    def fill_features(self, t_d8img, features, t_img, n_ys, n_xs):
        for i in range(1, t_img[..., 2].max()+1):
            t_i = t_img[..., 2] == i
            t_ys, t_xs = np.where(t_i)
            s_yxs = t_img[t_i, :2]
            d8 = t_img[t_i, -1][0]
            d8_r = d8_rev[d8]

            for n in range(len(s_yxs)):
                f_idx = (n_ys == s_yxs[n, 0]) & (n_xs == s_yxs[n, 1])
                f = features[f_idx][0]
                f = torch_d8_transform(f, d8_r)
                t_d8img[:, t_ys[n]*self.f_size:(t_ys[n]+1)*self.f_size,
                        t_xs[n]*self.f_size:(t_xs[n]+1)*self.f_size] = f

    def get_score(self, t_img, s_img):
        score = (self.a0 * get_fg_score(t_img, self.p0) +
                 self.a1 * get_bg_score(t_img, self.p1)**0.5 +
                 self.a2 * -(cv2.connectedComponents(s_img, connectivity=4)[0]
                             ** self.p2))

        return score

    def rearrange(self, n_ys, n_xs):
        t_h, t_w = self.t_h, self.t_w
        m_pad = t_h

        s_h = n_ys.max() + 1
        s_w = n_xs.max() + 1

        o_t_img = np.full((t_h, t_w, 4), -1, dtype=np.int16)

        o_s_img = np.zeros((s_h, s_w), dtype=np.uint8)
        o_s_img[n_ys, n_xs] = 1

        mask = np.ones((s_h+2*m_pad, s_w+2*m_pad), dtype=np.uint8)

        o_best_score = -99999999
        for _ in range(self.n0):
            s_img, t_img = o_s_img.copy(), o_t_img.copy()

            n = 0
            while s_img.max() != 0:
                best_score = -99999999

                s_fg_ys, s_fg_xs = np.where(s_img != 0)

                for _ in range(self.n1):
                    c_s_img, c_t_img = s_img.copy(), t_img.copy()

                    if self.rotations_only:
                        d8 = random.choice([0, 3, 5, 6])  # only rotation
                    else:
                        d8 = random.randint(0, 7)

                    c_t_img = d8_transform(c_t_img, d8)

                    t_empty_mask = c_t_img[..., 0] == -1
                    t_empty_ys, t_empty_xs = np.where(t_empty_mask)

                    r = random.randint(0, len(t_empty_ys)-1)
                    t_y, t_x = t_empty_ys[r], t_empty_xs[r]

                    r = random.randint(0, len(s_fg_ys)-1)
                    s_y, s_x = s_fg_ys[r], s_fg_xs[r]

                    mask.fill(1)

                    m_y = s_y+m_pad-t_y
                    m_x = s_x+m_pad-t_x

                    mask[m_y:m_y+t_h, m_x:m_x+t_w] = ~t_empty_mask

                    fimg = cv2.floodFill(c_s_img, mask[m_pad-1:-m_pad+1,
                                                       m_pad-1:-m_pad+1],
                                         (s_x, s_y), 2, flags=8)

                    if fimg[0] > 0:
                        blob_yxs = np.where(fimg[1] == 2)
                        c_t_img[blob_yxs[0]-s_y+t_y, blob_yxs[1]-s_x+t_x] =\
                            np.stack(blob_yxs +
                                     (np.full_like(blob_yxs[0], n+1),
                                      np.full_like(blob_yxs[0], d8)), axis=-1)

                        c_s_img[c_s_img == 2] = 0
                    else:
                        print("WTF?")

                    c_t_img = d8_transform(c_t_img, d8_rev[d8])

                    score = self.get_score(c_t_img, c_s_img)

                    if score > best_score:
                        best_state = c_s_img, c_t_img
                        best_score = score

                s_img, t_img = best_state
                n += 1

            if best_score > o_best_score:
                s_img, t_img = best_state

        return best_state
