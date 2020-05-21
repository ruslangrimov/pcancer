import math
import random

import torch
from torch import nn
import torch.nn.functional as F


class FeaturesMap(nn.Module):
    def __init__(self, use_dummy_feature, f_channels=512, max_height=70,
                 max_width=40):
        super().__init__()

        self.max_height = max_height
        self.max_width = max_width
        self.f_channels = f_channels
        self.use_dummy_feature = use_dummy_feature

        if self.use_dummy_feature:
            self.backend_feature =\
                nn.Parameter(torch.full((self.f_channels, self.max_height,
                                         self.max_width), 0))

    def forward(self, features, ys, xs, validation=None):
        if validation is None:
            validation = not self.training

        b_sz = features.shape[0]

        if self.use_dummy_feature:
            f_map = self.backend_feature.expand(self.f_channels,
                                                self.max_height,
                                                self.max_width)
        else:
            f_map = torch.full((self.f_channels, self.max_height,
                                self.max_width), 0,
                               dtype=features.dtype, device=features.device)

        res_f_maps = []
        for b in range(b_sz):
            real_mask = ys[b] > -1

            min_y, max_y = (ys[b, real_mask].min().item(),
                            ys[b, real_mask].max().item())
            min_x, max_x = (xs[b, real_mask].min().item(),
                            xs[b, real_mask].max().item())

            height = max_y - min_y + 1
            width = max_x - min_x + 1

            tmp_f_map = torch.full((self.f_channels, height, width), -1,
                                   dtype=features.dtype,
                                   device=features.device)

            tmp_f_map[:, ys[b, real_mask] - min_y, xs[b, real_mask] - min_x] =\
                features[b, :, real_mask]

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

        return torch.stack(res_f_maps)


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
            y_min, x_min = ys[b].min(), xs[b].min()
            y_max, x_max = ys[b].max(), xs[b].max()

            if not validation:
                y_rnd = random.randint(0, self.t_step)
                x_rnd = random.randint(0, self.t_step)
            else:
                y_rnd, x_rnd = 0, 0

            r_mask = ys[b] > -1

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
