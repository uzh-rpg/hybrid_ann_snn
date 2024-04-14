from typing import Optional
import torch
import pdb


class StackedHistogram:
    def __init__(self, bins: int, height: int, width: int, count_cutoff: Optional[int]=None):
        assert bins >= 1
        self.bins = bins
        assert height >= 1
        self.height = height
        assert width >= 1
        self.width = width
        self.count_cutoff = count_cutoff
        if self.count_cutoff is not None:
            assert self.count_cutoff >= 1
        self.channels = 2

    @staticmethod
    def _is_int_tensor(tensor: torch.Tensor) -> bool:
        return not torch.is_floating_point(tensor) and not torch.is_complex(tensor)

    def construct(self, x: torch.Tensor, y: torch.Tensor, pol: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        # TODO(magehrig): could weight the histogram based on time
        device = x.device
        assert y.device == pol.device == time.device == device
        assert len(x) == len(y) == len(pol) == len(time)
        # assert len(pol) <= 7500
        assert self._is_int_tensor(x)
        assert self._is_int_tensor(y)
        assert self._is_int_tensor(pol)
        assert self._is_int_tensor(time)

        bn, ch, ht, wd = self.bins, self.channels, self.height, self.width

        # NOTE: assume sorted time
        t0_int = time[0]
        t1_int = time[-1]
        if t1_int < t0_int:
            pdb.set_trace()
        t_norm = time - t0_int
        t_norm = t_norm / (t1_int - t0_int)
        t_norm = t_norm * bn
        t_idx = t_norm.floor()
        t_idx = torch.clamp(t_idx, max=bn-1)

        indices = x.long() +\
                  wd * y.long() +\
                  ht * wd * t_idx.long() +\
                  bn * ht * wd * pol.long()
        values = torch.ones_like(indices, dtype=torch.float32, device=device)

        with torch.no_grad():
            repr = torch.zeros((self.channels, self.bins, self.height, self.width), dtype=torch.float32, device=device)
            repr.put_(indices, values, accumulate=True)

        if self.count_cutoff is not None:
            torch.clamp(repr, max=self.count_cutoff)

        repr = torch.transpose(repr, dim0=0, dim1=1)

        return repr

class VoxelGrid():
    def __init__(self, channels: int, height: int, width: int, normalize: bool):
        self.voxel_grid = torch.zeros((channels, height, width), dtype=torch.float, requires_grad=False)
        self.nb_channels = channels
        self.normalize = normalize

    def construct(self, x: torch.Tensor, y: torch.Tensor, pol: torch.Tensor, time: torch.Tensor):
        assert x.shape == y.shape == pol.shape == time.shape
        assert x.ndim == 1

        C, H, W = self.voxel_grid.shape
        with torch.no_grad():
            # self.voxel_grid = self.voxel_grid.to(pol.device)
            voxel_grid = self.voxel_grid.clone()

            t_norm = time
            if t_norm.size == 0:
                pdb.set_trace()
            t_norm = (C - 1) * (t_norm - t_norm[0]) / (t_norm[-1] - t_norm[0])

            x0 = x.int()
            y0 = y.int()
            t0 = t_norm.int()

            value = 2 * pol - 1

            for tlim in [t0, t0 + 1]:
                mask = (x0 < W) & (x0 >= 0) & (y0 < H) & (y0 >= 0) & (tlim >= 0) & (
                            tlim < self.nb_channels)
                interp_weights = value * (1 - (x0 - x).abs()) * (1 - (y0 - y).abs()) * (
                            1 - (tlim - t_norm).abs())

                index = H * W * tlim.long() + \
                        W * y0.long() + \
                        x0.long()

                voxel_grid.put_(index[mask], interp_weights[mask], accumulate=True)

            # isn't this wrong? standardizing with the given voxel's mean and std. Shouldn't it be done
            # wrt the mean and average std? 
            if self.normalize:
                mask = torch.nonzero(voxel_grid, as_tuple=True)
                if mask[0].size()[0] > 0:
                    mean = voxel_grid[mask].mean()
                    std = voxel_grid[mask].std()
                    if std > 0:
                        voxel_grid[mask] = (voxel_grid[mask] - mean) / std
                    else:
                        voxel_grid[mask] = voxel_grid[mask] - mean
            voxel_grid = torch.unsqueeze(voxel_grid, dim=1)
        return voxel_grid