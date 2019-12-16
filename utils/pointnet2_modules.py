import torch
import torch.nn as nn
import torch.nn.functional as F

import pointnet2_utils
import pytorch_utils as pt_utils
from typing import List
import numpy as np

class _PointnetSAModuleBase(nn.Module):

    def __init__(self):
        super().__init__()
        self.npoint = None
        self.groupers = None
        self.mlps = None
        self.pool = False

    def forward(self, xyz: torch.Tensor,
                features: torch.Tensor = None) -> (torch.Tensor, torch.Tensor):
        r"""
        Parameters
        ----------
        xyz : torch.Tensor
            (B, N, 3) tensor of the xyz coordinates of the points
        features : torch.Tensor
            (B, N, C) tensor of the descriptors of the the points

        Returns
        -------
        new_xyz : torch.Tensor
            (B, npoint, 3) tensor of the new points' xyz
        new_features : torch.Tensor
            (B, npoint, \sum_k(mlps[k][-1])) tensor of the new_points descriptors
        """

        all_features = 0
        xyz_flipped = xyz.transpose(1, 2).contiguous()
        
        if self.npoint is not None:
            fps_idx = pointnet2_utils.furthest_point_sample(xyz, self.npoint) \
                      if self.pool else torch.from_numpy(np.arange(xyz.size(1))).int().cuda().repeat(xyz.size(0), 1)
            new_xyz = pointnet2_utils.gather_operation(xyz_flipped, fps_idx).transpose(1, 2).contiguous()
        else:
            new_xyz = None
        
        for i in range(len(self.groupers)):
            new_features = self.groupers[i](xyz, new_xyz, features)  # (B, C, npoint, nsample)
            if not self.pool and self.npoint is not None:
                new_features = [new_features, features]
            new_features = self.mlps[i](new_features)   # (B, mlp[-1], npoint)
            all_features += new_features
        
        return new_xyz, all_features


class PointnetSAModuleMSG(_PointnetSAModuleBase):
    r"""Pointnet set abstrction layer with multiscale grouping

    Parameters
    ----------
    npoint : int
        Number of points
    radii : list of float32
        list of radii to group with
    nsamples : list of int32
        Number of samples in each ball query
    mlps : list of list of int32
        Spec of the pointnet before the global max_pool for each scale
    bn : bool
        Use batchnorm
    """

    def __init__(
            self,
            *,
            npoint: int,
            radii: List[float],
            nsamples: List[int],
            mlps: List[List[int]],
            group_number = 1,
            use_xyz: bool = True,
            pool: bool = False,
            before_pool: bool = False,
            after_pool: bool = False,
            bias = True,
            init = nn.init.kaiming_normal
    ):
        super().__init__()

        assert len(radii) == len(nsamples) == len(mlps)
        self.pool = pool
        self.npoint = npoint
        self.groupers = nn.ModuleList()
        self.mlps = nn.ModuleList()
        
        if pool:
            C_in = (mlps[0][0] + 3) if use_xyz else mlps[0][0]
            C_out = mlps[0][1]
            pconv = nn.Conv2d(in_channels = C_in, out_channels = C_out, kernel_size = (1, 1), 
                                       stride = (1, 1), bias = bias)
            init(pconv.weight)
            if bias:
                nn.init.constant(pconv.bias, 0)
            convs = [pconv]
        
        for i in range(len(radii)):
            radius = radii[i]
            nsample = nsamples[i]
            self.groupers.append(
                pointnet2_utils.QueryAndGroup(radius, nsample, use_xyz=use_xyz)
                if npoint is not None else pointnet2_utils.GroupAll(use_xyz)
            )
            mlp_spec = mlps[i]
            if use_xyz:
                mlp_spec[0] += 3
            if npoint is None:
                self.mlps.append(pt_utils.GloAvgConv(C_in = mlp_spec[0], C_out = mlp_spec[1]))
            elif pool:
                self.mlps.append(pt_utils.PointConv(C_in = mlp_spec[0], C_out = mlp_spec[1], convs = convs))
            else:
                self.mlps.append(pt_utils.EnhancedPointConv(C_in = mlp_spec[0], C_out = mlp_spec[1], group_number = group_number, before_pool = before_pool, after_pool = after_pool))


class PointnetSAModule(PointnetSAModuleMSG):
    r"""Pointnet set abstrction layer

    Parameters
    ----------
    npoint : int
        Number of features
    radius : float
        Radius of ball
    nsample : int
        Number of samples in the ball query
    mlp : list
        Spec of the pointnet before the global max_pool
    bn : bool
        Use batchnorm
    """

    def __init__(
            self,
            *,
            mlp: List[int],
            npoint: int = None,
            radius: float = None,
            nsample: int = None,
            use_xyz: bool = True
    ):
        super().__init__(
            mlps=[mlp],
            npoint=npoint,
            radii=[radius],
            nsamples=[nsample],
            use_xyz=use_xyz
        )


class PointnetFPModule(nn.Module):
    r"""Propigates the features of one set to another

    Parameters
    ----------
    mlp : list
        Pointnet module parameters
    bn : bool
        Use batchnorm
    """

    def __init__(self, *, mlp: List[int], bn: bool = True):
        super().__init__()
        self.mlp = pt_utils.SharedMLP(mlp, bn=bn)

    def forward(
            self, unknown: torch.Tensor, known: torch.Tensor,
            unknow_feats: torch.Tensor, known_feats: torch.Tensor
    ) -> torch.Tensor:
        r"""
        Parameters
        ----------
        unknown : torch.Tensor
            (B, n, 3) tensor of the xyz positions of the unknown features
        known : torch.Tensor
            (B, m, 3) tensor of the xyz positions of the known features
        unknow_feats : torch.Tensor
            (B, C1, n) tensor of the features to be propigated to
        known_feats : torch.Tensor
            (B, C2, m) tensor of features to be propigated

        Returns
        -------
        new_features : torch.Tensor
            (B, mlp[-1], n) tensor of the features of the unknown features
        """

        dist, idx = pointnet2_utils.three_nn(unknown, known)
        dist_recip = 1.0 / (dist + 1e-8)
        norm = torch.sum(dist_recip, dim=2, keepdim=True)
        weight = dist_recip / norm

        interpolated_feats = pointnet2_utils.three_interpolate(
            known_feats, idx, weight
        )
        if unknow_feats is not None:
            new_features = torch.cat([interpolated_feats, unknow_feats],
                                     dim=1)  #(B, C2 + C1, n)
        else:
            new_features = interpolated_feats
        
        new_features = new_features.unsqueeze(-1)
        new_features = self.mlp(new_features)

        return new_features.squeeze(-1)