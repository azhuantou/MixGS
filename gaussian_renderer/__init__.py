#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel, GatheredGaussian
from utils.sh_utils import eval_sh
from utils.large_utils import in_frustum


def prefilter_voxel(viewpoint_camera, pc: GaussianModel, pipe, bg_color: torch.Tensor, scaling_modifier=1.0,
                    override_color=None):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """
    from diff_gaussian_rasterization_filter import GaussianRasterizationSettings, GaussianRasterizer
    tanfovx = math.tan(viewpoint_camera["FoVx"] * 0.5)
    tanfovy = math.tan(viewpoint_camera["FoVy"] * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera["image_height"]),
        image_width=int(viewpoint_camera["image_width"]),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera["world_view_transform"],
        projmatrix=viewpoint_camera["full_proj_transform"],
        sh_degree=1,
        campos=viewpoint_camera["camera_center"],
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    radii_pure = rasterizer.visible_filter(means3D=means3D,
                                           scales=scales[:, :3],
                                           rotations=rotations,
                                           cov3D_precomp=cov3D_precomp)

    return radii_pure > 0


def render_mix(viewpoint_camera, pc: GaussianModel, pipe, bg_color: torch.Tensor, vis_mask, decoded_data, scaling_modifier=1.0):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """
    ori_xyz = pc.get_xyz[vis_mask].detach()
    ori_rot = pc.get_rotation[vis_mask].detach()

    d_scaling = decoded_data["d_scaling"].to(torch.float32)
    d_rotation = decoded_data["d_rotation"].to(torch.float32)
    d_sh = decoded_data["d_color"].to(torch.float32)
    d_opacity = decoded_data["d_opacity"].to(torch.float32)

    num = len(d_scaling)
    means3D = ori_xyz + pc.get_offset[vis_mask].reshape(num, -1)

    # color pre compute
    pc_features = pc.get_features[vis_mask].transpose(1, 2)
    shs_view = pc_features.view(pc_features.shape[0], -1, (pc.max_sh_degree + 1) ** 2)
    dir_pp = (pc.get_xyz[vis_mask] - viewpoint_camera["camera_center"].repeat(pc_features.shape[0], 1))
    dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
    sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
    colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)

    colors_precomp = torch.cat([torch.sigmoid(d_sh), colors_precomp], dim=0)
    opacity = d_opacity
    rotations = pc.rotation_activation(ori_rot + d_rotation)
    res_scales = torch.clamp_min(d_scaling, pipe.scale_min)   # 0.002  0.0005

    ori_means3D = pc.get_xyz[vis_mask]
    ori_opacity = pc.get_opacity[vis_mask]
    ori_scales = pc.get_scaling[vis_mask]
    ori_rotations = pc.get_rotation[vis_mask]

    means3D = torch.cat([means3D, ori_means3D], dim=0)
    opacity = torch.cat([opacity, ori_opacity], dim=0)
    scales = torch.cat([res_scales, ori_scales], dim=0)
    rotations = torch.cat([rotations, ori_rotations], dim=0)

    screenspace_points = torch.zeros_like(means3D, dtype=means3D.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    tanfovx = math.tan(viewpoint_camera["FoVx"] * 0.5)
    tanfovy = math.tan(viewpoint_camera["FoVy"] * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera["image_height"]),
        image_width=int(viewpoint_camera["image_width"]),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera["world_view_transform"],
        projmatrix=viewpoint_camera["full_proj_transform"],
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera["camera_center"],
        prefiltered=False,
        debug=pipe.debug,
        antialiasing=False,
    )
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means2D = screenspace_points

    cov3D_precomp = None

    rendered_image, radii, depth_image = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=None, #shs,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter": radii > 0,
            "radii": radii,
            "depth": depth_image,
            "scale": res_scales,
            }


