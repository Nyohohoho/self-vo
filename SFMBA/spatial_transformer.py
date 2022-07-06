from __future__ import division
import torch
import torch.nn.functional as F

pixel_coords = None


def set_id_grid(depth):
    global pixel_coords
    b, _, h, w = depth.shape
    x_range = torch.arange(0, w).view(1, 1, w).expand(1, h, w).type_as(depth)  # [1, H, W]
    y_range = torch.arange(0, h).view(1, h, 1).expand(1, h, w).type_as(depth)  # [1, H, W]
    ones = torch.ones(1, h, w).type_as(depth)  # [1, H, W]

    pixel_coords = torch.stack([x_range, y_range, ones], dim=1)  # [1, 3, H, W]


def pixel2cam(depth, intrinsics_inv):
    global pixel_coords
    """Transform coordinates in the pixel frame to the camera frame.
    Args:
        depth: depth maps -- [B, 1, H, W]
        intrinsics_inv: intrinsics_inv matrix for each element of batch -- [B, 3, 3]
    Returns:
        array of (u,v,1) cam coordinates -- [B, 3, H, W]
    """
    b, _, h, w = depth.size()
    if (pixel_coords is None) or pixel_coords.shape[2] < h:
        set_id_grid(depth)
    current_pixel_coords = pixel_coords[:, :, :h, :w].expand(b, 3, h, w).reshape(b, 3, -1)  # [B, 3, H*W]
    cam_coords = (intrinsics_inv @ current_pixel_coords).reshape(b, 3, h, w)
    return cam_coords * depth


def cam2pixel(cam_coords, proj_c2p_rot, proj_c2p_tr):
    """Transform coordinates in the camera frame to the pixel frame.
    Args:
        cam_coords: pixel coordinates defined in the first camera coordinates system -- [B, 4, H, W]
        proj_c2p_rot: rotation matrix of cameras -- [B, 3, 4]
        proj_c2p_tr: translation vectors of cameras -- [B, 3, 1]
    Returns:
        array of [-1,1] coordinates -- [B, 2, H, W]
    """
    b, _, h, w = cam_coords.size()
    cam_coords_flat = cam_coords.reshape(b, 3, -1)  # [B, 3, H*W]
    if proj_c2p_rot is not None:
        pcoords = proj_c2p_rot @ cam_coords_flat
    else:
        pcoords = cam_coords_flat

    if proj_c2p_tr is not None:
        pcoords = pcoords + proj_c2p_tr  # [B, 3, H*W]

    x_3d = pcoords[:, 0]
    y_3d = pcoords[:, 1]
    z_3d = pcoords[:, 2].clamp(min=1e-3)

    # Normalized, -1 if on extreme left, 1 if on extreme right (x = w-1) [B, H*W]
    x_2d = 2 * (x_3d / z_3d) / (w - 1) - 1
    y_2d = 2 * (y_3d / z_3d) / (h - 1) - 1  # Idem [B, H*W]

    x_2d_mask = ((x_2d > 1) + (x_2d < -1)).detach()
    x_2d[x_2d_mask] = 2
    y_2d_mask = ((y_2d > 1) + (y_2d < -1)).detach()
    y_2d[y_2d_mask] = 2

    warped_pixel_coords = torch.stack([x_2d, y_2d], dim=2)  # [B, H*W, 2]
    return warped_pixel_coords.reshape(b, h, w, 2), z_3d.reshape(b, 1, h, w)


def warp_image(src_img, tgt_depth, pose_mat, intrinsics, intrinsics_inverse):
    """
    Inverse warp a source image to the target image plane.
    Args:
        src_img: the source image (where to sample pixels) -- [B, 3, H, W]
        tgt_depth: depth map of the target image -- [B, 1, H, W]
        pose_mat: 6DoF pose parameters from target to source -- [B, 3, 4]
        intrinsics: camera intrinsic matrix -- [B, 3, 3]
        intrinsics_inverse
    Returns:
        projected_img: Source image warped to the target image plane
        valid_mask: Float array indicating point validity
        projected_depth: sampled depth from source image
        computed_depth: computed depth of source image using the target depth
    """

    cam_coords = pixel2cam(tgt_depth, intrinsics_inverse)  # [B,3,H,W]

    # Get projection matrix for tgt camera frame to source pixel frame
    proj_cam_to_src_pixel = intrinsics @ pose_mat  # [B, 3, 4]

    rot, tr = proj_cam_to_src_pixel[:, :, :3], proj_cam_to_src_pixel[:, :, -1:]
    src_pixel_coords, _ = cam2pixel(cam_coords, rot, tr)  # [B,H,W,2]

    valid_points = src_pixel_coords.abs().max(dim=-1)[0] <= 1
    valid_mask = valid_points.unsqueeze(dim=1).float()

    projected_img = F.grid_sample(src_img, src_pixel_coords, padding_mode='border', align_corners=False)

    return valid_mask, projected_img


def warp_depth(src_depth, tgt_depth, pose_mat, intrinsics, intrinsics_inverse):
    """
    Inverse warp a source depth to the target image plane.
    Args:
        src_depth: the source image (where to sample pixels) -- [B, 1, H, W]
        tgt_depth: depth map of the target image -- [B, 1, H, W]
        pose_mat: 6DoF pose parameters from target to source -- [B, 3, 4]
        intrinsics: camera intrinsic matrix -- [B, 3, 3]
        intrinsics_inverse
    Returns:
        projected_img: Source image warped to the target image plane
        valid_mask: Float array indicating point validity
        projected_depth: sampled depth from source image
        computed_depth: computed depth of source image using the target depth
    """

    cam_coords = pixel2cam(tgt_depth, intrinsics_inverse)  # [B,3,H,W]

    # Get projection matrix for tgt camera frame to source pixel frame
    proj_cam_to_src_pixel = intrinsics @ pose_mat  # [B, 3, 4]

    rot, tr = proj_cam_to_src_pixel[:, :, :3], proj_cam_to_src_pixel[:, :, -1:]
    src_pixel_coords, computed_depth = cam2pixel(cam_coords, rot, tr)  # [B,H,W,2]

    valid_points = src_pixel_coords.abs().max(dim=-1)[0] <= 1
    valid_mask = valid_points.unsqueeze(dim=1).float()

    projected_depth = F.grid_sample(src_depth, src_pixel_coords, padding_mode='zeros', align_corners=False)

    return valid_mask, computed_depth, projected_depth
