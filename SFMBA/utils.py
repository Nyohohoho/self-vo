import torch
import torch.nn.functional as F


def pose_vec2mat(pose_vec):
    """
    Convert 6DoF parameters to transformation matrix.
    Args:s
        vec: 6DoF parameters in the order of tx, ty, tz, rx, ry, rz -- [B, 6]
    Returns:
        A transformation matrix -- [B, 3, 4]
    """

    def euler2mat(angle):
        """Convert euler angles to rotation matrix.
         Reference: https://github.com/pulkitag/pycaffe-utils/blob/master/rot_utils.py#L174
        Args:
            angle: rotation angle along 3 axis (in radians) -- size = [B, 3]
        Returns:
            Rotation matrix corresponding to the euler angles -- size = [B, 3, 3]
        """
        batch_size = angle.shape[0]
        x, y, z = angle[:, 0], angle[:, 1], angle[:, 2]

        cosz = torch.cos(z)
        sinz = torch.sin(z)

        zeros = z.detach() * 0
        ones = zeros.detach() + 1
        zmat = torch.stack([cosz, -sinz, zeros,
                            sinz, cosz, zeros,
                            zeros, zeros, ones], dim=1).reshape(batch_size, 3, 3)

        cosy = torch.cos(y)
        siny = torch.sin(y)

        ymat = torch.stack([cosy, zeros, siny,
                            zeros, ones, zeros,
                            -siny, zeros, cosy], dim=1).reshape(batch_size, 3, 3)

        cosx = torch.cos(x)
        sinx = torch.sin(x)

        xmat = torch.stack([ones, zeros, zeros,
                            zeros, cosx, -sinx,
                            zeros, sinx, cosx], dim=1).reshape(batch_size, 3, 3)

        rotMat = xmat @ ymat @ zmat
        return rotMat

    translation = pose_vec[:, :3]
    euler_angle = pose_vec[:, 3:]

    translation = translation.unsqueeze(-1)  # [B, 3, 1]
    rotation_matrix = euler2mat(euler_angle)  # [B, 3, 3]
    transform_matrix = torch.cat([rotation_matrix, translation], dim=2)  # [B, 3, 4]

    return transform_matrix


def compute_ssim(x, y):
    x = F.pad(x, pad=(1, 1, 1, 1), mode='reflect')
    y = F.pad(y, pad=(1, 1, 1, 1), mode='reflect')

    mu_x = F.avg_pool2d(x, kernel_size=3, stride=1)
    mu_y = F.avg_pool2d(y, kernel_size=3, stride=1)

    sigma_x = F.avg_pool2d(x ** 2, kernel_size=3, stride=1) - mu_x ** 2
    sigma_y = F.avg_pool2d(y ** 2, kernel_size=3, stride=1) - mu_y ** 2
    sigma_xy = F.avg_pool2d(x * y, kernel_size=3, stride=1) - mu_x * mu_y

    ssim_n = (2 * mu_x * mu_y + (0.01 ** 2)) * (2 * sigma_xy + (0.03 ** 2))
    ssim_d = (mu_x ** 2 + mu_y ** 2 + (0.01 ** 2)) * (sigma_x + sigma_y + (0.03 ** 2))
    ssim = torch.clamp(ssim_n / ssim_d, -1, 1)

    return ssim
