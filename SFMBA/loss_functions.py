import torch
import spatial_transformer as stn
import utils


def compute_derivative(input_tensor):
    grad_x = input_tensor - torch.roll(input_tensor, shifts=1, dims=3)
    grad_x[:, :, :, 0] = 0
    grad_y = input_tensor - torch.roll(input_tensor, shifts=1, dims=2)
    grad_y[:, :, 0, :] = 0
    return grad_x, grad_y


def compute_exposure_mask(img):
    r, g, b = torch.chunk(img, chunks=3, dim=1)
    r_mask = ((r + 1e-5) < torch.max(r)).float()
    g_mask = ((g + 1e-5) < torch.max(g)).float()
    b_mask = ((b + 1e-5) < torch.max(b)).float()
    exposure_mask = r_mask * g_mask * b_mask
    return exposure_mask


def compute_photometric_error(x, y):
    photo_err = 0.15 * (x - y).abs() + 0.85 * ((1 - utils.compute_ssim(x, y)) / 2)
    return photo_err


def compute_pairwise_loss(tgt_img, src_img, tgt_depth_scaled, tgt_uncertainty_scaled,
                          pose, affine_param, intrinsics, intrinsics_inverse):
    exposure_mask = compute_exposure_mask(tgt_img)
    tgt_img_affine = (affine_param[0] * tgt_img + affine_param[1]) * exposure_mask + tgt_img * (1 - exposure_mask)

    valid_mask, projected_img = stn.warp_image(
        src_img, tgt_depth_scaled, pose, intrinsics, intrinsics_inverse
    )
    auto_mask = ((tgt_img_affine - projected_img).abs() < (tgt_img_affine - src_img).abs()).float()
    masking = valid_mask * auto_mask
    photometric_error = compute_photometric_error(tgt_img_affine, projected_img)
    photometric_error_laplacian = (photometric_error / tgt_uncertainty_scaled) + torch.log(tgt_uncertainty_scaled)

    return photometric_error_laplacian, masking


def mean_on_mask(diff, valid_mask):
    mask = valid_mask.expand_as(diff)
    num_valid = mask.sum()
    if num_valid > 2500:
        mean_value = (diff * mask).sum() / num_valid
    else:
        mean_value = torch.tensor(0).float().to(diff).detach()
    return mean_value


def compute_loss(tgt_img, src_imgs, tgt_depth, tgt_uncertainty,
                 poses, affine_params, intrinsics, intrinsics_inverse):
    b, _, h, w = tgt_img.shape
    photometric_loss = 0
    num_of_samples = len(src_imgs)
    num_of_scales = len(tgt_depth)

    for s in range(num_of_scales):
        total_photometric_error = []
        total_masking = torch.ones(b, 1, h, w).type_as(tgt_img)
        for (src_img, pose, affine_param) in zip(src_imgs, poses, affine_params):
            photometric_error, masking = compute_pairwise_loss(
                tgt_img, src_img, tgt_depth[s], tgt_uncertainty[s], pose, affine_param, intrinsics, intrinsics_inverse
            )
            total_photometric_error.append(photometric_error)
            total_masking = total_masking * masking

        total_photometric_error = torch.stack(total_photometric_error, dim=1).min(dim=1).values
        photometric_loss += mean_on_mask(total_photometric_error, total_masking)

    photometric_loss /= (num_of_samples * num_of_scales)

    return photometric_loss


def compute_smooth_loss(depths, imgs):
    def get_smooth_loss(depth, img):
        depth_grad_x, depth_grad_y = compute_derivative(depth)
        img_grad_x, img_grad_y = compute_derivative(img)

        return (torch.mean(torch.exp(-img_grad_x.abs().mean(dim=1, keepdim=True)) * depth_grad_x.abs())
                + torch.mean(torch.exp(-img_grad_y.abs().mean(dim=1, keepdim=True)) * depth_grad_y.abs()))

    loss = 0
    num = len(imgs)
    for depth, img in zip(depths, imgs):
        scales = len(depth)
        scaled_loss = 0
        for s in range(scales):
            scaled_loss += get_smooth_loss(depth[s], img) / (2 ** s)
        loss += scaled_loss / scales

    loss /= num

    return loss


def compute_regularization_loss(affine_params):
    loss = 0
    num = len(affine_params)
    for affine_param in affine_params:
        loss += torch.mean((affine_param[0] - 1.0) ** 2 + affine_param[1] ** 2)
    loss /= num

    return loss


def compute_validation_loss(tgt_img, src_imgs, tgt_depth, poses, affine_params, intrinsics, intrinsics_inverse):
    photometric_loss = 0
    num_of_samples = len(src_imgs)

    for (src_img, pose, affine_param) in zip(src_imgs, poses, affine_params):
        exposure_mask = compute_exposure_mask(tgt_img)
        tgt_img_affine = (affine_param[0] * tgt_img + affine_param[1]) * exposure_mask + tgt_img * (1 - exposure_mask)

        valid_mask, projected_img = stn.warp_image(
            src_img, tgt_depth, pose, intrinsics, intrinsics_inverse
        )
        photometric_loss += mean_on_mask(compute_photometric_error(tgt_img_affine, projected_img), valid_mask)

    photometric_loss /= num_of_samples

    return photometric_loss
