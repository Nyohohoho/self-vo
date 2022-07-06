import argparse
import time
import csv
import datetime
from path import Path

import torch
import torch.backends.cudnn as cudnn
import torch.backends.cuda
import torch.optim
import torch.utils.data

import models
import utils
import custom_transforms
from datasets.kitti_folders import KittiRaw, KittiOdometry
from datasets.tum_folders import TUMSequence
import loss_functions
from logger import TermLogger, AverageMeter
from tensorboardX import SummaryWriter


parser = argparse.ArgumentParser(description='SFMBA',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('data', metavar='DIR', help='path to dataset')
parser.add_argument('--folder-type', type=str, choices=['sequence', 'pair'],
                    default='sequence', help='the dataset dype to train')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers')
parser.add_argument('--epochs', default=20, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--img-height', default=192, type=int, metavar='N', help='image height')
parser.add_argument('--img-width', default=640, type=int, metavar='N', help='image width')
parser.add_argument('-b', '--batch-size', default=4, type=int, metavar='N', help='mini-batch size')
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum for sgd, alpha parameter for adam')
parser.add_argument('--beta', default=0.999, type=float, metavar='M', help='beta parameters for adam')
parser.add_argument('--weight-decay', '--wd', default=0, type=float, metavar='W', help='weight decay')
parser.add_argument('--print-freq', default=1, type=int, metavar='N', help='print frequency')
parser.add_argument('--seed', default=0, type=int, help='seed for random functions, and network initialization')
parser.add_argument('--log-summary', default='progress_log_summary.csv', metavar='PATH',
                    help='csv where to save per-epoch train and valid stats')
parser.add_argument('--log-full', default='progress_log_full.csv', metavar='PATH',
                    help='csv where to save per-gradient descent train stats')
parser.add_argument('--wp', type=float, metavar='W', help='photo-loss-weight', default=1.0)
parser.add_argument('--ws', type=float, metavar='W', help='smooth-loss-weight', default=1e-3)
parser.add_argument('--wr', type=float, metavar='W', help='regularization-loss-weight', default=1e-3)
parser.add_argument('--dataset', type=str, choices=['kitti', 'nyu'], default='kitti', help='the dataset to train')
parser.add_argument('--pretrained-net', dest='pretrained_net',
                    default=None, metavar='PATH', help='path to pre-trained CNN')
parser.add_argument('--name', dest='name', type=str, required=True,
                    help='name of the experiment, checkpoints are stored in checpoints/name')


best_error = -1
n_iter = 0
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
torch.autograd.set_detect_anomaly(True)


def main():
    global best_error, n_iter, device
    args = parser.parse_args()

    timestamp = datetime.datetime.now().strftime("%m-%d-%H:%M")
    save_path = Path(args.name)
    args.save_path = 'checkpoints' / save_path / timestamp
    print('=> will save everything to {}'.format(args.save_path))
    args.save_path.makedirs_p()

    cudnn.deterministic = True
    cudnn.benchmark = True

    training_writer = SummaryWriter(args.save_path)
    # output_writers = []

    # Data loading code
    normalize = custom_transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    train_transform = custom_transforms.Compose([
        custom_transforms.Resize(height=args.img_height, width=args.img_width),
        custom_transforms.RandomHorizontalFlip(),
        custom_transforms.RandomScaleCrop(),
        custom_transforms.ArrayToTensor(),
        custom_transforms.RandomColorJitter(),
        normalize
    ])

    valid_transform = custom_transforms.Compose([
        custom_transforms.Resize(height=args.img_height, width=args.img_width),
        custom_transforms.ArrayToTensor(),
        normalize
    ])

    print("=> fetching scenes in '{}'".format(args.data))
    train_set = TUMSequence(
        args.data,
        train=True,
        transform=train_transform
    )

    val_set = TUMSequence(
        args.data,
        train=False,
        transform=valid_transform
    )

    print('{} samples found'.format(len(train_set)))
    print('{} samples found'.format(len(val_set)))
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=32, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # create model
    print("=> creating model")
    depth_net = models.DepthResNet().to(device)
    pose_net = models.PoseResNet().to(device)

    # load parameters
    """
    if args.pretrained_net:
        print("=> using pre-trained weights for CNN")
        weights = torch.load(args.pretrained_net, map_location=device)
        two_view_net.load_state_dict(weights['state_dict'], strict=False)
    """

    """
    depth_net = torch.nn.DataParallel(depth_net)
    pose_net = torch.nn.DataParallel(pose_net)
    """

    print('=> setting adam solver')
    optim_params = [
        {'params': depth_net.parameters(), 'lr': args.lr},
        {'params': pose_net.parameters(), 'lr': args.lr}
    ]
    optimizer = torch.optim.Adam(optim_params,
                                 betas=(args.momentum, args.beta),
                                 weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15], gamma=0.1)

    with open(args.save_path/args.log_summary, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        writer.writerow(['train_loss', 'validation_loss'])

    with open(args.save_path/args.log_full, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        writer.writerow([
            'photometric_loss', 'smooth_loss', 'regularization_loss', 'loss'
        ])

    logger = TermLogger(
        n_epochs=args.epochs, train_size=len(train_loader), valid_size=len(val_loader)
    )
    logger.epoch_bar.start()

    for epoch in range(args.epochs):
        logger.epoch_bar.update(epoch)

        # train for one epoch
        logger.reset_train_bar()
        train_loss = train(args, train_loader, depth_net, pose_net, optimizer, logger, training_writer)
        logger.train_writer.write(' * Avg Photo Loss : {:.4f}'.format(train_loss))

        # evaluate on validation set
        logger.reset_valid_bar()
        errors, error_names = validate_without_gt(args, val_loader, depth_net, pose_net, logger)

        error_string = ', '.join('{} : {:.4f}'.format(name, error) for name, error in zip(error_names, errors))
        logger.valid_writer.write(' * Avg {}'.format(error_string))

        for error, name in zip(errors, error_names):
            training_writer.add_scalar(name, error, epoch)

        decisive_error = errors[0]
        if best_error < 0:
            best_error = decisive_error

        # remember lowest error and save checkpoint
        is_best = decisive_error < best_error
        best_error = min(best_error, decisive_error)
        save_checkpoint(
            args.save_path, {
                'epoch': epoch + 1,
                'state_dict': depth_net.state_dict()
            }, {
                'epoch': epoch + 1,
                'state_dict': pose_net.state_dict()
            }, {
                'epoch': epoch + 1,
                'state_dict': optimizer.state_dict()
            },
            is_best)

        with open(args.save_path/args.log_summary, 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter='\t')
            writer.writerow([train_loss, decisive_error])

        scheduler.step()

    logger.epoch_bar.finish()


def train(args, train_loader, depth_net, pose_net, optimizer, logger, training_writer):
    global n_iter, device
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter(i=4, precision=4)

    wp, ws, wr = args.wp, args.ws, args.wr

    # switch to train mode
    depth_net.train()
    pose_net.train()

    end = time.time()
    logger.train_bar.update(0)

    for i, (tgt_img, src_imgs, intrinsics, intrinsics_inverse) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        tgt_img = tgt_img.to(device)
        src_imgs = [src_img.to(device) for src_img in src_imgs]
        intrinsics = intrinsics.to(device)
        intrinsics_inverse = intrinsics_inverse.to(device)

        tgt_depth_and_uncertainty = depth_net(tgt_img)
        tgt_depth = [element[0] for element in tgt_depth_and_uncertainty]
        tgt_uncertainty = [element[1] for element in tgt_depth_and_uncertainty]

        poses_and_affine_params = [pose_net(tgt_img, src_img) for src_img in src_imgs]
        poses = [utils.pose_vec2mat(element[0]) for element in poses_and_affine_params]
        affine_params = [element[1] for element in poses_and_affine_params]

        photometric_loss = loss_functions.compute_loss(
            tgt_img, src_imgs, tgt_depth, tgt_uncertainty, poses, affine_params, intrinsics, intrinsics_inverse
        )
        smooth_loss = loss_functions.compute_smooth_loss([tgt_depth], [tgt_img])
        regularization_loss = loss_functions.compute_regularization_loss(affine_params)

        loss = wp * photometric_loss + ws * smooth_loss + wr * regularization_loss

        # record loss and EPE
        losses.update([photometric_loss.item(), smooth_loss.item(), regularization_loss.item(), loss.item()],
                      args.batch_size)

        training_writer.add_scalar('photometric_loss', photometric_loss.item(), n_iter)
        training_writer.add_scalar('smooth_loss', smooth_loss.item(), n_iter)
        training_writer.add_scalar('regularization_loss', smooth_loss.item(), n_iter)
        training_writer.add_scalar('loss', loss.item(), n_iter)

        # compute gradient and do Adam step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        with open(args.save_path/args.log_full, 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter='\t')
            writer.writerow([photometric_loss.item(), smooth_loss.item(), loss.item()])
        logger.train_bar.update(i + 1)
        if i % args.print_freq == 0:
            logger.train_writer.write('Train: Time {} Data {} Loss {}'.format(batch_time, data_time, losses))

        n_iter += 1

    return losses.avg[0]


@torch.no_grad()
def validate_without_gt(args, val_loader, depth_net, pose_net, logger):
    global device
    batch_time = AverageMeter()
    losses = AverageMeter(i=1, precision=4)

    # switch to evaluate mode
    depth_net.eval()
    pose_net.eval()

    end = time.time()
    logger.valid_bar.update(0)
    for i, (tgt_img, src_imgs, intrinsics, intrinsics_inverse) in enumerate(val_loader):

        tgt_img = tgt_img.to(device)
        src_imgs = [src_img.to(device) for src_img in src_imgs]
        intrinsics = intrinsics.to(device)
        intrinsics_inverse = intrinsics_inverse.to(device)

        tgt_depth = depth_net(tgt_img)
        poses_and_affine_params = [pose_net(tgt_img, src_img) for src_img in src_imgs]
        poses = [utils.pose_vec2mat(element[0]) for element in poses_and_affine_params]
        affine_params = [element[1] for element in poses_and_affine_params]

        photometric_loss = loss_functions.compute_validation_loss(
            tgt_img, src_imgs, tgt_depth, poses, affine_params, intrinsics, intrinsics_inverse
        )

        losses.update([photometric_loss.item()])

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        logger.valid_bar.update(i + 1)
        if i % args.print_freq == 0:
            logger.valid_writer.write('valid: Time {} Loss {}'.format(batch_time, losses))

    logger.valid_bar.update(len(val_loader))
    return losses.avg, ['Photo']


def save_checkpoint(save_path, depth_state, pose_state, optimizer_state, is_best):
    file_prefixes = ['depth', 'pose', 'optimizer']
    states = [depth_state, pose_state, optimizer_state]
    for (prefix, state) in zip(file_prefixes, states):
        torch.save(state, save_path / '{}_ckpt.tar'.format(prefix))

    if is_best:
        for (prefix, state) in zip(file_prefixes, states):
            torch.save(state, save_path / '{}_best.tar'.format(prefix))


if __name__ == '__main__':
    main()
