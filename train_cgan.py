from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random
import warnings
# from tqdm import tqdm
from copy import deepcopy

import PIL.Image
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.utils.data.distributed
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import ToTensor

import src.utils.cfg as cfg
from src.data_loader import MitbihTrainDataLaoder
from src.models.tcgan.cgan_functions import train, save_samples, load_params, copy_params, cur_stages
from src.models.tcgan.tcgan_model import TCGANModel
from src.models.utils import Discriminator
from src.utils.adamw import AdamW
from src.utils.linear_lr_decay import LinearLrDecay
from src.utils.utils import set_log_dir, save_checkpoint, create_logger, gen_plot


def main():
    args = cfg.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.random_seed)
            torch.cuda.manual_seed_all(args.random_seed)
        else:
            torch.manual_seed(args.random_seed)

        np.random.seed(args.random_seed)
        random.seed(args.random_seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker_us(args.gpu, ngpus_per_node, args)


def main_worker_us(gpu, ngpus_per_node, args, start_epoch=0):
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

        # (change) Determine the device to be used
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu is not None else "cpu")

    # Initialize networks
    # import network
    gen_net = TCGANModel(seq_len=187, channels=1, num_classes=5, latent_dim=100, data_embed_dim=10,
                         label_embed_dim=10, depth=3, num_heads=5,
                         forward_drop_rate=0.5, attn_drop_rate=0.5)
    dis_net = Discriminator(in_channels=1, patch_size=1, data_emb_size=50, label_emb_size=10, seq_length=187, depth=3,
                            n_classes=5)

    # Move networks to the specified device
    gen_net.to(device)
    dis_net.to(device)

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
        gen_net = torch.nn.parallel.DistributedDataParallel(gen_net, device_ids=[args.gpu],
                                                            find_unused_parameters=True)
        dis_net = torch.nn.parallel.DistributedDataParallel(dis_net, device_ids=[args.gpu],
                                                            find_unused_parameters=True)
    else:
        if torch.cuda.is_available() and args.gpu is not None:
            gen_net = torch.nn.DataParallel(gen_net)
            dis_net = torch.nn.DataParallel(dis_net)

            # (unchanged code below)
    criterion = nn.BCELoss()
    optimizerG = torch.optim.Adam(gen_net.parameters(), lr=args.g_lr, betas=(args.beta1, 0.999))
    optimizerD = torch.optim.Adam(dis_net.parameters(), lr=args.g_lr, betas=(args.beta1, 0.999))

    # load dataset
    train_set = MitbihTrainDataLaoder()
    train_loader = data.DataLoader(train_set, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)

    experiment_name = 'first_experiment'
    args.path_helper = set_log_dir('logs', experiment_name)
    logger = create_logger(args.path_helper['log_path'])
    writer = SummaryWriter(args.path_helper['log_path'])

    writer_dict = {
        'writer': writer,
        'train_global_steps': start_epoch * len(train_loader),
        'valid_global_steps': start_epoch // args.val_freq,
    }

    for epoch in range(int(start_epoch), int(args.max_epoch)):
        train(gen_net, dis_net, gen_optimizer=optimizerG, dis_optimizer=optimizerD,
              train_loader=train_loader, epoch=epoch, writer_dict=writer_dict, args=args)

        if args.save_model and (epoch + 1) % args.save_frequency == 0:
            save_checkpoint(gen_net, dis_net, optimizerG, optimizerD, epoch, args)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    # import network
    gen_net = TCGANModel(seq_len=187, channels=1, num_classes=5, latent_dim=100, data_embed_dim=10,
                         label_embed_dim=10, depth=3, num_heads=5,
                         forward_drop_rate=0.5, attn_drop_rate=0.5)

    print(gen_net)
    dis_net = Discriminator(in_channels=1, patch_size=1, data_emb_size=50, label_emb_size=10, seq_length=187, depth=3,
                            n_classes=5)
    print(dis_net)

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
        gen_net = torch.nn.DataParallel(gen_net)
        dis_net = torch.nn.DataParallel(dis_net)

    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            gen_net.cuda(args.gpu)
            dis_net.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.dis_batch_size = int(args.dis_batch_size / ngpus_per_node)
            args.gen_batch_size = int(args.gen_batch_size / ngpus_per_node)
            args.batch_size = args.dis_batch_size

            args.num_workers = int((args.num_workers + ngpus_per_node - 1) / ngpus_per_node)
            gen_net = torch.nn.parallel.DistributedDataParallel(gen_net, device_ids=[args.gpu],
                                                                find_unused_parameters=True)
            dis_net = torch.nn.parallel.DistributedDataParallel(dis_net, device_ids=[args.gpu],
                                                                find_unused_parameters=True)
        else:
            gen_net.cuda()
            dis_net.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            gen_net = torch.nn.parallel.DistributedDataParallel(gen_net)
            dis_net = torch.nn.parallel.DistributedDataParallel(dis_net)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        gen_net.cuda(args.gpu)
        dis_net.cuda(args.gpu)
    else:
        gen_net = torch.nn.DataParallel(gen_net).cuda()
        dis_net = torch.nn.DataParallel(dis_net).cuda()
    print(dis_net) if args.rank == 0 else 0

    # set optimizer
    if args.optimizer == "adam":
        gen_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, gen_net.parameters()),
                                         args.g_lr, (args.beta1, args.beta2))
        dis_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, dis_net.parameters()),
                                         args.d_lr, (args.beta1, args.beta2))
    elif args.optimizer == "adamw":
        gen_optimizer = AdamW(filter(lambda p: p.requires_grad, gen_net.parameters()),
                              args.g_lr, weight_decay=args.wd)
        dis_optimizer = AdamW(filter(lambda p: p.requires_grad, dis_net.parameters()),
                              args.g_lr, weight_decay=args.wd)

    gen_scheduler = LinearLrDecay(gen_optimizer, args.g_lr, 0.0, 0, args.max_iter * args.n_critic)
    dis_scheduler = LinearLrDecay(dis_optimizer, args.d_lr, 0.0, 0, args.max_iter * args.n_critic)

    args.max_epoch = args.max_epoch * args.n_critic

    # load dataset
    train_set = MitbihTrainDataLaoder()
    train_loader = data.DataLoader(train_set, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)

    if args.max_iter:
        args.max_epoch = np.ceil(args.max_iter * args.n_critic / len(train_loader))

    # initial
    if torch.cuda.is_available():
        fixed_z = torch.cuda.FloatTensor(np.random.normal(0, 1, (100, args.latent_dim)))
    else:
        fixed_z = torch.FloatTensor(np.random.normal(0, 1, (100, args.latent_dim)))
    avg_gen_net = deepcopy(gen_net).cpu()
    gen_avg_param = copy_params(avg_gen_net)
    del avg_gen_net
    start_epoch = 0
    best_fid = 1e4

    # set writer
    writer = None
    if args.load_path:
        print(f'=> resuming from {args.load_path}')
        assert os.path.exists(args.load_path)
        checkpoint_file = os.path.join(args.load_path)
        assert os.path.exists(checkpoint_file)
        loc = 'cuda:{}'.format(args.gpu)
        checkpoint = torch.load(checkpoint_file, map_location=loc)
        start_epoch = checkpoint['epoch']
        best_fid = checkpoint['best_fid']

        dis_net.load_state_dict(checkpoint['dis_state_dict'])
        gen_optimizer.load_state_dict(checkpoint['gen_optimizer'])
        dis_optimizer.load_state_dict(checkpoint['dis_optimizer'])

        gen_net.load_state_dict(checkpoint['avg_gen_state_dict'])
        gen_avg_param = copy_params(gen_net, mode='gpu')
        gen_net.load_state_dict(checkpoint['gen_state_dict'])
        fixed_z = checkpoint['fixed_z']

        args.path_helper = checkpoint['path_helper']
        logger = create_logger(args.path_helper['log_path']) if args.rank == 0 else None
        print(f'=> loaded checkpoint {checkpoint_file} (epoch {start_epoch})')
        writer = SummaryWriter(args.path_helper['log_path']) if args.rank == 0 else None
        del checkpoint
    else:
        # create new log dir
        assert args.exp_name
        if args.rank == 0:
            args.path_helper = set_log_dir('logs', args.exp_name)
            logger = create_logger(args.path_helper['log_path'])
            writer = SummaryWriter(args.path_helper['log_path'])

    if args.rank == 0:
        logger.info(args)
    writer_dict = {
        'writer': writer,
        'train_global_steps': start_epoch * len(train_loader),
        'valid_global_steps': start_epoch // args.val_freq,
    }

    # train loop
    for epoch in range(int(start_epoch), int(args.max_epoch)):
        lr_schedulers = (gen_scheduler, dis_scheduler) if args.lr_decay else None
        cur_stage = cur_stages(epoch, args)
        print("cur_stage " + str(cur_stage)) if args.rank == 0 else 0
        print(f"path: {args.path_helper['prefix']}") if args.rank == 0 else 0

        train(args, gen_net, dis_net, gen_optimizer, dis_optimizer, gen_avg_param, train_loader, epoch, writer_dict,
              fixed_z, lr_schedulers)

        if args.rank == 0 and args.show:
            backup_param = copy_params(gen_net)
            load_params(gen_net, gen_avg_param, args, mode="cpu")
            save_samples(args, fixed_z, fid_stat, epoch, gen_net, writer_dict)
            load_params(gen_net, backup_param, args)

        # plot synthetic data
        gen_net.eval()
        plot_buf = gen_plot(gen_net, epoch)
        image = PIL.Image.open(plot_buf)
        image = ToTensor()(image).unsqueeze(0)
        writer.add_image('Image', image[0], epoch)
        is_best = False
        avg_gen_net = deepcopy(gen_net)
        load_params(avg_gen_net, gen_avg_param, args)
        save_checkpoint({
            'epoch': epoch + 1,
            'gen_model': args.gen_model,
            'dis_model': args.dis_model,
            'gen_state_dict': gen_net.module.state_dict(),
            'dis_state_dict': dis_net.module.state_dict(),
            'avg_gen_state_dict': avg_gen_net.module.state_dict(),
            'gen_optimizer': gen_optimizer.state_dict(),
            'dis_optimizer': dis_optimizer.state_dict(),
            'best_fid': best_fid,
            'path_helper': args.path_helper,
            'fixed_z': fixed_z
        }, is_best, args.path_helper['ckpt_path'], filename="checkpoint")
        del avg_gen_net


if __name__ == '__main__':
    main()
