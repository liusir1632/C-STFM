# Copyright (c) CAIRI AI Lab. All rights reserved

import argparse


def create_parser():
    parser = argparse.ArgumentParser(
        description='OpenSTL train/test a model')
    # Set-up parameters
    parser.add_argument('--device', default='cuda', type=str,
                        help='Name of device to use for tensor computations (cuda/cpu)')
    parser.add_argument('--dist', action='store_true', default=False,
                        help='Whether to use distributed training (DDP)')
    parser.add_argument('--display_step', default=10, type=int,
                        help='Interval in batches between display of training metrics')
    parser.add_argument('--res_dir', default='work_dirs', type=str)
    parser.add_argument('--ex_name', '-ex', default='Debug', type=str)
    parser.add_argument('--use_gpu', default=True, type=bool)
    parser.add_argument('--fp16', action='store_true', default=False,
                        help='Whether to use Native AMP for mixed precision training (PyTorch=>1.6.0)')
    parser.add_argument('--torchscript', action='store_true', default=False,
                        help='Whether to use torchscripted model')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--diff_seed', action='store_true', default=False,
                        help='Whether to set different seeds for different ranks')
    parser.add_argument('--fps', action='store_true', default=False,
                        help='Whether to measure inference speed (FPS)')
    parser.add_argument('--empty_cache', action='store_true', default=True,
                        help='Whether to empty cuda cache after GPU training')
    parser.add_argument('--find_unused_parameters', action='store_true', default=False,
                        help='Whether to find unused parameters in forward during DDP training')
    parser.add_argument('--broadcast_buffers', action='store_false', default=True,
                        help='Whether to set broadcast_buffers to false during DDP training')
    parser.add_argument('--resume_from', type=str, default=None, help='the checkpoint file to resume from')
    parser.add_argument('--auto_resume', action='store_true', default=False,
                        help='When training was interupted, resume from the latest checkpoint')
    parser.add_argument('--test', action='store_true', default=False, help='Only performs testing')
    parser.add_argument('--inference', '-i', action='store_true', default=False, help='Only performs inference')
    parser.add_argument('--deterministic', action='store_true', default=True,
                        help='whether to set deterministic options for CUDNN backend (reproducable)')
    parser.add_argument('--launcher', default='none', type=str,
                        choices=['none', 'pytorch', 'slurm', 'mpi'],
                        help='job launcher for distributed training')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--port', type=int, default=29500,
                        help='port only works when launcher=="slurm"')

    # dataset parameters
    parser.add_argument('--batch_size', '-b', default=64, type=int, help='Training batch size')
    parser.add_argument('--val_batch_size', '-vb', default=64, type=int, help='Validation batch size')
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--data_root', default='./data')
    parser.add_argument('--dataname', '-d', default='mmnist', type=str,
                        help='Dataset name (default: "mmnist")')
    parser.add_argument('--pre_seq_length', default=None, type=int, help='Sequence length before prediction')
    parser.add_argument('--aft_seq_length', default=None, type=int, help='Sequence length after prediction')
    parser.add_argument('--total_length', default=None, type=int, help='Total Sequence length for prediction')
    parser.add_argument('--use_augment', action='store_true', default=False,
                        help='Whether to use image augmentations for training')
    parser.add_argument('--use_prefetcher', action='store_true', default=False,
                        help='Whether to use prefetcher for faster data loading')
    parser.add_argument('--drop_last', action='store_true', default=False,
                        help='Whether to drop the last batch in the val data loading')

    # method parameters
    parser.add_argument('--method', '-m', default='SimVP', type=str,
                        choices=['ConvLSTM', 'convlstm', 'CrevNet', 'crevnet', 'DMVFN', 'dmvfn', 'E3DLSTM', 'e3dlstm',
                                 'MAU', 'mau', 'MIM', 'mim', 'PhyDNet', 'phydnet', 'PredNet', 'prednet',
                                 'PredRNN', 'predrnn', 'PredRNNpp', 'predrnnpp', 'PredRNNv2', 'predrnnv2',
                                 'SimVP', 'simvp', 'TAU', 'tau', 'MMVP', 'mmvp', 'SwinLSTM', 'swinlstm', 'swinlstm_d', 'swinlstm_b',
                                 
                                 'METNET','metnet1','metnet2','convlstm1','convlstm2','convlstm3','convlstm4',
                                 'convlstm5','convlstm6','convlstm7','convlstm8','convlstm9',
                                 'tau1','tau2','tau3','tau4','tau5','tau6','tau7','tau8','tau9','tau10','tau11','tau12','tau13','tau14',
                                 'tau15','tau16','tau17','tau18','tau19','tau20','tau21','tau22','tau23','tau24','tau25','tau26','tau27',
                                 'tau28','tau29','tau30','tau31','tau32','tau33','tau34','tau35','tau36','tau38','tau39','tau40','tau41',
                                 'tau42','tau43','tau44','tau45','tau46','tau47','tau50','tau51','tau52','tau53','tau56','tau57','tau58','tau59',
                                 'tau60','tau61','tau62','tau63','tau70','tau71','tau60z','tau72','tau73',
                                 'tau102','tau100','tau101','tau102','tau103','tau104','tau105','tau106','tau107','tau108','tau109','tau110','tau111','tau112',
                                 'tau120','tau121','tau122','tau123','tau124','tau125','tau126','tau127',
                                 'tau130','tau131','tau132','tau133','tau134','tau135','tau136','tau137','tau138',
                                 
                                 'tau150','tau151','tau152','tau153','tau154','tau155','tau156','tau157','tau158','tau159',
                                 'tau160','tau161','tau162','tau163','tau164','tau165','tau166',
                                 'tau167','tau168','tau169','tau170','tau171',
                                 
                                 'svp1','svp2','svp3','svp4','svp5','svp6','svp7','svp8','svp9','svp10','svp11','svp12','svp13','svp14','svp15','svp16','svp17','svp18','svp19',
                                 'svp20','svp21','svp201','svp22','svp23','svp24','svp25','svp26','svp27','svp28','svp29','svp30','svp31','svp32','svp33','svp34','svp35','svp36',
                                 'svp37','svp38','svp39','svp40','svp41','svp42','svp43','svp50','svp51','svp52','svp53','svp54','svp55','svp56','svp57','svp58','svp59','svp60',
                                 'svp61','svp62','svp63','svp64','svp65','svp66','svp67','svp68','svp69','svp70','svp71','svp72','svp73','svp74','svp75','svp76','svp77','svp78',
                                 'svp79','svp80','svp81','svp82','svp83','svp84','svp85','svp86','svp87','svp88','svp89','svp90','svp91','svp92','svp93','svp94','svp95','svp96',
                                 'svp97','svp98','svp99','svp100','svp101','svp102','svp103','svp104','svp105','svp106','svp107',
                                 
                                 'svp_moving_lstm1','svp_moving_lstm2','svp_moving_lstm3','svp_moving_lstm4',
                                 'svp_moving1','svp_moving2','svp_moving3','svp_moving4','svp_moving5',
                                 ],
                        help='Name of video prediction method to train (default: "SimVP")')
    
    # parser.add_argument('--config_file', '-c', default='configs/mmnist/simvp/SimVP_gSTA.py', type=str,
    #                     help='Path to the default config file')
    
    parser.add_argument('--config_file', '-c', default='', type=str,
                        help='Path to the default config file')

    parser.add_argument('--model_type', default=None, type=str,
                        help='Name of model for SimVP (default: None)')
    parser.add_argument('--drop', type=float, default=0.0, help='Dropout rate(default: 0.)')
    parser.add_argument('--drop_path', type=float, default=0.0, help='Drop path rate for SimVP (default: 0.)')
    parser.add_argument('--overwrite', action='store_true', default=False,
                        help='Whether to allow overwriting the provided config file with args')

    # Training parameters (optimizer)
    parser.add_argument('--epoch', '-e', default=100, type=int, help='end epochs (default: 200)')
    parser.add_argument('--log_step', default=1, type=int, help='Log interval by step')
    parser.add_argument('--opt', default='adam', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adam"')
    parser.add_argument('--opt_eps', default=None, type=float, metavar='EPSILON',
                        help='Optimizer epsilon (default: None, use opt default)')
    parser.add_argument('--opt_betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer betas (default: None, use opt default)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='Optimizer sgd momentum (default: 0.9)')
    parser.add_argument('--weight_decay', default=0., type=float, help='Weight decay')
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--clip_mode', type=str, default='norm',
                        help='Gradient clipping mode. One of ("norm", "value", "agc")')
    parser.add_argument('--early_stop_epoch', default=-1, type=int,
                        help='Check to early stop after this epoch')
    parser.add_argument('--no_display_method_info', action='store_true', default=False,
                        help='Do not display method info')

    # Training parameters (scheduler)
    parser.add_argument('--sched', default=None, type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "onecycle"')
    parser.add_argument('--lr', default=None, type=float, help='Learning rate (default: 1e-4)')
    parser.add_argument('--lr_k_decay', type=float, default=1.0,
                        help='learning rate k-decay for cosine/poly (default: 1.0)')
    parser.add_argument('--warmup_lr', type=float, default=1e-5, metavar='LR',
                        help='warmup learning rate (default: 1e-5)')
    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    parser.add_argument('--final_div_factor', type=float, default=1e4,
                        help='min_lr = initial_lr/final_div_factor for onecycle scheduler')
    parser.add_argument('--warmup_epoch', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--decay_epoch', type=float, default=100, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--decay_rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')
    parser.add_argument('--filter_bias_and_bn', type=bool, default=False,
                        help='Whether to set the weight decay of bias and bn to 0')

    return parser


def default_parser():
    default_values = {
        # Set-up parameters
        'device': 'cuda',
        'dist': False,
        'display_step': 10,
        'res_dir': 'work_dirs',
        'ex_name': 'Debug',
        'use_gpu': True,
        'fp16': False,
        'torchscript': False,
        'seed': 42,
        'diff_seed': False,
        'fps': False,
        'empty_cache': True,
        'find_unused_parameters': False,
        'broadcast_buffers': True,
        'resume_from': None,
        'auto_resume': False,
        'test': False,
        'inference': False,
        'deterministic': False,
        'launcher': 'pytorch',
        'local_rank': 0,
        'port': 29500,
        # dataset parameters
        'batch_size': 16,
        'val_batch_size': 16,
        'num_workers': 4,
        'data_root': './data',
        'dataname': 'mmnist',
        'pre_seq_length': 10,
        'aft_seq_length': 10,
        'total_length': 20,
        'use_augment': False,
        'use_prefetcher': False,
        'drop_last': False,
        # method parameters
        'method': 'SimVP',
        'config_file': 'configs/mmnist/simvp/SimVP_gSTA.py',
        'model_type': 'gSTA',
        'drop': 0,
        'drop_path': 0,
        'overwrite': False,
        # Training parameters (optimizer)
        'epoch': 200,
        'log_step': 1,
        'opt': 'adam',
        'opt_eps': None,
        'opt_betas': None,
        'momentum': 0.9,
        'weight_decay': 0,
        'clip_grad': None,
        'clip_mode': 'norm',
        'early_stop_epoch': -1,
        'no_display_method_info': False,
        # Training parameters (scheduler)
        'sched': 'onecycle',
        'lr': 1e-3,
        'lr_k_decay': 1.0,
        'warmup_lr': 1e-5,
        'min_lr': 1e-6,
        'final_div_factor': 1e4,
        'warmup_epoch': 0,
        'decay_epoch': 100,
        'decay_rate': 0.1,
        'filter_bias_and_bn': False,
    }
    return default_values
