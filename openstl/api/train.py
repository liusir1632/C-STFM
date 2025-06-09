# Copyright (c) CAIRI AI Lab. All rights reserved

import os
import os.path as osp
import time
import logging
import json
import numpy as np
from typing import Dict, List
from fvcore.nn import FlopCountAnalysis, flop_count_table

import torch
import torch.distributed as dist

from openstl.core import Hook, metric, Recorder, get_priority, hook_maps
from openstl.methods import method_maps
from openstl.utils import (set_seed, print_log, output_namespace, check_dir, collect_env,
                           init_dist, init_random_seed,
                           get_dataset, get_dist_info, measure_throughput, weights_to_cpu)

try:
    import nni
    has_nni = True
except ImportError: 
    has_nni = False


class BaseExperiment(object):
    """The basic class of PyTorch training and evaluation."""

    def __init__(self, args, dataloaders=None):
        """Initialize experiments (non-dist as an example)"""
        self.args = args
        self.config = self.args.__dict__
        self.device = self.args.device
        self.method = None
        self.args.method = self.args.method.lower()
        self._epoch = 0
        self._iter = 0
        self._inner_iter = 0
        self._max_epochs = self.config['epoch']
        self._max_iters = None
        self._hooks: List[Hook] = []
        self._rank = 0
        self._world_size = 1
        self._dist = self.args.dist
        self._early_stop = self.args.early_stop_epoch

        self.best_epoch = -1
        # 假设你有一个变量来存储最佳MSE
        self.best_mse = float('1000000')

        self._preparation(dataloaders)
        if self._rank == 0:
            print_log(output_namespace(self.args))
            if not self.args.no_display_method_info:
                self.display_method_info()

    def _acquire_device(self):
        """Setup devices"""
        if self.args.use_gpu:
            self._use_gpu = True
            if self.args.dist:
                device = f'cuda:{self._rank}'
                torch.cuda.set_device(self._rank)
                print_log(f'Use distributed mode with GPUs: local rank={self._rank}')
            else:
                # device = torch.device('cuda:0')
                device = self.device
                print_log(f'Use non-distributed mode with GPU: {device}')
        else:
            self._use_gpu = False
            device = torch.device('cpu')
            print_log('Use CPU')
            if self.args.dist:
                assert False, "Distributed training requires GPUs"
        return device

    def _preparation(self, dataloaders=None):
        """Preparation of environment and basic experiment setups"""
        if 'LOCAL_RANK' not in os.environ:
            os.environ['LOCAL_RANK'] = str(self.args.local_rank)

        # init distributed env first, since logger depends on the dist info.
        if self.args.launcher != 'none' or self.args.dist:
            self._dist = True
        if self._dist:
            assert self.args.launcher != 'none'
            dist_params = dict(backend='nccl', init_method='env://')
            if self.args.launcher == 'slurm':
                dist_params['port'] = self.args.port
            init_dist(self.args.launcher, **dist_params)
            self._rank, self._world_size = get_dist_info()
            # re-set gpu_ids with distributed training mode
            self._gpu_ids = range(self._world_size)
        self.device = self._acquire_device()
        if self._early_stop <= self._max_epochs // 5:
            self._early_stop = self._max_epochs * 2

        # log and checkpoint
        base_dir = self.args.res_dir if self.args.res_dir is not None else 'work_dirs'
        self.path = osp.join(base_dir, self.args.ex_name if not self.args.ex_name.startswith(self.args.res_dir) \
            else self.args.ex_name.split(self.args.res_dir+'/')[-1])
        self.checkpoints_path = osp.join(self.path, 'checkpoints')
        if self._rank == 0:
            check_dir(self.path)
            check_dir(self.checkpoints_path)

        sv_param = osp.join(self.path, 'model_param.json')
        if self._rank == 0:
            with open(sv_param, 'w') as file_obj:
                json.dump(self.args.__dict__, file_obj)

            for handler in logging.root.handlers[:]:
                logging.root.removeHandler(handler)
            timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
            prefix = 'train' if (not self.args.test and not self.args.inference) else 'test'
            logging.basicConfig(level=logging.INFO,
                                filename=osp.join(self.path, '{}_{}.log'.format(prefix, timestamp)),
                                filemode='a', format='%(asctime)s - %(message)s')

        # log env info
        env_info_dict = collect_env()
        env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
        dash_line = '-' * 60 + '\n'
        if self._rank == 0:
            print_log('Environment info:\n' + dash_line + env_info + '\n' + dash_line)

        # set random seeds
        if self._dist:
            seed = init_random_seed(self.args.seed)
            seed = seed + dist.get_rank() if self.args.diff_seed else seed
        else:
            seed = self.args.seed
        set_seed(seed)

        # prepare data
        self._get_data(dataloaders)
        # build the method
        self._build_method()
        # build hooks
        self._build_hook()
        # resume traing
        if self.args.auto_resume:
            self.args.resume_from = osp.join(self.checkpoints_path, 'latest.pth')
        if self.args.resume_from is not None:
            self._load(name=self.args.resume_from)
        self.call_hook('before_run')

    def _build_method(self):
        self.steps_per_epoch = len(self.train_loader)
        self.method = method_maps[self.args.method](self.args, self.device, self.steps_per_epoch)
        self.method.model.eval()
        # setup ddp training
        if self._dist:
            self.method.model.cuda()
            if self.args.torchscript:
                self.method.model = torch.jit.script(self.method.model)
            self.method._init_distributed()

    def _build_hook(self):
        for k in self.args.__dict__:
            if k.lower().endswith('hook'):
                hook_cfg = self.args.__dict__[k].copy()
                priority = get_priority(hook_cfg.pop('priority', 'NORMAL'))
                hook = hook_maps[k.lower()](**hook_cfg)
                if hasattr(hook, 'priority'):
                    raise ValueError('"priority" is a reserved attribute for hooks')
                hook.priority = priority  # type: ignore
                # insert the hook to a sorted list
                inserted = False
                for i in range(len(self._hooks) - 1, -1, -1):
                    if priority >= self._hooks[i].priority:  # type: ignore
                        self._hooks.insert(i + 1, hook)
                        inserted = True
                        break
                if not inserted:
                    self._hooks.insert(0, hook)

    def call_hook(self, fn_name: str) -> None:
        """Run hooks by the registered names"""
        for hook in self._hooks:
            getattr(hook, fn_name)(self)

    def _get_hook_info(self):
        """Get hook information in each stage"""
        stage_hook_map: Dict[str, list] = {stage: [] for stage in Hook.stages}
        for hook in self._hooks:
            priority = hook.priority  # type: ignore
            classname = hook.__class__.__name__
            hook_info = f'({priority:<12}) {classname:<35}'
            for trigger_stage in hook.get_triggered_stages():
                stage_hook_map[trigger_stage].append(hook_info)

        stage_hook_infos = []
        for stage in Hook.stages:
            hook_infos = stage_hook_map[stage]
            if len(hook_infos) > 0:
                info = f'{stage}:\n'
                info += '\n'.join(hook_infos)
                info += '\n -------------------- '
                stage_hook_infos.append(info)
        return '\n'.join(stage_hook_infos)

    def _get_data(self, dataloaders=None):
        """Prepare datasets and dataloaders"""
        if dataloaders is None:
            self.train_loader, self.vali_loader, self.test_loader = \
                get_dataset(self.args.dataname, self.config)
        else:
            self.train_loader, self.vali_loader, self.test_loader = dataloaders

        if self.vali_loader is None:
            self.vali_loader = self.test_loader
        self._max_iters = self._max_epochs * len(self.train_loader)

    def _save(self, name=''):
        """Saving models and meta data to checkpoints"""
        checkpoint = {
            'epoch': self._epoch + 1,
            'optimizer': self.method.model_optim.state_dict(),
            'state_dict': weights_to_cpu(self.method.model.state_dict()) \
                if not self._dist else weights_to_cpu(self.method.model.module.state_dict()),
            'scheduler': self.method.scheduler.state_dict()}
        torch.save(checkpoint, osp.join(self.checkpoints_path, name + '.pth'))

    def _load(self, name=''):
        """Loading models from the checkpoint"""
        filename = name if osp.isfile(name) else osp.join(self.checkpoints_path, name + '.pth')
        print("filename", filename)
        try:
            checkpoint = torch.load(filename)
        except Exception as e:
            print(f"An exception occurred: {str(e)}")
            return

        if not isinstance(checkpoint, dict):
            raise RuntimeError(f'No state_dict found in checkpoint file {filename}')

        self._load_from_state_dict(checkpoint['state_dict'])
        if checkpoint.get('epoch', None) is not None:
            self._epoch = checkpoint['epoch']
            self.method.model_optim.load_state_dict(checkpoint['optimizer'])
            self.method.scheduler.load_state_dict(checkpoint['scheduler'])

    def _load_from_state_dict(self, state_dict):
        if self._dist:
            try:
                self.method.model.module.load_state_dict(state_dict)
            except:
                self.method.model.load_state_dict(state_dict)
        else:
            self.method.model.load_state_dict(state_dict)

    def display_method_info(self):
        """Plot the basic infomation of supported methods"""
        T, C, H, W = self.args.in_shape
        if self.args.method in ['simvp', 'tau', 'tau1', 'tau2', 'tau3', 'tau4', 'tau5', 'tau6', 'tau7', 'tau8', 'tau9', 'tau10',
                                 'tau11', 'tau12', 'tau13', 'tau14', 'tau15', 'tau16', 'tau17', 'tau18', 'tau19', 'tau20', 'tau21',
                                 'tau22','tau23','tau24','tau25','tau26','tau27','tau28','tau29','tau30','tau31','tau32','tau33',
                                 'tau34','tau35','tau36','tau38','tau39','tau40','tau41','tau42','tau43','tau44','tau45','tau46',
                                 'tau47','tau50','tau51','tau52','tau53','tau56','tau57','tau58','tau59','tau63','tau72','tau73',
                                 
                                 'tau100','tau101','tau102','tau103','tau104','tau105','tau106','tau107','tau108','tau109','tau110','tau111','tau112',
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
                                ]:
            input_dummy = torch.ones(1, self.args.pre_seq_length, C, H, W).to(self.device)
        elif self.args.method == 'crevnet':
            # crevnet must use the batchsize rather than 1
            input_dummy = torch.ones(self.args.batch_size, 20, C, H, W).to(self.device)
        elif self.args.method == 'phydnet':
            _tmp_input1 = torch.ones(1, self.args.pre_seq_length, C, H, W).to(self.device)
            _tmp_input2 = torch.ones(1, self.args.aft_seq_length, C, H, W).to(self.device)
            _tmp_constraints = torch.zeros((49, 7, 7)).to(self.device)
            input_dummy = (_tmp_input1, _tmp_input2, _tmp_constraints)
        elif self.args.method in ['convlstm', 'predrnnpp', 'predrnn', 'mim', 'e3dlstm', 'mau',
                                  'convlstm1','convlstm4','convlstm5','convlstm6','convlstm7',
                                  'convlstm8','convlstm9',
                                  ]:
            Hp, Wp = H // self.args.patch_size, W // self.args.patch_size
            Cp = self.args.patch_size ** 2 * C
            _tmp_input = torch.ones(1, self.args.total_length, Hp, Wp, Cp).to(self.device)
            _tmp_flag = torch.ones(1, self.args.aft_seq_length - 1, Hp, Wp, Cp).to(self.device)
            input_dummy = (_tmp_input, _tmp_flag)

        elif self.args.method == 'metnet2':
            Hp, Wp = H // self.args.patch_size, W // self.args.patch_size
            Cp = self.args.patch_size ** 2 * C
            _tmp_input = torch.ones(1, self.args.total_length, Hp, Wp, Cp).to(self.device)
            _tmp_flag = torch.ones(1, self.args.aft_seq_length - 1, Hp, Wp, Cp).to(self.device)
            input_dummy = (_tmp_input, _tmp_flag)

        elif self.args.method == 'predrnnv2':
            Hp, Wp = H // self.args.patch_size, W // self.args.patch_size
            Cp = self.args.patch_size ** 2 * C
            _tmp_input = torch.ones(1, self.args.total_length, Hp, Wp, Cp).to(self.device)
            _tmp_flag = torch.ones(1, self.args.total_length - 2, Hp, Wp, Cp).to(self.device)
            input_dummy = (_tmp_input, _tmp_flag)
        elif self.args.method == 'dmvfn':
            input_dummy = torch.ones(1, 3, C, H, W, requires_grad=True).to(self.device)
        elif self.args.method == 'prednet':
           input_dummy = torch.ones(1, 1, C, H, W, requires_grad=True).to(self.device)
        else:
            raise ValueError(f'Invalid method name {self.args.method}')

        dash_line = '-' * 80 + '\n'
        info = self.method.model.__repr__()
        flops = FlopCountAnalysis(self.method.model, input_dummy)
        flops = flop_count_table(flops)
        if self.args.fps:
            fps = measure_throughput(self.method.model, input_dummy)
            fps = 'Throughputs of {}: {:.3f}\n'.format(self.args.method, fps)
        else:
            fps = ''
        print_log('Model info:\n' + info+'\n' + flops+'\n' + fps + dash_line)

    def train(self):
        """Training loops of STL methods"""
        recorder = Recorder(verbose=True, early_stop_time=min(self._max_epochs // 10, 10))
        num_updates = self._epoch * self.steps_per_epoch
        early_stop = False
        self.call_hook('before_train_epoch')

        eta = 1.0  # PredRNN variants
        for epoch in range(self._epoch, self._max_epochs):
            if self._dist and hasattr(self.train_loader.sampler, 'set_epoch'):
                self.train_loader.sampler.set_epoch(epoch)

            num_updates, loss_mean, eta = self.method.train_one_epoch(self, self.train_loader,
                                                                      epoch, num_updates, eta)

            self._epoch = epoch
            if epoch % self.args.log_step == 0:
                cur_lr = self.method.current_lr()
                cur_lr = sum(cur_lr) / len(cur_lr)
                with torch.no_grad():
                    vali_loss = self.vali()

                if self._rank == 0:
                    print_log('Epoch: {0}, Steps: {1} | Lr: {2:.7f} | Train Loss: {3:.7f} | Vali Loss: {4:.7f}\n'.format(
                        epoch + 1, len(self.train_loader), cur_lr, loss_mean.avg, vali_loss))
                    early_stop = recorder(vali_loss, self.method.model, self.path)
                    # self._save(name='latest_{}'.format(epoch + 1))
                    # self._save(name='latest')
                    # if self._epoch % 10 == 0 or self._epoch % 5 == 0 or self._epoch == 99:
                    #     self._save(name='latest_{}'.format(epoch + 1))
                    
                    # if self._epoch % 10 == 0 or self._epoch % 5 == 0 or self._epoch % 3 == 0:
                    if self._epoch % 10 == 0 or self._epoch % 5 == 0 or self._epoch == 99 or self._epoch == 199:
                        self._save(name='latest_{}'.format(epoch + 1))  
                        # # 删除之前的模型参数文件
                        # if (self._epoch - 9) > 0 and os.path.exists(os.path.join(self.checkpoints_path, f"latest_{epoch - 9}.pth")):
                        #     os.remove(os.path.join(self.checkpoints_path, f"latest_{epoch - 9}.pth"))                         
                        
            if self._use_gpu and self.args.empty_cache:
                torch.cuda.empty_cache()
            if epoch > self._early_stop and early_stop:  # early stop training
                print_log('Early stop training at f{} epoch'.format(epoch))

        if not check_dir(self.path):  # exit training when work_dir is removed
            assert False and "Exit training because work_dir is removed"
        best_model_path = osp.join(self.path, 'checkpoint.pth')
        
        self._load_from_state_dict(torch.load(best_model_path))
        time.sleep(1)  # wait for some hooks like loggers to finish
        self.call_hook('after_run')

    def vali(self):
        """A validation loop during training"""
        self.call_hook('before_val_epoch')
        results, eval_log = self.method.vali_one_epoch(self, self.vali_loader)
        self.call_hook('after_val_epoch')
        
        if self._rank == 0:
            print_log('val\t '+eval_log)
            # print_log('self.epoch\t' +str(self._epoch))
            
            # 解析eval_log以获取MSE值
            eval_values = eval_log.split(', ')
            mse_value = float(eval_values[1].split(':')[1])  
            current_epoch = self._epoch + 1
            
            # print_log('mse_value\t' +str(mse_value))
            
            # 检查当前MSE是否是迄今为止的最佳MSE
            if mse_value < self.best_mse:
                # # 在找到新的最佳MSE时保存模型参数
                # self._save(name='best_{}'.format(current_epoch))  # 调用保存模型参数的函数
                
                # # 删除之前最佳MSE轮次的模型参数文件
                # if self.best_epoch != -1 and os.path.exists(os.path.join(self.checkpoints_path, f"best_{self.best_epoch}.pth")):
                #     os.remove(os.path.join(self.checkpoints_path, f"best_{self.best_epoch}.pth"))    
                self.best_mse = mse_value
                self.best_epoch = current_epoch   
            
            print_log('best_epoch\t '+str(self.best_epoch) + '\t\tbest_mse\t' + str(self.best_mse))   
            
            if has_nni:
                nni.report_intermediate_result(results['mse'].mean())

        return results['loss'].mean()

    def test(self):
        """A testing loop of STL methods"""
        if self.args.test:
            best_model_path = osp.join(self.path, 'checkpoint.pth')
            self._load_from_state_dict(torch.load(best_model_path))

        self.call_hook('before_val_epoch')
        results = self.method.test_one_epoch(self, self.test_loader)

        # print("results['preds'].shape", results['preds'].shape)
        # print("results['trues'].shape", results['trues'].shape)

        self.call_hook('after_val_epoch')

        if 'weather' in self.args.dataname:
            metric_list, spatial_norm = self.args.metrics, True
            channel_names = self.test_loader.dataset.data_name if 'mv' in self.args.dataname else None
        else:
            metric_list, spatial_norm, channel_names = self.args.metrics, False, None
        eval_res, eval_log = metric(results['preds'], results['trues'],
                                    self.test_loader.dataset.mean, self.test_loader.dataset.std,
                                    metrics=metric_list, channel_names=channel_names, spatial_norm=spatial_norm)
        results['metrics'] = np.array([eval_res['mae'], eval_res['mse']])

        if self._rank == 0:
            print_log(eval_log)
            folder_path = osp.join(self.path, 'saved')
            check_dir(folder_path)

            for np_data in ['metrics', 'inputs', 'trues', 'preds']:
                np.save(osp.join(folder_path, np_data + '.npy'), results[np_data])

        return eval_res['mse']

    def inference(self):
        """A inference loop of STL methods"""
        best_model_path = osp.join(self.path, 'checkpoint.pth')
        
        self._load_from_state_dict(torch.load(best_model_path))

        self.call_hook('before_val_epoch')
        results = self.method.test_one_epoch(self, self.test_loader)
        self.call_hook('after_val_epoch')

        if self._rank == 0:
            folder_path = osp.join(self.path, 'saved')
            check_dir(folder_path)
            for np_data in ['inputs', 'trues', 'preds']:
                np.save(osp.join(folder_path, np_data + '.npy'), results[np_data])

        return None
