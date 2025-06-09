import os.path as osp
import warnings
warnings.filterwarnings('ignore')

from openstl.api import BaseExperiment
from openstl.utils import (create_parser, default_parser, get_dist_info, load_config,
                           setup_multi_processes, update_config)

# 导入NNI（Neural Network Intelligence）库
try:
    import nni
    has_nni = True
except ImportError: 
    has_nni = False


if __name__ == '__main__':
    # 默认命令行参数
    """
注意：
1、设置mse、mae、rmse、ssim这些指标, 要在base_method方法里, 52行的位置设置
2、两个参数要区分: dataneme 和 data_name


--resume_from
--deterministic
--find_unused_parameters
--no_display_method_info

可视化
python vis_video.py -d weather_t850_0_25 -w /gpfs/home/lijinwen/liujun/Open/work_dirs/weather/t2m_0_25/TAU72

多卡训练-很快

单变量
PORT=2073 CUDA_VISIBLE_DEVICES=3,4 bash dist_train.sh /gpfs/home/lijinwen/liujun/Open/configs/weather/tp_0_25/ConvLSTM.py 2 -m convlstm -d weather_tp_0_25 --find_unused_parameters -b 64 -vb 64
PORT=2090 CUDA_VISIBLE_DEVICES=0,1 bash dist_train.sh /gpfs/home/lijinwen/liujun/Open/configs/mmnist/simvp/SimVP_Uniformer.py 2 -m simvp -d mmnist
PORT=2076 CUDA_VISIBLE_DEVICES=0,2 bash dist_train.sh /gpfs/home/lijinwen/liujun/Open/configs/mmnist/E3DLSTM.py 2 -m e3dlstm -d mmnist
PORT=2075 CUDA_VISIBLE_DEVICES=1,2 bash dist_train.sh /gpfs/home/lijinwen/liujun/Open/configs/mmnist/PredRNNv2.py 2 -m predrnnv2 -d mmnist
PORT=2077 CUDA_VISIBLE_DEVICES=2,3 bash dist_train.sh /gpfs/home/lijinwen/liujun/Open/configs/mmnist/E3DLSTM.py 2 -m e3dlstm -d mmnist
PORT=2078 CUDA_VISIBLE_DEVICES=3,4 bash dist_train.sh /gpfs/home/lijinwen/liujun/Open/configs/mmnist/TAU.py 2 -m tau -d mmnist
PORT=2057 CUDA_VISIBLE_DEVICES=3,4 bash dist_train.sh /gpfs/home/lijinwen/liujun/Open/configs/mmnist/TAU57.py 2 -m tau57 -d mmnist
PORT=2101 CUDA_VISIBLE_DEVICES=0,1 bash dist_train.sh /gpfs/home/lijinwen/liujun/Open/configs/weather/tp_0_25/TAU101.py 2 -m tau57 -d weather_tp_0_25
PORT=2102 CUDA_VISIBLE_DEVICES=3,4 bash dist_train.sh /gpfs/home/lijinwen/liujun/Open/configs/weather/tp_0_25/TAU102.py 2 -m tau57 -d weather_tp_0_25

PORT=2080 CUDA_VISIBLE_DEVICES=1,2 bash dist_train.sh /gpfs/home/lijinwen/liujun/Open/configs/weather/tp_0_25/TAU.py 2 -m tau -d weather_tp_0_25    3层
PORT=2081 CUDA_VISIBLE_DEVICES=1,2 bash dist_train.sh /gpfs/home/lijinwen/liujun/Open/configs/weather/tp_0_25/TAU57.py 2 -m tau57 -d weather_tp_0_25    3层


多变量
PORT=2007 CUDA_VISIBLE_DEVICES=4,5 bash dist_train.sh /home/ceshi03/test_data/Open_25jinwen/configs/weather/t2m_0_25/TAU57.py 2 -m tau57 -d weather_t_0_25 --resume_from /home/ceshi03/test_data/Open_25jinwen/work_dirs/weather/t2m_0_25/TAU57/checkpoints/latest_100.pth

PORT=2007 CUDA_VISIBLE_DEVICES=4,5 bash dist_train.sh /home/ceshi03/test_data/Open_25jinwen/configs/weather/u10_0_25/TAU57.py 2 -m tau57 -d weather_u10_0_25 --resume_from /home/ceshi03/test_data/Open_25jinwen/work_dirs/weather/v10_0_25/TAU57/checkpoints/latest_100.pth
PORT=2008 CUDA_VISIBLE_DEVICES=2,3 bash dist_train.sh /home/ceshi03/test_data/Open_25jinwen/configs/weather/temperature_850_0_25/TAU57.py 2 -m tau57 -d weather_t_0_25


PORT=2001 CUDA_VISIBLE_DEVICES=2,3 bash dist_train.sh /gpfs/home/lijinwen/liujun/Open/configs/weather/mv6_0_25/ConvLSTM.py 2 -m convlstm -d weather_mv6_0_25 --find_unused_parameters
PORT=2002 CUDA_VISIBLE_DEVICES=2,3 bash dist_train.sh /gpfs/home/lijinwen/liujun/Open/configs/weather/mv6_0_25/E3DLSTM.py 2 -m e3dlstm -d weather_mv6_0_25
PORT=2003 CUDA_VISIBLE_DEVICES=2,3 bash dist_train.sh /gpfs/home/lijinwen/liujun/Open/configs/weather/mv6_0_25/PredRNNv2.py 2 -m predrnnv2 -d weather_mv6_0_25
PORT=2004 CUDA_VISIBLE_DEVICES=2,3 bash dist_train.sh /gpfs/home/lijinwen/liujun/Open/configs/weather/mv6_0_25/SimVP_Uniformer.py 2 -m simvp -d weather_mv6_0_25
PORT=2005 CUDA_VISIBLE_DEVICES=2,3 bash dist_train.sh /gpfs/home/lijinwen/liujun/Open/configs/weather/mv6_0_25/SimVP_gSTA.py 2 -m simvp -d weather_mv6_0_25
PORT=2046 CUDA_VISIBLE_DEVICES=2,4 bash dist_train.sh /gpfs/home/lijinwen/liujun/Open/configs/weather/mv6_0_25/TAU.py 2 -m tau -d weather_mv6_0_25
PORT=2007 CUDA_VISIBLE_DEVICES=2,3 bash dist_train.sh /gpfs/home/lijinwen/liujun/Open/configs/weather/mv6_0_25/TAU57.py 2 -m tau57 -d weather_mv6_0_25  要修改模型


测试的
PORT=2316 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash dist_test.sh /home/liujie/lj/Open/configs/weather/v10_0_25/TAU.py 8 /home/liujie/lj/Open/work_dirs/weather/v10_0_25/TAU -d weather_v10_0_25 -b 8 -vb 8



PORT=2316 CUDA_VISIBLE_DEVICES=2,3 bash dist_test.sh /home/ceshi03/test_data/Open_25jinwen/configs/weather/t2m_0_25/TAU57.py 2 /home/ceshi03/test_data/Open_25jinwen/work_dirs/weather/t2m_0_25/TAU57 -d weather_t2m_0_25 -b 64 -vb 64
    
python -m torch.distributed.launch --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --nproc_per_node=2 --master_port=2316 /path/to/your/train.py --config /home/ceshi03/test_data/Open_25jinwen/configs/weather/t2m_0_25/TAU57.py --checkpoint /home/ceshi03/test_data/Open_25jinwen/work_dirs/weather/t2m_0_25/TAU57 -d weather_t2m_0_25 -b 64 -vb 64


    """
    
    # 解析命令行参数，将其存储在args变量中
    args = create_parser().parse_args()
    # 将args转化为一个字典对象config，用于存储配置信息
    config = args.__dict__

    # 如果NNI库可用，尝试获取下一个超参数配置，然后将这些超参数合并到config字典中
    if has_nni:
        tuner_params = nni.get_next_parameter()
        config.update(tuner_params)

    # 根据命令行参数中的dataname和method构建一个配置文件路径cfg_path。
    # 如果指定了config_file，则使用指定的配置文件路径
    cfg_path = osp.join('./configs', args.dataname, f'{args.method}.py') \
        if args.config_file is None else args.config_file

    # 如果args.overwrite为True，将加载的配置文件中的配置信息合并到config中，同时排除了一些键，如'method'。
    if args.overwrite:
        config = update_config(config, load_config(cfg_path),
                               exclude_keys=['method'])
    # 如果args.overwrite为False，将加载的配置文件中的配置信息合并到config中，但排除了一些键，同时还将默认值填充到未提供的配置项中。
    else:
        loaded_cfg = load_config(cfg_path)
        config = update_config(config, loaded_cfg,
                               exclude_keys=['method', 
                                             'drop_path', 'warmup_epoch'])
        default_values = default_parser()
        for attribute in default_values.keys():
            if config[attribute] is None:
                config[attribute] = default_values[attribute]

    # 设置多进程参数
    setup_multi_processes(config)

    print('>'*35 + ' training ' + '<'*35)
    
    # 创建一个BaseExperiment对象，该对象用于执行模型的训练。
    # 然后，通过exp.train()方法执行模型的训练操作
    exp = BaseExperiment(args)
    rank, _ = get_dist_info()
    exp.train()
    
    # 如果进程的rank是0（即主进程），则代码打印一条提示信息，
    # 然后调用exp.test()方法执行模型的测试操作，将测试结果存储在mse变量中
    if rank == 0:
        print('>'*35 + ' testing  ' + '<'*35)
    mse = exp.test()

    # 如果进程的rank是0并且已启用 NNI 支持
    # 代码通过nni.report_final_result(mse)向 NNI 报告最终的测试结果。
    if rank == 0 and has_nni:
        nni.report_final_result(mse)

