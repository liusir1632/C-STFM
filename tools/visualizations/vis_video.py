import argparse
import os
import numpy as np

from openstl.datasets import dataset_parameters
from openstl.utils import (show_video_gif_multiple, show_video_gif_single, show_video_line,
                           show_taxibj, show_weather_bench)


# 用于进行最小-最大归一化操作
def min_max_norm(data):
    _min, _max = np.min(data), np.max(data)
    data = (data - _min) / (_max - _min)
    return data


# 解析命令行参数
def parse_args():
    parser = argparse.ArgumentParser(
        description='Visualization of a STL model')

    parser.add_argument('--dataname', '-d', default=None, type=str,
                        help='The name of dataset (default: "mmnist")')
    parser.add_argument('--index', '-i', default=0, type=int, help='The index of a video sequence to show')
    parser.add_argument('--work_dirs', '-w', default=None, type=str,
                        help='Path to the work_dir or the path to a set of work_dirs')
    parser.add_argument('--vis_dirs', '-v', action='store_true', default=False,
                        help='Whether to visualize a set of work_dirs')
    parser.add_argument('--reload_input', action='store_true', default=False,
                        help='Whether to reload the input and true for each method')
    parser.add_argument('--save_dirs', '-s', default='vis_figures', type=str,
                        help='The path to save visualization results')
    parser.add_argument('--vis_channel', '-vc', default=-1, type=int,
                        help='Select a channel to visualize as the heatmap')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    
    # 如果未提供数据集名称（dataname）和工作目录路径（work_dirs），则触发 AssertionError
    assert args.dataname is not None and args.work_dirs is not None, \
        'The name of dataset and the path to work_dirs are required'

    #  将工作目录路径保存到 base_dir 变量中
    base_dir = args.work_dirs
    
    # 检查 work_dirs 是否是一个有效的目录
    assert os.path.isdir(args.work_dirs)
    
    # 如果 vis_dirs 参数为真（True），则执行下面的代码块
    if args.vis_dirs:
        # 获取工作目录下的所有子目录的名称，这些子目录代表不同的STL方法
        method_list = os.listdir(args.work_dirs)
    
    # 如果 vis_dirs 参数为假（False），则执行下面的代码块
    else:
        # 创建一个包含工作目录名称的列表，用于表示单个STL方法
        method_list = [args.work_dirs.split('/')[-1]]
        # 更新 base_dir 变量，以包含所有STL方法的共同父目录
        base_dir = base_dir.split(method_list[0])[0]

    # 根据数据集名称来判断是否使用RGB格式的数据。如果数据集名称是指定的几种情况，将 use_rgb 设为假（False）；否则设为真（True）
    use_rgb = False if args.dataname in ['mfmnist', 'mmnist', 'kth20', 'kth', 'kth40'] else True
    
    # 将命令行参数解析结果的字典形式存储在 config 变量中
    config = args.__dict__
    
    # 更新 config 变量，将数据集参数添加到其中
    config.update(dataset_parameters[args.dataname])
    
    # 将 args.index 和数据集参数中的 aft_seq_length 存储到 idx 和 ncols 变量中
    idx, ncols = args.index, config['aft_seq_length']
    
    # 如果保存结果的目录不存在，执行下面的代码块
    if not os.path.isdir(args.save_dirs):
        # 创建保存结果的目录
        os.mkdir(args.save_dirs)
    
    #  如果指定了要可视化的通道（vis_channel 参数不为 -1），执行下面的代码块
    if args.vis_channel != -1:  # choose a channel
        # 创建一个通道后缀，用于在结果文件名中标识可视化的通道
        c_surfix = f"_C{args.vis_channel}"
        
        # 检查指定的通道是否在有效范围内
        assert 0 <= args.vis_channel <= config['in_shape'][1], 'Channel index out of range'
    # 如果未指定要可视化的通道，执行下面的代码块
    
    else:
        # 将通道后缀设置为空字符串
        c_surfix = ""
        # 检查数据集名称是否在特定的列表中，如果是，则触发 AssertionError
        assert args.dataname not in ['taxibj', 'weather_uv10_5_625'], 'Please select a channel'

    
    # 创建三个空字典，用于存储预测、输入和真实数据
    # loading results
    predicts_dict, inputs_dict, trues_dict = dict(), dict(), dict()
    
    # 创建一个空列表 empty_keys，用于存储未成功加载结果数据的STL方法的名称
    empty_keys = list()
    
    # 开始一个循环，遍历STL方法的名称列表
    for method in method_list:
        try:
            # 尝试加载指定STL方法的预测结果数据，将其存储在 predicts_dict 字典中
            predicts_dict[method] = np.load(os.path.join(base_dir, method, 'saved/preds.npy'))
            # 如果数据集名称包含字符串 'weather'，执行下面的代码块
            if 'weather' in args.dataname:
                # 对加载的预测数据进行最小-最大归一化处理
                predicts_dict[method] = min_max_norm(predicts_dict[method])
        
        # 如果加载失败，执行下面的代码块
        except:
            # 将加载失败的STL方法的名称添加到 empty_keys 列表中
            empty_keys.append(method)
            
            # 打印出加载失败的STL方法名称
            print('Failed to read the results of', method)
    
    # 确保至少加载了一个STL方法的结果数据，否则触发 AssertionError
    assert len(predicts_dict.keys()) >= 1, 'The results should not be empty'
    
    # 遍历加载失败的STL方法名称列表
    for k in empty_keys:
        # 从 method_list 列表中移除加载失败的STL方法
        method_list.pop(method_list.index(k))

    # 重新遍历STL方法名称列表
    for method in method_list:
        # 加载第一个STL方法的输入数据
        inputs = np.load(os.path.join(base_dir, method_list[0], 'saved/inputs.npy'))
        
        # 加载第一个STL方法的真实数据
        trues = np.load(os.path.join(base_dir, method_list[0], 'saved/trues.npy'))
        
        # 如果数据集名称包含字符串 'weather'，执行下面的代码块
        if 'weather' in args.dataname:
            #  对输入数据和真实数据进行最小-最大归一化处理
            inputs = min_max_norm(inputs)
            trues = min_max_norm(trues)
            
            # 使用 show_weather_bench 函数对输入数据和真实数据进行可视化处理，并将其转置为适合显示的格式
            inputs = show_weather_bench(inputs[idx, 0:ncols, ...], src_img=None, cmap='GnBu').transpose(0, 3, 1, 2)
            trues = show_weather_bench(trues[idx, 0:ncols, ...], src_img=None, cmap='GnBu').transpose(0, 3, 1, 2)
        
        elif 'taxibj' in args.dataname:
            inputs = show_taxibj(inputs[idx, 0:ncols, ...], cmap='viridis').transpose(0, 3, 1, 2)
            trues = show_taxibj(trues[idx, 0:ncols, ...], cmap='viridis').transpose(0, 3, 1, 2)
        
        # 如果不属于以上两种情况的数据集，执行下面的代码块
        else:
            # 从输入和真实数据中选择特定索引的数据
            inputs, trues = inputs[idx], trues[idx]
        
        # 如果没有指定重新加载输入和真实数据，执行下面的代码块
        if not args.reload_input:  # load the input and true for each method
            # 跳出当前循环，因为只需要加载一次输入和真实数据
            break
        else:
            # 将输入和真实数据存储在 inputs_dict 和 trues_dict 字典中，以备后续STL方法使用
            inputs_dict[method], trues_dict[method] = inputs, trues

    # 开始一个循环，遍历STL方法名称列表
    # plot gifs and figures of the STL methods
    for i, method in enumerate(method_list):
        # 打印STL方法的名称和对应预测结果的形状信息
        print(method, predicts_dict[method][idx].shape)
        
        #  如果指定了重新加载输入和真实数据，执行下面的代码块
        if args.reload_input:
            # 使用已加载的输入和真实数据
            inputs, trues = inputs_dict[method], trues_dict[method]
        
        # 如果数据集名称包含字符串 'weather'，执行下面的代码块
        if 'weather' in args.dataname:
            #  使用 show_weather_bench 函数对预测数据进行可视化处理，并将其保存在 preds 变量中
            preds = show_weather_bench(predicts_dict[method][idx, 0:ncols, ...],
                                       src_img=None, cmap='GnBu', vis_channel=args.vis_channel)
            
            # 将预测数据的维度转置，以适合显示的格式
            preds = preds.transpose(0, 3, 1, 2)
        
        elif 'taxibj' in args.dataname:
            preds = show_taxibj(predicts_dict[method][idx, 0:ncols, ...],
                                cmap='viridis', vis_channel=args.vis_channel)
            preds = preds.transpose(0, 3, 1, 2)
        else:
            # 使用预测数据
            preds = predicts_dict[method][idx]

        # 如果是STL方法列表的第一个方法，执行下面的代码块
        if i == 0:
            # 使用 show_video_line 函数对输入数据进行可视化，创建并保存图像文件，然后输出到指定路径
            show_video_line(inputs.copy(), ncols=config['pre_seq_length'], vmax=0.6, cbar=False,
                out_path='{}/{}_input{}'.format(args.save_dirs, args.dataname+c_surfix, str(idx)+'.png'),
                format='png', use_rgb=use_rgb)
            
            # 使用 show_video_line 函数对真实数据进行可视化，创建并保存图像文件，然后输出到指定路径
            show_video_line(trues.copy(), ncols=config['aft_seq_length'], vmax=0.6, cbar=False,
                out_path='{}/{}_true{}'.format(args.save_dirs, args.dataname+c_surfix, str(idx)+'.png'),
                format='png', use_rgb=use_rgb)
            
            # 使用 show_video_gif_single 函数创建并保存输入数据和真实数据的GIF动画文件，输出到指定路径
            show_video_gif_single(inputs.copy(), use_rgb=use_rgb,
                out_path='{}/{}_{}_{}_input'.format(args.save_dirs, args.dataname+c_surfix, method, idx))
            show_video_gif_single(trues.copy(), use_rgb=use_rgb,
                out_path='{}/{}_{}_{}_true'.format(args.save_dirs, args.dataname+c_surfix, method, idx))

        # 使用 show_video_line 函数对预测数据进行可视化，创建并保存图像文件，然后输出到指定路径
        show_video_line(preds, ncols=ncols, vmax=0.6, cbar=False,
                        out_path='{}/{}_{}_{}'.format(args.save_dirs, args.dataname+c_surfix, method, str(idx)+'.png'),
                        format='png', use_rgb=use_rgb)
        
        # 使用 show_video_gif_multiple 函数创建包含输入、真实和预测数据的GIF动画文件，输出到指定路径
        show_video_gif_multiple(inputs, trues, preds, use_rgb=use_rgb,
                                out_path='{}/{}_{}_{}'.format(args.save_dirs, args.dataname+c_surfix, method, idx))
        
        # 使用 show_video_gif_single 函数创建并保存预测数据的GIF动画文件，输出到指定路径
        show_video_gif_single(preds, use_rgb=use_rgb,
                              out_path='{}/{}_{}_{}_pred'.format(args.save_dirs, args.dataname+c_surfix, method, idx))


if __name__ == '__main__':
    main()
