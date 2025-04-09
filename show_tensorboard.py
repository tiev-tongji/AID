import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

# 从TensorBoard日志目录加载事件文件
def load_tensorboard_data(log_dir):
    ea = event_accumulator.EventAccumulator(log_dir)
    ea.Reload()  # 加载事件数据
    return ea

# 计算给定数据的均值和方差
def calculate_mean_and_std(data):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    return mean, std

# 提取指定tag的数据
def extract_data(ea, tag):
    data = ea.Scalars(tag)
    steps = [x.step for x in data]
    values = [x.value for x in data]
    return np.array(steps), np.array(values)

# 提取所有seed文件夹中的数据
def extract_seed_data(base_dir, tags):
    all_seed_data = {}

    # 遍历所有seed文件夹
    for seed_folder in os.listdir(base_dir):
        if seed_folder.endswith('_'):
            continue
        seed_path = os.path.join(base_dir, seed_folder)
        if os.path.isdir(seed_path):
            seed_data = {}
            # 加载每个文件夹中的TensorBoard数据
            for tag in tags:
                ea = load_tensorboard_data(seed_path)
                steps, values = extract_data(ea, tag)
                if tag not in seed_data:
                    seed_data[tag] = []
                seed_data[tag].append(values)

            all_seed_data[seed_folder] = seed_data

    return all_seed_data

# 对不同长度的数组进行对齐，使它们具有相同的形状
def align_data(all_seed_data, tag):
    # 获取所有seeds的steps长度
    seed_steps = [len(data[tag][0]) for data in all_seed_data.values()]
    min_steps = min(seed_steps)
    min_steps = min(min_steps, 100)
    
    aligned_data = []

    for seed_folder, seed_data in all_seed_data.items():
        values = seed_data[tag]
        for value in values:
            if len(value) > min_steps:
                aligned_value = value[:min_steps]
            else:
                aligned_value = value
            aligned_data.append(aligned_value)

    return np.array(aligned_data)

# 平滑函数：滑动平均或指数加权平均
def smooth_data(data, smooth_factor=0.8):
    """
    对数据进行滑动平均或指数加权平均。
    
    参数：
    - data: 需要平滑的数据 (1D array)
    - smooth_factor: 平滑因子，0-1之间，值越大越平滑

    返回：
    - 平滑后的数据
    """
    smoothed = np.zeros_like(data)
    smoothed[0] = data[0]
    for t in range(1, len(data)):
        smoothed[t] = smooth_factor * smoothed[t - 1] + (1 - smooth_factor) * data[t]
    return smoothed

# 绘制根据不同选择绘制的图表（使用Seaborn）
def plot_tensorboard_data(all_seed_data_dict, tags, plot_type='mean_std', save_path=None, names=None, tag_names=None, num_train_steps_per_itr=2000):
    sns.set_theme(style="whitegrid", font_scale=2.0) # 设置Seaborn的样式
    if len(tags) == 6:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
    elif len(tags) == 4:
        fig, axes = plt.subplots(1, 4, figsize=(18, 8))
        axes = axes.flatten()
    elif len(tags) == 2:
        fig, axes = plt.subplots(1, 2, figsize=(24, 8))
        axes = axes.flatten()
    elif len(tags) == 1:
        fig, axes = plt.subplots(1, 1, figsize=(12, 8))
        axes = [axes]
    # fig.suptitle(save_path.split('.')[0])

    # 每个 base_dir 对应一个颜色
    color_palette = sns.color_palette("tab10", len(all_seed_data_dict) // 2 + 1)  # 可选 "Set2" "tab10", "hsv", 或其他高对比度调色板

    # color_palette = sns.color_palette("Set2", 4)

    # base 对应 直线和虚线
    line_styles = ['--', '-']

    for i, tag in enumerate(tags):
        for idx, (base_dir, all_seed_data) in enumerate(all_seed_data_dict.items()):
            # 对不同的seed数据进行对齐
            aligned_data = align_data(all_seed_data, tag)
            
            # 计算均值和方差
            mean, std = calculate_mean_and_std(aligned_data)
            
            mean = np.squeeze(mean)
            std = np.squeeze(std)

            smoothed_mean = smooth_data(mean)
            
            # 使用Seaborn绘制带误差条的曲线图
            sns.lineplot(x=np.arange(len(smoothed_mean)) * (num_train_steps_per_itr / 1000), y=smoothed_mean, ax=axes[i], label=names[idx], color=color_palette[idx // 2], linestyle=line_styles[idx % 2])
            axes[i].fill_between(np.arange(len(smoothed_mean)) * (num_train_steps_per_itr / 1000), smoothed_mean - std, smoothed_mean + std, color=color_palette[idx // 2], alpha=0.1)

        axes[i].set_title(f'{tag_names[i]}', fontsize=30)
        axes[i].set_xlabel(f'Train Steps(1e3)')
        axes[i].set_ylabel('Average Return')
        axes[i].grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.5)
        axes[i].legend(loc='lower right', bbox_to_anchor=(1.0, 0.0), frameon=False)

    plt.tight_layout()

    # 如果提供了保存路径，则保存图片
    if save_path:
        plt.savefig(save_path)
        print(f"图像已保存到：{save_path}")

# 用于执行绘图的接口
def visualize_tensorboard(base_dirs, tags=['tag1', 'tag2'], plot_type='mean_std', save_path=None, names=None, tags_names=None, num_train_steps_per_itr=2000):
    all_seed_data_dict = {}
    
    # 为每个 base_dir 提取数据
    for base_dir in base_dirs:
        all_seed_data = extract_seed_data(base_dir, tags)
        all_seed_data_dict[base_dir] = all_seed_data

    plot_tensorboard_data(all_seed_data_dict, tags, plot_type, save_path, names, tag_names=tags_names, num_train_steps_per_itr=num_train_steps_per_itr)

if __name__ == '__main__':
    # base_dir1 = ['logs_1/point-robot/FOCAL0135/focal_mix_baseline', 'logs_1/point-robot/FOCAL0135/focal_mix_z0_hvar_p10_weighted',
    #              'logs_1/point-robot/CLASSIFIER0349/classifier_mix_baseline', 'logs_1/point-robot/CLASSIFIER0349/classifier_mix_z0_hvar_p10_weighted',
    #              'logs_1/point-robot/UNICORN1237/unicorn_mix_baseline', 'logs_1/point-robot/UNICORN1237/unicorn_mix_z0_hvar_weighted',
    #              'logs_1/point-robot/FOCAL0135/idaq_mix_baseline']

    # base_dir2 = ['logs_1/ant-goal/FOCAL0356/focal_mix_baseline', 'logs_1/ant-goal/FOCAL0356/focal_mix_z0_hvar_p0.1_weighted',
    #              'logs_1/ant-goal/CLASSIFIER0123/classifier_mix_baseline', 'logs_1/ant-goal/CLASSIFIER0123/classifier_mix_z0_hvar_p5_weighted',
    #              'logs_1/ant-goal/UNICORN1234/unicorn_mix_baseline0.1', 'logs_1/ant-goal/UNICORN1234/unicorn_mix_z0_p2_hvar0.1_weighted',
    #              'logs_1/ant-goal/FOCAL0356/idaq_mix_baseline_-650']

    # base_dir3 = ['logs_1/cheetah-vel/FOCAL04710/focal_mix_baseline', 'logs_1/cheetah-vel/FOCAL04710/focal_mix_z0_hvar_p2_weighted',
    #              'logs_1/cheetah-vel/CLASSIFIER_0256/classifier_mix_baseline', 'logs_1/cheetah-vel/CLASSIFIER_0256/classifier_mix_z0_hvar_p10_weighted',
    #              'logs_1/cheetah-vel/UNICORN_0123/unicorn_mix_baseline0.5', 'logs_1/cheetah-vel/UNICORN_0123/unicorn_z0_p0.5_hvar_0.5_weighted',
    #              'logs_1/cheetah-vel/FOCAL04710/idaq_mix_baseline']
    
    # base_dir4 = ['logs_1/walker-rand-params/FOCAL0145/focal_mix_baseline', 'logs_1/walker-rand-params/FOCAL0145/focal_mix_z0_hvar_p5_weighted',
    #              'logs_1/walker-rand-params/CLASSIFIER0124/classifier_mix_baseline', 'logs_1/walker-rand-params/CLASSIFIER0124/classifier_mix_z0_hvar_p25_weighted',
    #              'logs_1/walker-rand-params/UNICORN0123/unicorn_mix_baseline0.5', 'logs_1/walker-rand-params/UNICORN0123/unicorn_mix_z0_p2_hvar0.5_weighted',
    #              'logs_1/walker-rand-params/FOCAL0145/idaq_mix_baseline_r80']
    
    # base_dir5 = ['logs_1/hopper-rand-params/FOCAL0257/focal_mix_baseline', 'logs_1/hopper-rand-params/FOCAL0257/focal_mix_z0_hvar_p0.1_weighted',
    #              'logs_1/hopper-rand-params/CLASSIFIER0124/classifier_mix_baseline', 'logs_1/hopper-rand-params/CLASSIFIER0124/classifier_mix_z0_hvar_p10_weighted',
    #              'logs_1/hopper-rand-params/UNICORN037/unicorn_mix_baseline0.5', 'logs_1/hopper-rand-params/UNICORN037/unicorn_mix_z0_p2_hvar0.5_weighted',
    #              'logs_1/hopper-rand-params/FOCAL0257/idaq_mix_baseline_r140']

    # base_dir_low = ['logs/point-robot/low/focal_low_baseline', 'logs/point-robot/low/focal_low_z0_hvar',
    #              'logs/point-robot/low/classifier_low_baseline', 'logs/point-robot/low/classifier_low_z0_hvar',
    #              'logs/point-robot/low/unicorn_low_baseline', 'logs/point-robot/low/unicorn_low_z0_hvar']

    base_dir_mid = ['logs/point-robot/mid/focal_mid_baseline', 'logs/point-robot/mid/focal_mid_z0_hvar',
                 'logs/point-robot/mid/classifier_mid_baseline', 'logs/point-robot/mid/classifier_mid_z0_hvar',
                 'logs/point-robot/mid/unicorn_mid_baseline', 'logs/point-robot/mid/unicorn_mid_z0_hvar']

    # base_dir_expert = ['logs/point-robot/expert/focal_expert_baseline', 'logs/point-robot/expert/focal_expert_z0_hvar',
                #  'logs/point-robot/expert/classifier_expert_baseline', 'logs/point-robot/expert/classifier_expert_z0_hvar',
                #  'logs/point-robot/expert/unicorn_expert_baseline', 'logs/point-robot/expert/unicorn_expert_z0_hvar']
    
    # base_dirs = [base_dir1, base_dir2, base_dir3, base_dir4, base_dir5]
    # base_dirs = [base_dir_low]
    base_dirs = [base_dir_mid]
    # base_dirs = [base_dir_expert]
    
    names = ['focal', 'focal+ours', 'classifier', 'classifier+ours',  'unicorn', 'unicorn+ours', 'idaq']

    plot_type = 'mean_std'  # 可以选择 'mean_std', 'mean' 或 'std'
    
    for base_dir in base_dirs:
        save_path = f'{base_dir[0].split("/")[1]}'
        num_train_steps_per_itr = 1000
        visualize_tensorboard(base_dir, ['Return/first_OnlineReturn_all_test_tasks'], plot_type, save_path+'1.pdf', names, ["First Episode Test"], num_train_steps_per_itr)
        visualize_tensorboard(base_dir, ['Return/second_OnlineReturn_all_test_tasks'], plot_type, save_path+'2.pdf', names, ["Second Episode Test"], num_train_steps_per_itr)
        # visualize_tensorboard(base_dir, ['Return/first_OnlineReturn_all_train_tasks'], plot_type, save_path+'_train1.pdf', names, ["First Episode Train"], num_train_steps_per_itr)
        # visualize_tensorboard(base_dir, ['Return/second_OnlineReturn_all_train_tasks'], plot_type, save_path+'_train2.pdf', names, ["Second Episode Train"], num_train_steps_per_itr)
        # visualize_tensorboard(base_dir, ['Return/Average_OfflineReturn_all_test_tasks'], plot_type, save_path+'_offline_test.pdf', names, ["Offline Return Test"], num_train_steps_per_itr)
        # visualize_tensorboard(base_dir, ['Return/Average_OfflineReturn_all_train_tasks'], plot_type, save_path+'_offline_train.pdf', names, ["Offline Return Train"], num_train_steps_per_itr)
