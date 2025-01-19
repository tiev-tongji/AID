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
    max_steps = max(seed_steps)
    
    aligned_data = []

    for seed_folder, seed_data in all_seed_data.items():
        values = seed_data[tag]
        # 如果数据长度不一致，填充缺失的部分
        for value in values:
            if len(value) < max_steps:
                padding = np.full((max_steps - len(value),), np.nan)  # 用NaN填充
                aligned_value = np.concatenate([value, padding])
            else:
                aligned_value = value
            aligned_data.append(aligned_value)

    return np.array(aligned_data)

# 绘制根据不同选择绘制的图表（使用Seaborn）
def plot_tensorboard_data(all_seed_data_dict, tags, plot_type='mean_std', save_path=None):
    # sns.set_theme(style="whitegrid")  # 设置Seaborn的样式
    sns.set_theme(style="white") 
    # fig, axes = plt.subplots(len(tags), figsize=(10, len(tags) * 5))
    if len(tags) == 6:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    elif len(tags) == 4:
        fig, axes = plt.subplots(1, 4, figsize=(18, 8))
    fig.suptitle(save_path.split('.')[0])
    
    if len(tags) == 1:
        axes = [axes]  # 保证axes始终是一个数组
    axes = axes.flatten()

    # 每个 base_dir 对应一个颜色
    color_palette = sns.color_palette("Set2", len(all_seed_data_dict))

    for i, tag in enumerate(tags):
        for idx, (base_dir, all_seed_data) in enumerate(all_seed_data_dict.items()):
            # 对不同的seed数据进行对齐
            aligned_data = align_data(all_seed_data, tag)
            
            # 计算均值和方差
            mean, std = calculate_mean_and_std(aligned_data)
            
            mean = np.squeeze(mean)
            std = np.squeeze(std)
            
            # 使用Seaborn绘制带误差条的曲线图
            sns.lineplot(x=np.arange(len(mean)), y=mean, ax=axes[i], label=base_dir.split('/')[-1], color=color_palette[idx])
            axes[i].fill_between(np.arange(len(mean)), mean - std, mean + std, color=color_palette[idx], alpha=0.3)

        axes[i].set_title(f'{tag.split("/")[-1]}')
        axes[i].legend(loc='lower right', bbox_to_anchor=(1.0, 0.0), frameon=False)

    plt.tight_layout()

    # 如果提供了保存路径，则保存图片
    if save_path:
        plt.savefig(save_path)
        print(f"图像已保存到：{save_path}")

# 用于执行绘图的接口
def visualize_tensorboard(base_dirs, tags=['tag1', 'tag2'], plot_type='mean_std', save_path=None):
    all_seed_data_dict = {}
    
    # 为每个 base_dir 提取数据
    for base_dir in base_dirs:
        all_seed_data = extract_seed_data(base_dir, tags)
        all_seed_data_dict[base_dir] = all_seed_data

    plot_tensorboard_data(all_seed_data_dict, tags, plot_type, save_path)

if __name__ == '__main__':
    # base_dirs = ['logs/cheetah-vel/classifier_mix_baseline', 'logs/cheetah-vel/classifier_mix_z0_hvar_weighted']
    base_dirs = ['logs/cheetah-vel/focal_mix_baseline', 'logs/cheetah-vel/focal_mix_z0_hvar_weighted']
    # base_dirs = ['logs/cheetah-vel/focal_mix_z0_hvar_mean', 'logs/cheetah-vel/focal_mix_z0_hvar_min', 'logs/cheetah-vel/focal_mix_z0_hvar_weighted'] # 消融实验 mean min
    # base_dirs = ['logs/cheetah-vel/focal_mix_z0_hvar_mean', 'logs/cheetah-vel/focal_mix_z0_hvar_weighted'] # 消融实验 mean
    # base_dirs = ['logs/cheetah-vel/focal_mix_z0_hvar_mean', 'logs/cheetah-vel/focal_mix_z0_hvar_weighted'] # 消融实验 mean
    # base_dirs = ['logs/point-robot/idaq_mix_baseline', 'logs/point-robot/focal_mix_z0_hvar_weighted'] # IDAQ
    tags = [
        # 'Return/Average_OfflineReturn_all_test_tasks', 'Return/Average_OfflineReturn_all_train_tasks',
            'Return/first_OnlineReturn_all_test_tasks', 'Return/first_OnlineReturn_all_train_tasks',
            'Return/second_OnlineReturn_all_test_tasks', 'Return/second_OnlineReturn_all_train_tasks']
    # tags = ['Return/Average_OnlineReturn_all_test_tasks']
    plot_type = 'mean_std'  # 可以选择 'mean_std', 'mean' 或 'std'
    name = base_dirs[0].split("/")
    save_path = f'{name[1]}_{name[2].split("_")[0]}.png'
    visualize_tensorboard(base_dirs, tags, plot_type, save_path)
