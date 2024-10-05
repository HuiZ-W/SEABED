import os
import shutil
import glob
import random

# 定义数据集的路径
dataset_dir = '/home/GED_Process/NeuralGED/data/newdata/yago/raw_data'
train_dir = os.path.join(dataset_dir, 'train')
val_dir = os.path.join(dataset_dir, 'val')
test_dir = os.path.join(dataset_dir, 'test')

# 获取所有json文件
json_files = glob.glob(os.path.join(dataset_dir, '*.json'))
random.shuffle(json_files)

# 计算训练集、验证集和测试集的大小
num_files = len(json_files)
num_train = int(num_files * 0.6)
num_val = int(num_files * 0.2)

# 分割文件列表
train_files = json_files[:num_train]
val_files = json_files[num_train:num_train + num_val]
test_files = json_files[num_train + num_val:]

# 创建目录函数
def create_dir_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# 创建训练集、验证集和测试集目录
create_dir_if_not_exists(train_dir)
create_dir_if_not_exists(val_dir)
create_dir_if_not_exists(test_dir)

# 移动文件的函数
def move_files(files, destination):
    for file in files:
        shutil.move(file, destination)

# 将文件移动到相应的目录
move_files(train_files, train_dir)
move_files(val_files, val_dir)
move_files(test_files, test_dir)

print("数据集分割完成。")
