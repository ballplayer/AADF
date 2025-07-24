import os
import random
import shutil
import torch
import torchvision.datasets
from torchvision import transforms


# 这种分割方法需要先通过PyTorch读取所有数据为tensor格式，然后再进行分割
def dataset_split_pytorch(src_data_folder):
    transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
    dataset = torchvision.datasets.ImageFolder(root=src_data_folder, transform=transform)
    full_data_size = len(dataset)
    print("总数据集的长度为：{}".format(full_data_size))
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size, val_size])
    print("训练数据集的长度为：{}".format(len(train_dataset)))
    print("验证数据集的长度为：{}".format(len(val_dataset)))
    print("测试数据集的长度为：{}".format(len(test_dataset)))


# 这种分割方法直接通过文件夹进行分割，不需要先读取数据
def dataset_split_folder(dataset_path):
    classes = os.listdir(dataset_path)

    # 创建 train 文件夹
    os.mkdir(os.path.join(dataset_path, 'train'))
    # 创建 val 文件夹
    os.mkdir(os.path.join(dataset_path, 'val'))
    # 创建 test 文件夹
    os.mkdir(os.path.join(dataset_path, 'test'))
    # 在 train 和 val 文件夹中创建各类别子文件夹
    for cls in classes:
        os.mkdir(os.path.join(dataset_path, 'train', cls))
        os.mkdir(os.path.join(dataset_path, 'val', cls))
        os.mkdir(os.path.join(dataset_path, 'test', cls))

    train_frac = 0.8  # 训练集比例
    val_frac = 0.1  # 验证集比例
    # random.seed(123) # 随机数种子，便于复现

    print('{:^18} {:^18} {:^18} {:^18}'.format('类别', '训练集数据个数', '验证集数据个数', '验证集数据个数'))

    for cls in classes:  # 遍历每个类别
        # 读取该类别的所有图像文件名
        old_dir = os.path.join(dataset_path, cls)
        images_filename = os.listdir(old_dir)
        random.shuffle(images_filename)  # 随机打乱

        # 划分训练集、验证集、测试集
        trainset_numer = int(len(images_filename) * train_frac)  # 训练集图像个数
        valset_numer = int(len(images_filename) * val_frac)  # 验证集图像个数
        # tesetset_numer = len(images_filename) - trainset_numer - valset_numer  # 测试集图像个数

        trainset_images = images_filename[:trainset_numer]  # 获取拟移动至 train 目录的验证集图像文件名
        valset_images = images_filename[trainset_numer:trainset_numer+valset_numer]  # 获取拟移动至 val 目录的验证集图像文件名
        testset_images = images_filename[trainset_numer+valset_numer:]  # 获取拟移动至 test 目录的训练集图像文件名

        # 移动图像至 train 目录
        for image in trainset_images:
            old_img_path = os.path.join(dataset_path, cls, image)  # 获取原始文件路径
            new_train_path = os.path.join(dataset_path, 'train', cls, image)  # 获取 train 目录的新文件路径
            shutil.move(old_img_path, new_train_path)  # 移动文件

        # 移动图像至 val 目录
        for image in valset_images:
            old_img_path = os.path.join(dataset_path, cls, image)  # 获取原始文件路径
            new_val_path = os.path.join(dataset_path, 'val', cls, image)  # 获取 val 目录的新文件路径
            shutil.move(old_img_path, new_val_path)  # 移动文件

        # 移动图像至 val 目录
        for image in testset_images:
            old_img_path = os.path.join(dataset_path, cls, image)  # 获取原始文件路径
            new_val_path = os.path.join(dataset_path, 'test', cls, image)  # 获取 val 目录的新文件路径
            shutil.move(old_img_path, new_val_path)  # 移动文件

        # 删除旧文件夹
        assert len(os.listdir(old_dir)) == 0  # 确保旧文件夹中的所有图像都被移动走
        shutil.rmtree(old_dir)  # 删除文件夹

        # 工整地输出每一类别的数据个数
        print('{:^18} {:^18} {:^18} {:^18}'.format(cls, len(trainset_images), len(valset_images), len(valset_images)))

    # 重命名数据集文件夹
    # shutil.move(dataset_path, dataset_name + '_split')


if __name__ == '__main__':
    src_data_folder = '../datasets/hymenoptera_data2'
    # dataset_split_pytorch(src_data_folder)
    dataset_split_folder(src_data_folder)
