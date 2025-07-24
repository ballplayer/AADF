from torch.utils.data import Dataset
from PIL import Image
import os


class RobotDataset(Dataset):
    def __init__(self, root_dir, img_dir, label_dir, transform=None, target_transform=None):
        self.root_dir = root_dir    # data/robot
        self.img_dir = img_dir      # img/train
        self.label_dir = label_dir  # ann/train
        self.img_path = os.path.join(self.root_dir, self.img_dir)   # data/robot/img/train
        self.img_list = sorted(os.listdir(self.img_path))   # ['000000.jpg', '000001.jpg', ...]
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        img_name = self.img_list[index]
        img_item_path = os.path.join(self.root_dir, self.img_dir, img_name)
        img = Image.open(img_item_path)
        if self.transform:
            img = self.transform(img)

        label_name = img_name.split(".")[0] + ".png"
        label_item_path = os.path.join(self.root_dir, self.label_dir, label_name)
        label = Image.open(label_item_path)
        if self.target_transform:
            label = self.target_transform(label)
        return img, label

    def __len__(self):
        return len(self.img_list)


if __name__ == '__main__':
    root_dir = "data/robot"
    image_dir = "img/train"
    label_dir = "ann/train"
    dataset = RobotDataset(root_dir, image_dir, label_dir)
    print(dataset[0])
