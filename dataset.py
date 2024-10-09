from numpy import dtype
from torch.utils.data import Dataset
from torch.nn.functional import one_hot
from PIL import Image
import torch
import os

# 设置CaptchaDataset继承Dataset，用于读取验证码数据
class CaptchaDataset(Dataset):
    # init函数用于初始化
    # 函数传入数据的路径data_dir和数据转换对象transform
    # 将验证码使用的字符集characters，通过参数传入
    def __init__(self, data_dir, transform, characters):
        self.file_list = list() #保存每个训练数据的路径
        # 使用os.listdir，获取data_dir中的全部文件
        files = os.listdir(data_dir)
        for file in files: #遍历files
            # 将path添加到file_list列表
            self.file_list.append(file)
        self.data_dir = data_dir
        # 将数据转换对象transform保存到类中
        self.transform = transform
        self.characters = characters

        # 创建一个字符到数字的字典
        self.char2int = {}
        # 在创建字符到数字的字典时，使用外界传入的字符集characters
        for i, char in enumerate(characters):
            self.char2int[char] = i

    def __len__(self):
        # 直接返回数据集中的样本数量
        # 重写该方法后可以使用len(dataset)语法，来获取数据集的大小
        return len(self.file_list)

    # 函数传入索引index，函数应当返回与该索引对应的数据和标签
    # 通过dataset[i]，就可以获取到第i个样本了
    def __getitem__(self, index):
        file = self.file_list[index] #获取数据的路径
        # 打开文件，并使用convert('L')，将图片转换为灰色
        # 不需要通过颜色来判断验证码中的字符，转为灰色后，可以提升模型的鲁棒性
        image = Image.open(os.path.join(self.data_dir, file)).convert('L')
        # 使用transform转换数据，将图片数据转为张量数据
        image = self.transform(image)
        # 获取该数据图片中的字符标签
        label = file.split('_')[0]
        return image, self.getOneHotFromLabels(label) #返回image和label

    def getOneHotFromLabels(self, label):
        labelSensor = torch.tensor([self.char2int[ch] for ch in label], dtype=torch.long)
        return one_hot(labelSensor, len(self.characters)).view(-1,)
    
    def getLabelFromOneHot(self, onehot):
        labels = onehot.view(-1, len(self.characters)).argmax(dim=1)
        return "".join([self.characters[charIndex] for charIndex in labels])
        
from torch.utils.data import DataLoader
from torchvision import transforms
import json

if __name__ == '__main__':
  
    with open("config.json", "r") as f:
        config = json.load(f)

    height = config["resize_height"]  # 图片的高度
    width = config["resize_width"]  # 图片的宽度
    # 定义数据转换对象transform
    # 使用transforms.Compose定义数据预处理流水线
    # 在transform添加Resize和ToTensor两个数据处理操作
    transform = transforms.Compose([
        transforms.Resize((height, width)),  # 将图片缩放到指定的大小
        transforms.ToTensor()])  # 将图片数据转换为张量

    data_path = config["test_data_path"]  # 训练数据储存路径
    characters = config["characters"]  # 验证码使用的字符集
    batch_size = config["batch_size"]
    epoch_num = config["epoch_num"]

    # 定义CaptchaDataset对象dataset
    dataset = CaptchaDataset(data_path, transform, characters)
    # 定义数据加载器data_load
    # 其中参数dataset是数据集
    # batch_size=8代表每个小批量数据的大小是8
    # shuffle = True表示每个epoch，都会随机打乱数据的顺序
    data_load = DataLoader(dataset,
                           batch_size = batch_size,
                           shuffle = True)

    # 使用dataloader对数据进行遍历
    # batch_idx表示当前遍历的批次
    # data和label表示这个批次的训练数据和标记
    for batch_idx, (data, label) in enumerate(data_load):
        print(f"batch_idx = {batch_idx}, data = {data.shape}, label = {label.shape}")
