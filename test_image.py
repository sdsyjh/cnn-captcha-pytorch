from model2 import CNNModel2
from dataset import CaptchaDataset
from PIL import Image

import torch
import torchvision.transforms as transforms

import json

if __name__ == '__main__':
    with open("config.json", "r") as f:
        config = json.load(f)

    height = config["resize_height"]  # 图片的高度
    width = config["resize_width"]  # 图片的宽度

    # 定义数据转换对象transform
    # 将图片缩放到指定的大小，并将图片数据转换为张量
    transform = transforms.Compose([
        transforms.Resize((height, width)),
        transforms.ToTensor()])

    data_path = config["test_data_path"]  # 训练数据储存路径
    characters = config["characters"]  # 验证码使用的字符集
    digit_num = config["digit_num"]
    class_num = len(characters) * digit_num
    test_model_path = config["test_model_path"]

    print("resize_height = %d" % (height))
    print("resize_width = %d" % (width))
    print("data_path = %s" % (data_path))
    print("characters = %s" % (characters))
    print("digit_num = %d" % (digit_num))
    print("class_num = %d" % (class_num))
    print("test_model_path = %s" % (test_model_path))
    print("")

    dataset = CaptchaDataset(data_path, transform, characters)

    # 定义设备对象device，这里如果cuda可用则使用GPU，否则使用CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # 创建一个CNNModel模型对象，并转移到GPU上
    model = CNNModel2(height, width, digit_num, class_num).to(device)
    # 调用load_state_dict，读取已经训练好的模型文件captcha.digit
    test_model_path = "./model/check.epoch70"
    model.load_state_dict(torch.load(test_model_path, weights_only=True))
    model.eval()
    image = Image.open("./data/test-digit/0Ok5_778.jpg").convert('L')
        # 使用transform转换数据，将图片数据转为张量数据
    image = transform(image).to(device)
    output = model(image.unsqueeze(dim=0))
    print(f"code = {dataset.getLabelFromOneHot(output[0])}")
