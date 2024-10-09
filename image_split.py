from ast import main
import os
from tkinter.tix import MAIN
from PIL import Image
import torch
from torchvision import transforms

transform = transforms.Compose([transforms.ToTensor()])
index = 1

def makeImage(filename):
    image = Image.open(os.path.join("./source",filename))
    label = filename.split("_")[0]
    imageData = transform(image)
    imageDatas = [imageData[:,:,i:i+10] for i in range(0, 40, 10)]
    for i in range(4):
        for j in range(4):
            for m in range(4):
                for n in range(4):
                    newfile = f"{label[i]}{label[j]}{label[m]}{label[n]}_{index}.bmp"
                    tensor = torch.concat([imageDatas[i],imageDatas[j],imageDatas[m],imageDatas[n]],dim=2)
                    image = transforms.ToPILImage()(tensor)
                    image.save("./images/"+newfile)


if __name__ == "__main__":
    for filename in os.listdir("./source"):
        print(f"filename: {filename}")
        makeImage(filename)