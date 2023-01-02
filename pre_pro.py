# Calculate the correct rate of snr dataset
import os
import json

import matplotlib
import torch
from PIL import Image
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from model import resnet34
from config import data_source


def main():
    # 读取有多少kind
    data_root = os.path.abspath(os.path.join(os.getcwd(), "../../../data_set"))
    file_str = os.path.join(data_root, data_source, 'var_snr')  # flower data set path
    file_names = os.listdir(file_str)
    file_len = len(file_names)
    save_recoder_txt = os.path.join(data_root, data_source, 'snr_recoder.txt')
    data_transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    with open(save_recoder_txt, 'w', encoding='utf-8') as f:  # 使用with open()新建对象f
        for file_name in file_names:
            assert os.path.exists(file_str), "{} path does not exist.".format(file_str)
            validate_dataset = datasets.ImageFolder(root=os.path.join(file_str, file_name))
            val_num = len(validate_dataset)
            validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                          batch_size=1, shuffle=False,
                                                          num_workers=2)
            # read class_indict
            json_path = './class_indices.json'
            assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)
            json_file = open(json_path, "r")
            class_indict = json.load(json_file)

            kind_len = len(class_indict)

            # create model
            model = resnet34(num_classes=kind_len).to(device)
            print(kind_len)

            # load model weights
            # weights_path = file_str+"resNet34.pth"
            weights_path = os.path.join(data_root, data_source, "resNet34.pth")
            assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
            model.load_state_dict(torch.load(weights_path, map_location=device))
            print(weights_path)

            # prediction
            model.eval()

            # 开始验证
            acc = 0.0
            with torch.no_grad():
                val_bar = tqdm(validate_loader, colour='green')
                for val_data in val_bar:
                    val_images, val_labels = val_data
                    outputs = model(val_images.to(device))
                    # loss = loss_function(outputs, test_labels)
                    predict_y = torch.max(outputs, dim=1)[1]
                    acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

            val_accurate = acc / val_num
            process_recoder = file_name.split('-')[1] + ' ' + str(val_accurate)
            f.write(process_recoder + '\n')

if __name__ == '__main__':
    main()
