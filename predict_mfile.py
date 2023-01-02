import os
import json

import matplotlib
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')

from model import resnet34
from config import data_source, weights_file_name


def main():
    set_name = 'test-noisy'
    # 读取有多少kind
    data_root = os.path.abspath(os.path.join(os.getcwd(), "../../../data_set"))  # get data root path
    file_str = os.path.join(data_root, data_source)  # flower data set path
    kind_file = file_str + "/" + set_name
    list_names = os.listdir(kind_file)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # read class_indict
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)
    json_file = open(json_path, "r")
    class_indict = json.load(json_file)
    kind_len = len(class_indict)
    print(kind_len)
    # create model
    model = resnet34(num_classes=kind_len).to(device)

    # load model weights
    # weights_path = file_str+"resNet34.pth"
    weights_path = file_str + weights_file_name
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    model.load_state_dict(torch.load(weights_path, map_location=device))

    # prediction
    model.eval()

    save_recoder_txt = os.path.join(data_root, data_source, set_name+"_"+'result.txt')

    with open(save_recoder_txt, 'w', encoding='utf-8') as f:  # 使用with open()新建对象f
        for list_name in list_names:
            imag_file_path = os.path.join(kind_file, list_name)
            img_path = imag_file_path
            assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
            img=Image.open(img_path).convert('RGB')
            print(img_path)

            # [N, C, H, W]
            img = data_transform(img)
            # expand batch dimension
            img = torch.unsqueeze(img, dim=0)

            with torch.no_grad():
                # predict class
                output = model(img.to(device))
                predict_y = torch.max(output, dim=1)[1]

                process_recoder = list_name.split('.')[0] + '.flac' + ' ' + class_indict[str(int(predict_y))]
                f.write(process_recoder + '\n')


if __name__ == '__main__':
    main()
