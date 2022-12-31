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
    # 读取有多少kind
    data_root = os.path.abspath(os.path.join(os.getcwd(), "../../../data_set"))  # get data root path
    file_str = os.path.join(data_root, data_source)  # flower data set path
    kind_file = file_str + "/test/"
    kind_names = os.listdir(kind_file)
    kind_len = len(kind_names)

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

    # create model
    model = resnet34(num_classes=kind_len).to(device)

    # load model weights
    # weights_path = file_str+"resNet34.pth"
    weights_path = file_str + weights_file_name
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    model.load_state_dict(torch.load(weights_path, map_location=device))

    # prediction
    model.eval()

    for str_kind in kind_names:
        imag_file_path = kind_file + str_kind
        datanames = os.listdir(imag_file_path)
        dataname = list(datanames)
        data_len = len(dataname)

        # load image
        cnt = 0
        for i in range(data_len):
            img_path = imag_file_path + "/" + dataname[i]
            assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
            img = Image.open(img_path)

            # [N, C, H, W]
            img = data_transform(img)
            # expand batch dimension
            img = torch.unsqueeze(img, dim=0)

            with torch.no_grad():
                # predict class
                output = torch.squeeze(model(img.to(device))).cpu()
                predict = torch.softmax(output, dim=0)
                predict_cla = torch.argmax(predict).numpy()

                print_res = "file:{:10}   class: {}   prob: {:.3}".format(dataname[i], class_indict[str(predict_cla)],
                                                                          predict[predict_cla].numpy())

                if str_kind == class_indict[str(predict_cla)]:
                    cnt += 1

        # print(cnt)
        print("kind_name:{} acc={}/{}={:.3}".format(str_kind, cnt, data_len, cnt / data_len))


if __name__ == '__main__':
    main()
