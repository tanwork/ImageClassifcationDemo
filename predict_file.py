import torch
from model import resnet34
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from torchvision import transforms, datasets
import json
import os

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    kind_str = "crispy"
    kind_num = 0

    # load image
    # image_path = "D:/DeepLearning/ImageClassification/ResNet/flowers"
    image_path = "../../data_set/process1/test/"
    predict_dataset = datasets.ImageFolder(root=image_path,
                                           transform=data_transform)

    # pre_num = len(predict_dataset)
    predict_loader = torch.utils.data.DataLoader(predict_dataset, batch_size=1,
                                                 shuffle=False, num_workers=0)
    # 获取文件名
    datanames_list = []
    datanames = os.listdir(os.path.join(image_path, kind_str))
    for i in datanames:
        datanames_list.append(i)

    # read class_indict
    try:
        json_file = open('./class_indices.json', 'r')
        class_indict = json.load(json_file)
    except Exception as e:
        print(e)
        exit(-1)

    # create model
    model = resnet34(num_classes=3).to(device)
    # load model weights
    weights_path = "../../data_set/resNet34(0.905).pth"
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    model.load_state_dict(torch.load(weights_path, map_location=device))

    # predition
    model.eval()

    all_num = len(datanames_list)
    acc = 0

    with torch.no_grad():
        i = 0
        while i < all_num:
            for pre_data in predict_loader:
                pre_images, pre_labels = pre_data

                outputs = torch.squeeze(model(pre_images.to(device))).cpu()

                img_path = image_path+kind_str+"/" + datanames_list[i]

                predict = torch.softmax(outputs, dim=0)
                predict_cla = torch.argmax(predict).numpy()
                print_res = datanames_list[i] + " is " + class_indict[str(predict_cla)] + \
                            ". The probability is {:.2f}%".format(predict[predict_cla].item() * 100)

                # 显示图片
                # print(img_path)
                # img = mpimg.imread(img_path)
                # plt.imshow(img)
                # plt.axis('off')
                # plt.title(print_res)
                # plt.show()

                if predict_cla == kind_num:
                    acc += 1

                i = i + 1

    print(acc/all_num)

if __name__ == '__main__':
    main()
