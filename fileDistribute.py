import json
import os
import random
import shutil
from shutil import copy2
from config import data_source


# Randomly divide the data into train sets and val sets
def main():
    split_rate = [0.7, 0.3]
    split_names = ['train', 'val']

    data_root = os.path.abspath(os.path.join(os.getcwd(), "../../../data_set"))  # get data root path
    image_path = os.path.join(data_root, data_source, 'gray_image')  # flower data set path

    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    image_path1 = os.path.join(image_path, "data_all/")
    for split_name in split_names:
        split_path = image_path + "/" + split_name
        print(split_path)
        if os.path.isdir(split_path):
            pass
        else:
            os.makedirs(split_path)

    # read class_indict
    class_indict = os.listdir(image_path1)
    kind_len = len(class_indict)

    for j in range(kind_len):
        kind_str = class_indict[j]
        kind_path = os.path.join(image_path1, kind_str)

        str_path_train = os.path.join(image_path, "train")
        str_path_val = os.path.join(image_path, "val")
        str_path_test = os.path.join(image_path, "test")
        train_path = os.path.join(str_path_train, kind_str)
        val_path = os.path.join(str_path_val, kind_str)
        test_path = os.path.join(str_path_test, kind_str)

        if os.path.isdir(train_path):
            pass
        else:
            os.makedirs(train_path)
        if os.path.isdir(val_path):
            pass
        else:
            os.makedirs(val_path)
        if os.path.isdir(test_path):
            pass
        else:
            os.makedirs(test_path)

        class_names = os.listdir(kind_path)
        train_path = train_path + '/'
        val_path = val_path + '/'
        test_path = test_path + '/'

        for class_name in class_names:
            current_data_path = kind_path
            current_all_data = os.listdir(current_data_path)
            current_data_length = len(current_all_data)
            current_data_index_list = list(range(current_data_length))
            random.shuffle(current_data_index_list)

            train_stop_flag = current_data_length * split_rate[0]
            val_stop_flag = current_data_length * (split_rate[0] + split_rate[1])

        current_idx = 0
        train_num = 0
        val_num = 0
        test_num = 0

        for i in current_data_index_list:
            src_img_path = os.path.join(current_data_path, current_all_data[i])
            if current_idx <= train_stop_flag:
                copy2(src_img_path, train_path)
                train_num += 1
            elif (current_idx > train_stop_flag) and (current_idx <= val_stop_flag):
                copy2(src_img_path, val_path)
                val_num += 1
            else:
                copy2(src_img_path, test_path)
                test_num += 1
            current_idx += 1
        print("Done!", train_num, val_num, test_num)


if __name__ == '__main__':
    main()
