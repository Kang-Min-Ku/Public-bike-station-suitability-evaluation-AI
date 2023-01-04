import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from models.LeNet5 import LeNet5
from models.AlexNet import AlexNet
from models.ResNet18 import ResNet18
from utils import set_random_seed
import os
from glob import glob
import cv2
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader


set_random_seed(123)

# =============================== EDIT HERE ===============================
"""
    Build model architecture and do experiment.
"""
# lenet / alexnet / resnet18
model_name = "lenet"
# cifar10 / svhn
dataset = "svhn"

# Hyper-parameters
num_epochs = 32
learning_rate = 0.001
reg_lambda = 0.001
batch_size = 128

test_every = 1
print_every = 1

# batch normalization for ResNet18
use_batch_norm = True
# =========================================================================


def main():

    data_list = glob("crop_map/*.png")
    data_num = len(data_list)
    image_info = dict()

    for data in data_list:

        *_, label, station_id, map_type, zoom, _ = (
            data.replace(".", "_").replace("\\", "_").split("_")
        )

        if f"{station_id}_{zoom}" in image_info:
            image_info[f"{station_id}_{zoom}"]["image"].append(data)
        else:
            image_info[f"{station_id}_{zoom}"] = dict()
            image_info[f"{station_id}_{zoom}"]["image"] = [data]
            image_info[f"{station_id}_{zoom}"]["label"] = label

    dataset = np.zeros((len(image_info), 6, 384, 384))
    label = []
    print(len(image_info))

    for i, data in enumerate(image_info.items()):
        stationId_zoom, info = data[0], data[1]
        img1 = cv2.imread(info["image"][0]) / 255  # scaling
        img2 = cv2.imread(info["image"][0]) / 255
        dataset[i, 0:3, :, :] = np.moveaxis(img1, -1, 0)
        dataset[i, 3:6, :, :] = np.moveaxis(img2, -1, 0)
        label.append(info["label"])

    label = np.array(label)
    print(dataset.shape, label.shape)
    
    # data label encoder
    items = label
    encoder = LabelEncoder()
    encoder.fit(items)
    label = encoder.transform(items)

    # data split train set, test set
    x = torch.tensor(dataset, dtype=torch.float32)
    y = torch.tensor(label, dtype=torch.int64)

    X_train, X_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=1004, shuffle=True
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=1004, shuffle=True
    )

    # 데이터 셋이  나누어진 것에 대한 구조보기
    print(X_train.shape, X_test.shape, X_val.shape)
    print(y_train.shape, y_test.shape, y_val.shape)

    # Dataset 작성

    trainset = TensorDataset(X_train, y_train)
    validset = TensorDataset(X_val, y_val)
    testset = TensorDataset(X_test, y_test)

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=1
    )
    validloader = torch.utils.data.DataLoader(
        validset, batch_size=batch_size, shuffle=False, num_workers=1
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=1
    )

    num_class = 1
    input_channel = 6
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_name == "lenet":
        model = LeNet5(input_channel, num_class, learning_rate, reg_lambda, device)
    elif model_name == "alexnet":
        model = AlexNet(input_channel, num_class, learning_rate, reg_lambda, device)
    elif model_name == "resnet18":
        model = ResNet18(
            input_channel, num_class, learning_rate, reg_lambda, device, use_batch_norm
        )
    model = model.to(device)
    # dataiter = iter(trainloader)
    # images, labels = next(dataiter)
    # print(images)
    # print(labels)
    print(f"Model: {model_name}")
    print("Training Starts...")
    model.train_(trainloader, validloader, num_epochs, test_every, print_every)

    # TEST ACCURACY
    model.restore()
    real_y, pred_y = model.predict(testloader)

    correct = len(np.where(pred_y == real_y)[0])
    total = len(pred_y)
    test_acc = correct / total

    print("Test Accuracy (Top-1) at Best Epoch : %.2f" % (test_acc))

    model.plot_accuracy("del")


if __name__ == "__main__":
    main()
