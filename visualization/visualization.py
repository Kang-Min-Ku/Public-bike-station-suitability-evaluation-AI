import numpy as np
import cv2
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import data

def get_convs(model):
    conv_list = []
    children_layers = model.children()
    for layer in children_layers:
        if isinstance(layer, nn.Conv2d):
            if layer.kernel_size == (1,1):
                continue
            conv_list.append(layer)
        elif len(list(layer.children())) > 0:
            conv_list.append(get_convs(layer))
    return flatten_list(conv_list)

def is_list_in_list(lst):
    return sum([isinstance(element, list) for element in lst]) > 0

def flatten_list(lst):
    flat_list = []
    for element in lst:
        flat_element = []
        if isinstance(element, list):
            flat_element = flatten_list(element)
            flat_list += flat_element
        else:
            flat_list.append(element)
    return flat_list

def convert_3d_to_2d(tensor):
    assert len(tensor.shape) == 3
    gray_scale = torch.mean(tensor, 0)
    return gray_scale

def plot_features(file, img, conv_layers, interest_layer=None, num_cols=3):
    #img is numpy array
    count = 0
    outputs = []

    transformaton = data.compose_transform(img.shape[-1], (img.shape[0], img.shape[1]))
    img = torch.unsqueeze(transformaton(img), 0)
    outputs.append(torch.squeeze(img))

    for layer in conv_layers:
        img = layer(img)
        outputs.append(torch.squeeze(img))

    feature_map = [convert_3d_to_2d(output) for output in outputs]

    if interest_layer is None:
        interest_layer = range(len(feature_map))
    
    num_plots = len(feature_map)
    num_rows = 1 + num_plots//num_cols
    fig = plt.figure(figsize=(num_cols, num_rows))

    for idx in interest_layer:
        count += 1
        ax1 = fig.add_subplot(num_rows, num_cols, count)
        img = feature_map[idx].detach().numpy().astype(np.float32)
        img = (img - np.mean(img))/np.std(img)
        img = np.minimum(1, np.maximum(0, (img + 0.5)))
        ax1.imshow(img)
        ax1.set_title(f"{idx}")
        ax1.axis('off')
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])

    plt.tight_layout()
    #plt.show()
    plt.savefig(file)

def plot_filters_single_channel(file, conv_layers, interest_dim=None, interest_channel=None, num_cols = 12):
    count = 0
    
    conv_tensor = conv_layers.weight
    channel_dim, num_channel, kernel_width, kernel_height = conv_tensor.shape
    conv_array = conv_tensor.detach().numpy()

    if interest_dim is None:
        interest_dim = range(channel_dim)

    if interest_channel is None:
        interest_channel = range(num_channel)

    num_plots = len(interest_dim) * len(interest_channel)
    num_rows = 1 + num_plots//num_cols
    fig = plt.figure(figsize=(num_cols, num_rows))

    for i in interest_dim:
        for j in interest_channel:
            count += 1
            ax1 = fig.add_subplot(num_rows, num_cols, count)
            img = conv_array[i,j].astype(np.float32)
            img = (img - np.mean(img))/np.std(img)
            img = np.minimum(1, np.maximum(0, (img + 0.5)))
            ax1.imshow(img)
            ax1.set_title(f"{i},{j}")
            ax1.axis('off')
            ax1.set_xticklabels([])
            ax1.set_yticklabels([])

    plt.tight_layout()
    #plt.show()
    plt.savefig(file)



if __name__ == "__main__":
    import model as m
    model = m.load_torch_hub("resnet18", True)
    convs = get_convs(model)
    convs = flatten_list(convs)
    plot_filters_single_channel("visual.png", convs[14], [2,3,4], [1,3,5,6,10,11,12])
    cat = cv2.imread("cat106.jpg")
    plot_features("feature.png", cat, convs)
