import torch
import torch.nn as nn
from torch.utils.data import Subset, Dataset, DataLoader
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import os
import sklearn
from sklearn.model_selection import KFold
import copy

#resnet18/34/50/101/152
def load_torch_hub(model_name, is_pretrain=False):
    return torch.hub.load("pytorch/vision:v0.10.0", model_name, weights=is_pretrain)

#only work for pretrained resnet model #나중에 수정
def fit_model_structure(model, in_channel, out_features):
    model.conv1 = nn.Sequential(
    nn.Conv2d(  in_channels=in_channel,
                out_channels=model.conv1.out_channels,
                kernel_size=(1,1),
                stride=(2,2),
                bias=model.conv1.bias
                ),
    nn.Conv2d(  in_channels=model.conv1.out_channels,
                out_channels=model.conv1.out_channels,
                kernel_size=model.conv1.kernel_size,
                stride=model.conv1.stride,
                padding=model.conv1.padding,
                bias=model.conv1.bias
                )
    )

    model.fc = nn.Sequential(
        nn.Linear(in_features=model.fc.in_features,out_features=out_features),
        nn.Sigmoid()
    )
    reset_model_parameters(model)
    return model

def reset_model_parameters(model):
    for layer in model.children():
        if hasattr(layer, "reset_parameters"):
            layer.reset_parameters()

def predict(model, test_loader, device):
    model.eval()
    correct = []
    pred = []
    with torch.no_grad():
        for test_x, test_y in tqdm(test_loader):
            test_x = test_x.to(device)
            pred_y = model(test_x)

            pred_label = pred_y
            pred_label[pred_label >= 0.5] = 1
            pred_label[pred_label < 0.5] = 0

            correct.append(test_y.numpy())
            pred.append(pred_label.cpu().numpy())
    correct = np.concatenate(correct, axis=0)
    pred = np.concatenate(pred, axis=0)
    return pred, correct

#BCELoss() / Adam()
#check weight change
def train(  model,
            train_loader,
            valid_loader,
            loss_function,
            optimizer,
            num_epochs,
            device,
            test_every,
            model_name
    ):

    model = model.to(device)
    train_accuracy = []
    valid_accuracy = []
    best_acc = 0.0
    best_epoch = 0

    total_predict = 0
    total_correct = 0
    model.train()
    for epoch in range(1, num_epochs+1):
        epoch_loss = 0.
        for _ , (train_x, train_y) in enumerate(tqdm(train_loader, desc="Training")):
            train_x = train_x.to(device)
            pred_y = model(train_x).to(torch.float64)
            #print(train_y.dtype, pred_y.dtype)
            loss = loss_function(pred_y, train_y.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #logistic
            pred_label = pred_y
            pred_label[pred_label >= 0.5] = 1
            pred_label[pred_label < 0.5] = 0
            #store batch result
            epoch_loss += loss
            total_predict += train_y.size(0)
            total_correct += (pred_label.cpu() == train_y).sum().item()
        
        epoch_loss /= len(train_loader)

        if epoch % test_every == 0:
            print("[Epoch %d] loss: %.3f" % (epoch, epoch_loss))
            #train acc
            train_acc = total_correct / total_predict
            train_accuracy.append(train_acc)
            #valid acc
            valid_y, valid_pred_y = predict(model, valid_loader, device)
            valid_correct = (valid_y == valid_pred_y).sum().item()
            valid_total = len(valid_pred_y)
            valid_acc = valid_correct / valid_total
            valid_accuracy.append(valid_acc)

            print("Train accuracy = %.3f // Valid accuracy = %.3f" % (train_acc, valid_acc))

            if best_acc < valid_acc:
                print("Best model changed at epoch %d" % epoch)
                best_acc = valid_acc
                best_epoch = epoch
                #create dir
                try:
                    if not os.path.exists("best_model"):
                        os.makedirs("best_model")
                except OSError:
                    print("OS Error creating best_model")
                #save_best_model
                print(model)
                torch.save(model.state_dict(), f"./best_model/{model_name}_best.pt")

    try:
        if not os.path.exists("plot"):
            os.makedirs("plot")
    except OSError:
        print("OS Error creating plot")
    plot_accuracy(f"./plot/{model_name}_accuracy.png",train_accuracy, valid_accuracy, num_epochs, test_every)

    print("Training finishied..")
    print("Best valid acc = %.3f at epoch %d" % (best_acc, best_epoch))

    return best_acc

def cross_validation(n_splits,
                     model,
                     train_dataset,
                     batch_size,
                     loss_function,
                     optimizer,
                     num_epochs,
                     device,
                     test_every,
                     shuffle=True,
                     random_seed=42
    ):
    
    cross_accuracy = []
    kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_seed)
    for idx, (train_idx, valid_idx) in enumerate(kf.split(train_dataset)):
        print(f"cross validation {idx}")
        model_name = f"cross_validation_{idx}"
        model_cross = copy.deepcopy(model)
        
        cross_train_dataset = Subset(train_dataset, train_idx)
        train_loader = DataLoader(cross_train_dataset, batch_size=batch_size, shuffle=shuffle)
        cross_valid_dataset = Subset(train_dataset, valid_idx)
        valid_loader = DataLoader(cross_valid_dataset, batch_size=batch_size, shuffle=shuffle)

        acc = train(model_cross,
                    train_loader,
                    valid_loader,
                    loss_function,
                    optimizer,
                    num_epochs,
                    device,
                    test_every,
                    model_name)
        
        cross_accuracy.append(acc)

    return cross_accuracy
    

def restore(model, file_name):
    with open(os.path.join(file_name), 'rb') as fd:
        state_dict = torch.load(fd)
    return model.load_state_dict(state_dict)

def plot_accuracy(file_name, train_acc, test_acc, num_epochs, test_every):
    epochs = list(np.arange(1, num_epochs+1, test_every, dtype=np.int32))

    plt.figure()
    plt.plot(epochs, train_acc, label='Train Acc.')
    plt.plot(epochs, test_acc, label='Valid Acc.')

    plt.title('Epoch - Train/Valid Acc.')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.savefig(file_name)
    #plt.show()    