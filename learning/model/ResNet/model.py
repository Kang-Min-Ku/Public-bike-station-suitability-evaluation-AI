import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import os

def load_torch_hub(model_name, is_pretrain=False):
    return torch.hub.load("pytorch/vision:v0.10.0", model_name, pretrained=is_pretrain)

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
    return correct, pred

#BCELoss() / Adam()
def train(  model,
            train_loader,
            valid_loader,
            loss_function,
            optimizer,
            num_epochs,
            device,
            test_every,
            model_name):

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
            train_y = train_y.to(device)
            pred_y = model(train_x)

            loss = loss_function(pred_y, train_y)
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
            print(f"[Epoch {epoch}] loss: {np.round(epoch_loss,4)}")
            #train acc
            train_acc = total_correct / total_predict
            train_accuracy.append(train_acc)
            #valid acc
            valid_y, valid_pred_y = predict(model, valid_loader, device)
            valid_correct = (valid_y == valid_pred_y).sum().item()
            valid_total = len(valid_pred_y)
            valid_acc = valid_correct / valid_total
            valid_accuracy.append(valid_acc)

            print(f"Train accuracy = {np.round(train_acc,4)} // Valid accuracy = {np.round(valid_acc,4)}")

            if best_acc < valid_acc:
                print(f"Best model changed at epoch {epoch}")
                best_acc = valid_acc
                best_epoch = epoch
                #create dir
                try:
                    if not os.path.exists("best_model"):
                        os.makedirs("best_model")
                except OSError:
                    print("OS Error creating best_model")
                #save_best_model
                torch.save(model.state_dict(), f"./best_model/{model_name}_best.pt")
    print("Training finishied..")
    print(f"Best valid acc = {np.round(best_acc,4)} at epoch {best_epoch}")

    return best_acc

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