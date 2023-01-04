from torch.utils.data import Subset
import numpy as np
import sklearn.metrics as metric

def split_data(dataset, train_ratio, valid_ratio, test_ratio, seed_number=42):
    assert train_ratio+valid_ratio+test_ratio == 1
    
    if seed_number is not None:
        np.random.seed(seed_number)
    num_data = len(dataset)
    num_train = int(train_ratio * num_data)
    num_valid = int(valid_ratio * num_data)
    num_test = int(test_ratio * num_data)
    
    perm = np.random.permutation(num_data)
    train_idx = perm[:num_train]
    valid_idx = perm[num_train:num_train+num_valid]
    test_idx = perm[num_train+num_valid:]

    train_dataset = Subset(dataset, train_idx)
    valid_dataset = Subset(dataset, valid_idx)
    test_dataset = Subset(dataset, test_idx)

    return train_dataset, valid_dataset, test_dataset

def evaluation_metric(pred, correct):
    eval_metric = {}
    eval_metric["confusion_matrix"]=metric.confusion_matrix(correct, pred)
    eval_metric["accuracy"]=metric.accuracy_score(correct, pred)
    eval_metric["precision"]=metric.precision_score(correct, pred)
    eval_metric["recall"]=metric.recall_score(correct, pred)
    eval_metric["f1_score"]=metric.f1_score(correct, pred)
    
    return eval_metric