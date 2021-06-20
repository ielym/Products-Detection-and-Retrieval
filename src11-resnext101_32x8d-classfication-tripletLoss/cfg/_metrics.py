import torch

def top1_accuracy(y_pred, y_true, topk=1):
    with torch.no_grad():
        maxk = topk
        batch_size = y_true.size(0)

        _, pred = y_pred.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(y_true.view(1, -1).expand_as(pred))

        correct_k = correct[:topk].view(-1).float().sum(0, keepdim=True)
        res = correct_k.mul_(100.0 / batch_size)
        return res

def topk_accuracy(y_pred, y_true, topk=5):
    with torch.no_grad():
        maxk = topk
        batch_size = y_true.size(0)

        _, pred = y_pred.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(y_true.view(1, -1).expand_as(pred))

        correct_k = correct[:topk].view(-1).float().sum(0, keepdim=True)
        res = correct_k.mul_(100.0 / batch_size)
        return res

def accuracy(y_pred, y_true):
    with torch.no_grad():
        y_pred = torch.argmax(y_pred, dim=1)
        y_true = torch.argmax(y_true, dim=1)

        batch_size = y_true.size(0)
        correct = y_pred.eq(y_true.expand_as(y_pred)).float().sum(0, keepdim=True)
        acc = correct * 100.0 / batch_size
        return acc