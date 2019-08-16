import torch


def l2(y_pred, y_true):
    return torch.sqrt(((y_pred - y_true) ** 2).sum())


def acc(y_pred, y_true):
    return (y_pred.argmax(dim=1) == y_true.argmax(dim=1)).sum().type(
        torch.FloatTensor
    ) / y_true.size(0)


def iou_acc(y_pred, y_true, threshold=0.5):
    y_pred_byte = y_pred > threshold
    y_true_byte = y_true > threshold

    union = torch.add(y_pred_byte, y_true_byte).sum()
    interception = torch.mul(y_pred_byte, y_true_byte).sum()

    if union == 0:
        if interception == 0:
            return 1
        else:
            return 0

    return interception / union
