import torch
import torch.nn.functional as F


def l2(y_pred, y_true):
    return torch.sqrt(((y_pred - y_true) ** 2).sum())


def acc(y_pred, y_true):
    return (y_pred.argmax(dim=1) == y_true.argmax(dim=1)).sum().type(
        torch.FloatTensor
    ) / y_true.size(0)


def iou_acc(y_pred, y_true, threshold=0.02):
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


class DiceLoss(torch.nn.Module):
    def __init__(self, base_weight=0.5, dice_weight=0.5, base_loss=F.mse_loss):
        super(DiceLoss, self).__init__()

        self.base_weight = base_weight
        self.dice_weight = dice_weight

        self.base_loss = base_loss

        self.smooth = 1.0

    def dice_loss(self, y_pred, y_true):
        product = torch.mul(y_pred, y_true)
        intersection = torch.sum(product)
        coefficient = (2.0 * intersection + self.smooth) / (
                    torch.sum(y_pred) + torch.sum(y_true) + self.smooth)
        return -coefficient + 1

    def forward(self, y_pred, y_true):
        return self.base_weight * self.base_loss(y_pred, y_true) + \
               self.dice_weight * self.dice_loss(y_pred, y_true)
