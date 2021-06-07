import torch.nn as nn


class DiceLoss(nn.Module):

    def __init__(self):
        super(DiceLoss, self).__init__()
        self.smooth = 1.0

    def forward(self, y_pred, y_true):
        assert y_pred.size() == y_true.size()
        y_pred_bone = y_pred[:, 0].contiguous().view(-1)
        y_true_bone = y_true[:, 0].contiguous().view(-1)
        intersection_bone = (y_pred_bone * y_true_bone).sum()
        dsc_bone = (2. * intersection_bone + self.smooth) / (
            y_pred_bone.sum() + y_true_bone.sum() + self.smooth
        )
        y_pred_fat = y_pred[:, 1].contiguous().view(-1)
        y_true_fat = y_true[:, 1].contiguous().view(-1)
        intersection_fat = (y_pred_fat * y_true_fat).sum()
        dsc_fat = (2. * intersection_fat + self.smooth) / (
                y_pred_fat.sum() + y_true_fat.sum() + self.smooth
        )
        return 2. - dsc_bone - dsc_fat
