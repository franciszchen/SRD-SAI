import torch
import torch.nn as nn
import torch.nn.functional as F


class DualUniform_2Optim_Fnorm(nn.Module):
    def __init__(self):
        super(DualUniform_2Optim_Fnorm, self).__init__()

    def forward(self, f_var, f_fix):
        return self.dual_loss(f_var, f_fix)

    def dual_loss(self, f_var, f_fix):
        """
        输入是两个支路的A方阵，来自于SAI
        接下来对两个A按行进行softmax进行soften
        两个A求差后，再计算F范数
        """
        bsz = f_var.shape[0]
        
        F_var = torch.nn.functional.softmax(f_var, dim=1)
        F_fix = torch.nn.functional.softmax(f_fix, dim=1)

        F_diff = F_fix - F_var
        loss = (F_diff * F_diff).view(-1, 1).sum(0) / (bsz * bsz)
        return loss


