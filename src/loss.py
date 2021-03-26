import torch
import torch.nn as nn

class DistributionLoss(nn.Module):

    def __init__(self):
        super().__init__()

        self.kl_loss = nn.KLDivLoss(reduction='batchmean')

    def forward(self, src, tgt, eps=1e-9):
        """
        Compute the distribution loss given output and target distributions

        Arguments:
        src -- tensor of softmax predicted output classes (B, C)
        tgt -- tensor of target output classes (C,)
        eps -- small epsilon value for KL 

        Returns:
        loss -- KL divegence loss
        """
        assert len(src.shape) == 2 and \
               len(tgt.shape) == 1 and \
               src.shape[1] == tgt.shape[0]

        batch_prob = src.mean(dim=0)

        # nn.KLDivLoss accepts the input tensor as log probabilities
        batch_logprob = (batch_prob + eps).log()
        loss = self.kl_loss(batch_logprob, tgt)

        return loss

