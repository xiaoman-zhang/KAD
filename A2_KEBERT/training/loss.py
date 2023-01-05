import torch
import torch.nn as nn
from torch.nn import functional as F

try:
    import torch.distributed.nn
    from torch import distributed as dist
    has_distributed = True
except ImportError:
    has_distributed = False

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None


class ClipLoss(nn.Module):
    def __init__(
            self,
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            use_horovod=False,
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod

        # cache state
        self.prev_num_logits = 0
        self.labels = {}

    def get_labels(self,cui_lists,device):
        #caculate labels
        num_logits = len(cui_lists)
        labels = torch.eye(num_logits, device=device, dtype=torch.float)
        for i in range(num_logits):
            search_keywords = cui_lists[i]
            for j in range(i,num_logits):
                match_keywords = cui_lists[j]
                if search_keywords == match_keywords and search_keywords[0] == 'C':
                    labels[i,j]=1
                    labels[j,i]=1
        labels = F.normalize(labels,dim=0)
        return labels
    
    def SoftCrossEntropy(self,inputs, target, reduction='average'):
        log_likelihood = -F.log_softmax(inputs, dim=1)
        batch = inputs.shape[0]
        if reduction == 'average':
            loss = torch.sum(torch.mul(log_likelihood, target)) / batch
        else:
            loss = torch.sum(torch.mul(log_likelihood, target))
        return loss

    def forward(self, text1_features, text2_features, cui_lists, logit_scale):
        device = text1_features.device
        if self.world_size > 1:
            all_text1_features, all_text2_features = gather_features(
                text1_features, text2_features,
                self.local_loss, self.gather_with_grad, self.rank, self.world_size, self.use_horovod)

            if self.local_loss:
                logits_per_text1 = logit_scale * text1_features @ all_text2_features.T
                logits_per_text2 = logit_scale * text2_features @ all_text1_features.T
            else:
                logits_per_text1 = logit_scale * all_text1_features @ all_text2_features.T
                logits_per_text2 = logits_per_text1.T
        else:
            logits_per_text1 = logit_scale * text1_features @ text2_features.T
            logits_per_text2 = logit_scale * text2_features @ text1_features.T
        
        # calculated ground-truth and cache if enabled
        num_logits = logits_per_text1.shape[0]
        labels = self.get_labels(cui_lists,device)

        total_loss = (
            self.SoftCrossEntropy(logits_per_text1, labels) +
            self.SoftCrossEntropy(logits_per_text2, labels)
            ) / 2
        return total_loss
