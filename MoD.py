import torch.nn as nn
import torch.nn.functional as F
import torch

def add_MoD(model, n_embd, is_train=False, skip_factor=0.5):
    # adding MoD to every other layer in transformer
    new_layers = torch.nn.ModuleList()
    for i, layer in enumerate(model.transformer.h):
        if i % 2 == 0:
            new_layer = MoD(skip_factor, n_embd, layer, is_train)
        else:
            new_layer = layer
        new_layers.append(new_layer)

    model.transformer.h = new_layers
    return model

class MoD(nn.Module):
    def __init__(self, skip_factor, dim, layer, is_train=False):
        super().__init__()

        self.skip_factor = skip_factor
        self.dim = dim
        self.layer = layer
        self.is_train = is_train

        # linear projection for token weighting
        self.router = nn.Linear(self.dim, 1)


    def forward(self, x, **kwargs):

        batch, seq_len, dim = x.shape

        # getting logits
        logits = self.router(x)

        k_value = int(seq_len*self.skip_factor)
        if k_value ==0: # edge case while decoding first token
            k_value = 1

        # selecting topk seq logits for token selection for self attention
        weights, idx = torch.topk(logits, k=k_value, dim=1, sorted=False)
        tokens, idx = torch.sort(idx, dim=1)
        filter_x = torch.gather(x, dim=1, index=tokens.expand(-1,-1, dim)) # batch, k, hidden_dim

        # attention and layer norm as passed into the class
        out = self.layer(filter_x) # batch, k, hidden_dim

        # softmax ruins the causality of the layer as it peeks in ahead in the sequence
        # it can be taken care with aux loss or MLP predictor as mentioned in paper, using sigmoid is another option
        # we focus on training convergence
        tok_weights = F.softmax(weights, dim=1)

        router_weights = torch.gather(tok_weights, dim=1, index=idx)
        xw_out = router_weights * out

        out = torch.scatter_add(x, dim=1, index=tokens.expand(-1,-1,dim), src=xw_out)
        return out
