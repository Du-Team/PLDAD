import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
import torch.nn as nn

class GAT(torch.nn.Module):

    def __init__(self, num_nodes, seq_len, num_levels=1, device=torch.device('cuda:0'), 
                 hidden_channels=16, dropout=0.5):
        super(GAT, self).__init__()
        self.num_nodes = num_nodes
        self.seq_len = seq_len
        self.device = device
        self.num_levels = num_levels
        self.hidden_channels = hidden_channels
        self.dropout = dropout

        self.conv1 = GATConv(seq_len, hidden_channels)
        self.conv2 = GATConv(hidden_channels, seq_len)

        source_nodes = torch.arange(num_nodes, dtype=torch.long).repeat(num_nodes)
        target_nodes = torch.arange(num_nodes, dtype=torch.long).repeat_interleave(num_nodes)
        base_edge_index = torch.stack([source_nodes, target_nodes], dim=0)

        self.register_buffer('edge_index', base_edge_index, persistent=False)
        self._batched_edge_index_cache = {}

    def _get_batched_edge_index(self, batch_size, device):
        cached = self._batched_edge_index_cache.get(batch_size)
        if cached is not None and cached.device == device:
            return cached

        base_edge = self.edge_index.to(device=device, non_blocking=True)
        offsets = (torch.arange(batch_size, device=device, dtype=torch.long) * self.num_nodes).view(batch_size, 1, 1)
        batched_edge_index = (base_edge.unsqueeze(0) + offsets).permute(1, 0, 2).reshape(2, -1).contiguous()
        self._batched_edge_index_cache[batch_size] = batched_edge_index
        return batched_edge_index

    def forward(self, x):

        batch_size, seq_len, num_nodes = x.shape

        x = x.permute(0, 2, 1).contiguous()  
        x = x.view(-1, seq_len)  

        edge_index = self._get_batched_edge_index(batch_size, x.device)

        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)

        x = x.view(batch_size, num_nodes, seq_len)
        x = x.permute(0, 2, 1)  

        return x
