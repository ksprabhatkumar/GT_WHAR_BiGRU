import torch
import torch.nn as nn
from torch_geometric.nn import GINConv

class BodyNodeAttentionUnit(nn.Module):
    """Implements the Body-Node Attention Branch and Fusion (Fig. 4 & 5)"""
    def __init__(self, in_features, hidden_dim):
        super().__init__()
        # Branch 1: Sensing Aggregation (GIN + BN + ReLU)
        agg_nn = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
        self.gin_agg = GINConv(agg_nn, train_eps=True)
        
        # Branch 2: Node Attention (GIN + Tanh)
        att_nn = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.Tanh()
        )
        self.gin_att = GINConv(att_nn, train_eps=True)

    def forward(self, x, edge_index):
        node_features = self.gin_agg(x, edge_index)
        attention_weights = self.gin_att(x, edge_index)
        # Fusion Branch: Multiply attention weights with features
        return node_features * attention_weights

class GraphTemporalLearningUnitCell(nn.Module):
    """Implements the Custom Graph-GRU Cell (Fig. 6 & Eq. 2-7)"""
    def __init__(self, in_features, hidden_dim):
        super().__init__()
        # Input features to the graph will be concat of X_t and h_{t-1}
        combined_dim = in_features + hidden_dim
        
        # The 3 Graph networks placed before the gates (Gr, Gu, Gc)
        self.Gr = BodyNodeAttentionUnit(combined_dim, hidden_dim)
        self.Gu = BodyNodeAttentionUnit(combined_dim, hidden_dim)
        self.Gc = BodyNodeAttentionUnit(combined_dim, hidden_dim)
        
        # Gate weights
        self.W_r = nn.Linear(hidden_dim, hidden_dim)
        self.W_u = nn.Linear(hidden_dim, hidden_dim)
        self.W_c = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x_t, h_prev, edge_index):
        # Concatenate current input and previous hidden state
        inp = torch.cat([x_t, h_prev], dim=-1)
        
        # Generate graph embeddings for gates
        g_r = self.Gr(inp, edge_index)
        g_u = self.Gu(inp, edge_index)
        
        # Reset and Update Gates (Eq 3 & 4)
        r_t = torch.sigmoid(self.W_r(g_r))
        u_t = torch.sigmoid(self.W_u(g_u))
        
        # Candidate hidden state (Eq 6)
        h_prev_reset = r_t * h_prev
        inp_c = torch.cat([x_t, h_prev_reset], dim=-1)
        g_c = self.Gc(inp_c, edge_index)
        c_t = torch.tanh(self.W_c(g_c))
        
        # Final hidden state (Eq 7)
        h_t = u_t * h_prev + (1 - u_t) * c_t
        return h_t

class GT_WHAR(nn.Module):
    def __init__(self, num_nodes=5, in_features=9, hidden_dim=64, num_classes=19):
        super().__init__()
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        
        edges = [[0, 1], [1, 0], [0, 2], [2, 0], [0, 3], [3, 0], [0, 4], [4, 0]]
        self.base_edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        
        self.forward_cell = GraphTemporalLearningUnitCell(in_features, hidden_dim)
        self.backward_cell = GraphTemporalLearningUnitCell(in_features, hidden_dim)
        
        # Final classifier takes concatenated forward and backward states of ALL nodes
        # --- NEW: Added Dropout to prevent overfitting! ---
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(hidden_dim * 2 * num_nodes, num_classes)

    def get_batched_edge_index(self, batch_size, device):
        edge_index = self.base_edge_index.to(device)
        num_edges = edge_index.size(1)
        batched_edge_index = edge_index.repeat(1, batch_size)
        offset = torch.arange(0, batch_size, device=device) * self.num_nodes
        offset = offset.view(-1, 1).repeat(1, num_edges).view(1, -1)
        return batched_edge_index + offset

    def forward(self, x):
        B, T, N, F = x.shape
        device = x.device
        edge_index = self.get_batched_edge_index(B, device)
        
        # Initialize hidden states
        h_forward = torch.zeros(B * N, self.hidden_dim, device=device)
        h_backward = torch.zeros(B * N, self.hidden_dim, device=device)
        
        # Process Forward Sequence
        for t in range(T):
            x_t = x[:, t, :, :].reshape(B * N, F)
            h_forward = self.forward_cell(x_t, h_forward, edge_index)
            
        # Process Backward Sequence
        for t in range(T - 1, -1, -1):
            x_t = x[:, t, :, :].reshape(B * N, F)
            h_backward = self.backward_cell(x_t, h_backward, edge_index)
            
        # Reshape to [Batch, Nodes * Hidden] and Concatenate
        h_forward = h_forward.view(B, N * self.hidden_dim)
        h_backward = h_backward.view(B, N * self.hidden_dim)
        
        out = torch.cat([h_forward, h_backward], dim=-1)
        
        # --- NEW: Apply Dropout before classifying ---
        out = self.dropout(out) 
        
        return self.classifier(out)