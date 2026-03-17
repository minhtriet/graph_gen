import torch
from torch import nn



class GCNLayer(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.projection = nn.Linear(c_in, c_out)

    def forward(self, node_feeats, adj_matrix):
        e_st, e_end = b.edges[:,0], b.edges[:,1]
        dists = torch.norm(b.x[e_st] - b.x[e_end], dim=1).reshape(-1, 1)
        
        # compute messages
        tmp = torch.hstack([b.h[e_st], b.h[e_end], dists])
        m_ij = self.f_e(tmp)
        
        # predict edges
        e_ij = self.f_inf(m_ij)
        
        # average e_ij-weighted messages  
        # m_i is num_nodes x hid_dim
        m_i = index_sum(b.h.shape[0], e_ij*m_ij, b.edges[:,0], self.cuda)
        
        # update hidden representations
        b.h += self.f_h(torch.hstack([b.h, m_i]))

