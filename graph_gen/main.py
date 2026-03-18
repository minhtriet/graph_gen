import torch
from torch import nn


class ECNN(nn.Module):
    def __init__(self, M, n):
        """
        The definition are from Appendix C https://arxiv.org/pdf/2102.09844
        M: Number of nodes
        n: dimension of the node
        """
        super().__init__()
        self.h = nn.Embedding(M, n)
        # edge function
        self.phi_e = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.SiLU(),
            nn.Linear(hidden_features, out_features),
            nn.SiLU()
        )
        # coordinate function
        self.phi_x = nn.Sequential(
            nn.Linear(hidden_features, out_features),
            nn.SiLU(),
            nn.Linear(hidden_features, out_features),
        )
        self.phi_h = nn.Sequential(
            nn.Linear(hidden_features, out_features),
            nn.SiLU(),
            nn.Linear(hidden_features, out_features),
        )
        complete_graph = torch.one((M, M))
        self_loop = torch.I(M)
        self.all_neighbors = complete_graph - self_loop
    
    def forward(self, batch):
        """
        batch: [:, x, a]
        a: edge attribute
        x: coordinate embedding
        h: node embedding
        M : number of nodes
        """
        C = 1 / (self.M - 1)
        # encoder
        dist = torch.norm(batch(x[i] - x[j]))
        m_ij = self.phi_e(self.h)
        x_i = xi + C*(x_i - x_j)*self.phi_x(m_ij)
        m_i = m_ij @ self.all_neighbors
        h = phi_h(h_i, m_i)
        # decoder
        pass



class RefGCNLayer(nn.Module):
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

