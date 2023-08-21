import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dnc import DNC
import math

'''
Our model
'''
def elementswise_join(l1, l2, l3):
    result = [[x,y,z] for x, y, z in zip(l1, l2, l3)]
    flat_list = [item for sublist in result for item in sublist]
    return flat_list


class GraphAttentionV2Layer(nn.Module):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            n_heads: int,
            is_concat: bool = True,
            dropout: float = 0.6,
            leaky_relu_negative_slope: float = 0.2,
            share_weights: bool = False,
            is_output_layer: bool = False,
    ) -> None:
        super().__init__()
        self.is_concat = is_concat
        self.n_heads = n_heads
        self.share_weights = share_weights
        self.is_output_layer = is_output_layer

        if is_concat:
            assert out_features % n_heads == 0
            self.n_hidden = out_features // n_heads
        else:
            self.n_hidden = out_features

        # Linear layer for initial source transformation;
        # i.e. to transform the source node embeddings before self-attention
        self.linear_l = nn.Linear(in_features, self.n_hidden * n_heads, bias=False)
        # If `share_weights` is `True` the same linear layer is used for the target nodes
        if share_weights:
            self.linear_r = self.linear_l
        else:
            self.linear_r = nn.Linear(in_features, self.n_hidden * n_heads, bias=False)

        # Linear layer to compute attention score $e_{ij}$
        self.attn = nn.Linear(self.n_hidden, 1, bias=False)
        # The activation for attention score $e_{ij}$
        self.activation = nn.LeakyReLU(negative_slope=leaky_relu_negative_slope)
        # Softmax to compute attention $\alpha_{ij}$
        self.softmax = nn.Softmax(dim=1)
        # Dropout layer to be applied for attention
        self.dropout = nn.Dropout(dropout)

        self.output_act = nn.ELU()
        self.output_dropout = nn.Dropout(dropout)

    # def forward(self, h: torch.Tensor, adj_mat: torch.Tensor, use_einsum=True) -> torch.Tensor:
    def forward(self, h: torch.Tensor, adj_mat: torch.Tensor, use_einsum=True) -> torch.Tensor:
        """
        * `h`, "h" is the input node embeddings of shape `[n_nodes, in_features]`.
        * `adj_mat` is the adjacency matrix of shape `[n_nodes, n_nodes, n_heads]`.
        We use shape `[n_nodes, n_nodes, 1]` since the adjacency is the same for each head.
        Adjacency matrix represent the edges (or connections) among nodes.
        `adj_mat[i][j]` is `True` if there is an edge from node `i` to node `j`.
        """
        # Number of nodes
        n_nodes = h.shape[0]

        g_l = self.linear_l(h).view(n_nodes, self.n_heads, self.n_hidden)
        g_r = self.linear_r(h).view(n_nodes, self.n_heads, self.n_hidden)

        # First, calculate g_li * g_rj for all pairs of i and j
        g_l_repeat = g_l.repeat(n_nodes, 1, 1)
        g_r_repeat_interleave = g_r.repeat_interleave(n_nodes, dim=0)

        # combine g_l and g_r
        g_sum = g_l_repeat + g_r_repeat_interleave
        g_sum = g_sum.view(n_nodes, n_nodes, self.n_heads, self.n_hidden)

        # calculate attention score e_ij
        e = self.attn(self.activation(g_sum)).squeeze(-1)

        # The adjacency matrix should have shape
        # `[n_nodes, n_nodes, n_heads]` or`[n_nodes, n_nodes, 1]`
        # print(adj_mat.shape)
        # print(n_nodes)
        assert adj_mat.shape[0] == 1 or adj_mat.shape[0] == n_nodes
        assert adj_mat.shape[1] == 1 or adj_mat.shape[1] == n_nodes
        assert adj_mat.shape[2] == 1 or adj_mat.shape[2] == self.n_heads

        # mask e_ij based on the adjacency matrix
        e = e.masked_fill(adj_mat == 0, float('-inf'))

        # apply softmax to calculate attention
        a = self.softmax(e)
        # apply dropout
        a = self.dropout(a)

        #
        # calculate the final output for each head
        #
        if use_einsum:
            h_prime = torch.einsum('ijh,jhf->ihf', a, g_r)
        else:
            h_prime = torch.matmul(a, g_r)

        # concatenate the output of each head
        if self.is_concat:
            # print('h_prime', np.shape(h_prime))
            h_prime = h_prime.reshape(n_nodes, -1)
        else:
            h_prime = torch.mean(h_prime, dim=1)

        # do not apply activation and dropout for the output layer
        if self.is_output_layer:
            return h_prime
        # apply activation and dropout
        return self.output_dropout(self.output_act(h_prime))

class GATV2(nn.Module):
    def __init__(
            self,
            in_features: int,
            n_hidden: int,
            n_classes: int,
            adj,
            device=torch.device('cpu:0'),
            n_heads: int = 4,
            dropout: float = 0.4,
            num_of_layers: int = 2,
            share_weights: bool = True
    ) -> None:
        """
        * `in_features` is the number of features per node
        * `n_hidden` is the number of features in the first graph attention layer
        * `n_classes` is the number of classes
        * `n_heads` is the number of heads in the graph attention layers
        * `dropout` is the dropout probability
        * `num_of_layers` is the number of graph attention layers
        * `share_weights` if set to True, the same matrix will be applied to the source and the target node of every edge
        """
        super().__init__()
        self.num_of_layers = num_of_layers
        self.device = device
        self.x = torch.eye(in_features).to(device)
        self.adj = torch.FloatTensor(adj).to(device)
        self.layers = nn.ModuleList()

        # add input layer
        self.layers.append(GraphAttentionV2Layer(in_features, n_hidden, n_heads, is_concat=True, dropout=dropout,
                                                 share_weights=share_weights))

        # add hidden layers
        for i in range(num_of_layers - 2):
            self.layers.append(GraphAttentionV2Layer(n_hidden, n_hidden, 1, share_weights=share_weights))

        # # add output layer
        # self.layers.append(
        #     GraphAttentionV2Layer(n_hidden, n_classes, 1, is_concat=False, dropout=dropout, share_weights=share_weights,
        #                           is_output_layer=True))

    # def forward(self, x: torch.Tensor, adj_mat: torch.Tensor) -> torch.Tensor:
    def forward(self):
        """
        * `x` is the features vectors of shape `[n_nodes, in_features]`
        * `adj_mat` is the adjacency matrix of the form
         `[n_nodes, n_nodes, n_heads]` or `[n_nodes, n_nodes, 1]`
        """
        x = self.x
        adj = torch.reshape(self.adj,(np.shape(self.adj)[0],np.shape(self.adj)[1],1))
        print(np.shape(x))
        for i in range(self.num_of_layers-1):
            x = self.layers[i](x, adj)
            print('x size', np.shape(x))
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, V)
        return output

    def split_heads(self, x):
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x):
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)

    def forward(self, Q, K, V, mask=None):
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))

        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        output = self.W_o(self.combine_heads(attn_output))
        return output

class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        print('x', np.shape(x))
        print(np.shape(self.pe[:, :x.size(1)]))
        return x + self.pe[:, :x.size(1)]


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_output = self.self_attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x

class REFINE(nn.Module):
    def __init__(self, vocab_size, ehr_adj, ddi_adj, emb_dim=64, num_heads=2, num_layers=2, d_ff=256, max_seq_length=1000, dropout=0.1, device=torch.device('cpu:0'), ddi_in_memory=True):
        super(REFINE, self).__init__()
        K = len(vocab_size)
        # vocab_size[0] is diag, vocab_size[1] is med, vocab_size[2] is lab test
        ''' Length here is 3 (3 types of codes)'''
        self.K = K
        self.vocab_size = vocab_size
        self.device = device
        self.tensor_ddi_adj = torch.FloatTensor(ddi_adj).to(device)
        self.tensor_dco_adj = torch.FloatTensor(ehr_adj).to(device)
        self.ddi_in_memory = ddi_in_memory
        self.embeddings_d = nn.Embedding(vocab_size[0], emb_dim)
        self.embeddings = nn.ModuleList(
            [nn.Linear(vocab_size[i+1]*3, emb_dim) for i in range(K - 1)])
        print("structure", self.embeddings[0])
        self.dropout = nn.Dropout(p=0.4)
        # self.positional_encoding = PositionalEncoding(emb_dim, max_seq_length)
        self.positional_encoding = nn.ModuleList(
            [PositionalEncoding(emb_dim, max_seq_length) for _ in range(K)])
        # self.encoder_layers = nn.ModuleList(
        #     [EncoderLayer(emb_dim, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.encoder_layers_d = nn.ModuleList(
            [EncoderLayer(emb_dim, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.encoder_layers_r = nn.ModuleList(
            [EncoderLayer(emb_dim, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.encoder_layers_s = nn.ModuleList(
            [EncoderLayer(emb_dim, num_heads, d_ff, dropout) for _ in range(num_layers)])


        self.ehr_gat = GATV2(in_features=vocab_size[1], n_hidden=emb_dim, adj=ehr_adj, n_classes=vocab_size[1], device=device)
        self.ddi_gat = GATV2(in_features=vocab_size[1], n_hidden=emb_dim, adj=ddi_adj, n_classes=vocab_size[1], device=device)
        self.inter = nn.Parameter(torch.FloatTensor(1))

        self.output = nn.Linear(emb_dim, vocab_size[1])
        self.init_weights()

    def forward(self, input):
        # input (adm, 3, codes)

        # generate medical embeddings and queries
        i1_seq = []
        i2_seq = []
        i3_seq = []
        def mean_embedding(embedding):
            return embedding.mean(dim=1).unsqueeze(dim=0)  # (1,1,dim)
        for idx, adm in enumerate(input):
            i1 = mean_embedding(self.dropout(self.embeddings_d(torch.LongTensor(adm[0]).unsqueeze(dim=0).to(self.device)))) # (1,1,dim)
            """ [1,1,64]"""
            print(type(adm[0]))
            print('i1', np.shape(i1))
            r_1d = elementswise_join(adm[2], adm[3], adm[4])
            print(np.shape(r_1d))
            i2 = self.dropout(self.embeddings[1](torch.FloatTensor(r_1d).unsqueeze(dim=0).to(self.device)))
            i2 = torch.reshape(i2,(1,1,64))
            print('i2', np.shape(i2))
            if len(input) > 1 and idx < len(input)-1:
                s_1d = elementswise_join(adm[6], adm[7], adm[8])
                print('s_1d', np.shape(s_1d))
                i3 = self.dropout(self.embeddings[0](torch.FloatTensor(s_1d).unsqueeze(dim=0).to(self.device)))
                i3 = torch.reshape(i3, (1, 1, 64))
                i3_seq.append(i3)
                print('i3', np.shape(i3))
            # i2 = mean_embedding(self.dropout(self.embeddings[1](torch.LongTensor(adm[1]).unsqueeze(dim=0).to(self.device))))
            i1_seq.append(i1)
            i2_seq.append(i2)

        i1_seq = torch.cat(i1_seq, dim=1) #(1,seq,dim)
        i2_seq = torch.cat(i2_seq, dim=1) #(1,seq,dim)
        print('i2_seq', np.shape(i2_seq))
        if i3_seq != []:
            i3_seq = torch.cat(i3_seq, dim=1)  # (1,seq,dim)
            print('i3_seq', np.shape(i3_seq))


        # Transformer module

        d_embedded = self.dropout(self.positional_encoding[0](i1_seq))
        d_enc_output = d_embedded
        for enc_layer_d in self.encoder_layers_d:
            d_enc_output = enc_layer_d(d_enc_output)
        l_embedded = self.dropout(self.positional_encoding[1](i2_seq))
        l_enc_output = l_embedded
        for enc_layer_r in self.encoder_layers_r:
            l_enc_output = enc_layer_r(l_enc_output)
        if i3_seq != []:
            m_embedded = self.dropout(self.positional_encoding[2](i3_seq))
            m_enc_output = m_embedded
            for enc_layer_s in self.encoder_layers_s:
                m_enc_output = enc_layer_s(m_enc_output)

        print(np.shape(d_enc_output))
        print(np.shape(l_enc_output))
        # if i3_seq != []:
        #     print(np.shape(m_enc_output))

        if i3_seq != []:
            print(np.size(l_enc_output))
            # print(np.size(m_enc_output))
            zero_pad = torch.zeros(1, 1, 64)
            m_enc_output = torch.cat([zero_pad,m_enc_output], dim=1)
            print('concat', np.shape(m_enc_output))
            query = d_enc_output + l_enc_output + m_enc_output
        else:
            query = d_enc_output + l_enc_output
        print('q_before', np.shape(query))
        #  Drug co-occurrence and Drug severity graph encoding
        query = query.squeeze(dim=0)
        q = query[-1:]  # (1,dim)
        Z = self.ehr_gat() - self.ddi_gat() * self.inter  # (size, dim)
        print('drug memory', np.shape(Z))
        print('q', np.shape(q))
        # '''O:read from global memory bank and dynamic memory bank'''
        print(torch.mm(q, Z.t()))
        lambda_wt = F.softmax(torch.mm(q, Z.t()), dim=-1)  # (1, size)
        print('lambda', lambda_wt)
        weighted_druginfo = torch.mm(lambda_wt, Z)  # (1, dim)
        # '''R:convert O and predict'''
        # output = self.output(torch.cat([query, fact1, fact2], dim=-1)) # (1, dim)
        print('weighted_druginfo', weighted_druginfo)
        output = self.output(q + weighted_druginfo)
        print('output', output)
        if self.training:
            pred_prob = F.sigmoid(output)
            beta = 0.65
            # print('neg_pred_prob', np.shape(neg_pred_prob))
            pred_prob = pred_prob.t() * pred_prob  # (voc_size, voc_size)
            diff = (self.tensor_ddi_adj * beta) - (self.tensor_dco_adj * (1-beta))
            balanced_loss = pred_prob.mul(diff).mean()
            # print('ddi_adj',np.shape(self.tensor_ddi_adj))
            # return output, batch_neg
            return output,balanced_loss
        else:
            return output

    def init_weights(self):
        """Initialize weights."""
        initrange = 0.1
        for item in self.embeddings:
            item.weight.data.uniform_(-initrange, initrange)

        self.inter.data.uniform_(-initrange, initrange)




