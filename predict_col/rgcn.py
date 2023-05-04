import torch
import torch.nn as nn


class RGCNLayer(nn.Module):
    def __init__(self, hidden_dim, relation_num, dropout=0.1):
        super(RGCNLayer, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.relation_num = relation_num
        
        self.proj = nn.Linear(2 + self.hidden_dim, self.hidden_dim)
        self.ln = nn.LayerNorm(self.hidden_dim)
        self.W = nn.Parameter(torch.rand(hidden_dim, 3, hidden_dim))
        self.V = nn.Parameter(torch.rand(relation_num, 3))
        self.dropout = dropout

        nn.init.xavier_normal_(self.W)
        nn.init.xavier_normal_(self.V)

    def forward(self, columns, logits, adj):
        # [B, H], [B, L, H], [B, L, L]
        x = self.proj(torch.cat([columns, logits], dim=-1))
        x = self.ln(x)
        x = nn.Dropout(p=self.dropout)(nn.GELU()(x))

        bsz = x.size(0)
        supports = [
            torch.bmm(torch.eye(x.size(1)).repeat(bsz, 1, 1).type_as(x), x).unsqueeze(1)
        ]
        for i in range(1, self.relation_num):
            a = (adj == i).type_as(x)
            supports.append(
                torch.bmm(a, x).unsqueeze(1)
            )
        supports = torch.cat(supports, dim=1)  # [B, relation_num, L, H]

        output = torch.matmul(self.V, self.W).permute(1, 0, 2)  # [R, H, H]
        output = torch.matmul(supports, output)     # [B, R, L, H] x [R, H, H] = [B, R, L, H]
        output = torch.sum(output, dim=1)           # [B, L, H]
        output = nn.Dropout(p=self.dropout)(output)

        return output


class RGCN(nn.Module):
    def __init__(self, encoder, relation_num, num_layers=2, dropout=0.1, device=0):
        super(RGCN, self).__init__()
        
        self.device = device
        self.encoder = encoder
        self.hidden_dim = encoder.config.hidden_size
        self.relation_num = relation_num
        
        self.ln = nn.LayerNorm(self.hidden_dim).to(device)
        self.gcn_layers = nn.ModuleList([
            RGCNLayer(self.hidden_dim, relation_num, dropout).to(device) for i in range(num_layers)
        ])
        self.proj = nn.Linear(self.hidden_dim, 2).to(device)
        
    def forward(self, inputs):
        columns = inputs['columns']     # [B, L, N]
        adj = inputs['adj']             # [B, L, L]
        
        bsz, column_num, column_len = columns['input_ids'].size()
        columns = {
            k: v.view(-1, column_len) for k, v in columns.items()
        }
        
        encoder_result = self.encoder(**columns, output_hidden_states=True)
        logits = encoder_result.logits.view(bsz, column_num, -1).to(self.device)     # [B, L, 2]
        columns_embedding = self.ln(encoder_result.hidden_states[1].to(self.device)).view(bsz, column_num, column_len, -1)      # [B, L, N, H]
        columns_embedding = torch.mean(columns_embedding, dim=2)
        
        for layer in self.gcn_layers:
            columns_embedding = layer(columns_embedding, logits, adj)
        
        output = self.proj(columns_embedding)
        return nn.LogSoftmax(dim=-1)(output)
