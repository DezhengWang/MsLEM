import torch
from torch import nn
from torch.nn import functional as F
from collections import OrderedDict
from einops import rearrange

from models.MsLEM.embed import DSW_embedding
from models.MsLEM.tools import Transpose


class Encoder(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1, dim_feedforward=None, activation=F.relu, batch_first=True):
        super(Encoder, self).__init__()
        if dim_feedforward is None:
            dim_feedforward = d_model * 4
        self.encoder = nn.TransformerEncoderLayer(d_model=d_model,
                                                  nhead=nhead,
                                                  dim_feedforward=dim_feedforward,  #
                                                  dropout=dropout,
                                                  activation=activation,
                                                  batch_first=batch_first
                                                  )

        self.norm_ttn = nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(d_model), Transpose(1, 2))

    def forward(self, data):
        b = data.shape[0]
        src = rearrange(data, 'b f l d -> (b f) l d')
        src = self.encoder(src)
        src = self.norm_ttn(src)
        src = rearrange(src, '(b f) l d -> b f l d', b=b)
        return src


class RBFSVM(nn.Module):
    def __init__(self, gamma=0.5):
        super(RBFSVM, self).__init__()
        self.alpha = torch.nn.Parameter(torch.randn(32), requires_grad=True)
        self.bias = torch.nn.Parameter(torch.zeros(1), requires_grad=True)
        self.gamma = gamma
        self.sigmoid = nn.Sigmoid()

    def rbf_kernel(self, X1, X2):
        dist = torch.cdist(X1, X2, p=2) ** 2
        return torch.exp(-self.gamma * dist)

    def forward(self, x):
        K = self.rbf_kernel(x, x)
        output = torch.mv(K, self.alpha[:x.shape[0]]) + self.bias
        output = self.sigmoid(output)
        return output


class CAST(nn.Module):
    def __init__(self, args, **factory_kwargs):
        super(CAST, self).__init__()
        self.seq_length = args.seq_length
        self.d_model = args.d_model
        self.num_en_layers = args.num_layers
        self.enc_feature = args.ndata
        self.pred_len = args.nclass
        self.num_globalfeature = args.nfeature
        dropout = args.dropout
        self.nhead = 8
        self.output_attention = False

        self.seg_len = getattr(args, 'seg_len', 20)  # default 6

        self.en_encoding = DSW_embedding(self.seg_len, self.d_model)
        self.enc_pos_embedding = nn.Parameter(torch.randn(1, self.enc_feature, (self.seq_length // self.seg_len), self.d_model))
        self.pre_norm = nn.LayerNorm(self.d_model)

        self.dropout = nn.Dropout(dropout)

        en_layers = OrderedDict()
        for i in range(self.num_en_layers):
            en_layers[f"encoder_layer_{i}"] = Encoder(
                d_model=self.d_model,
                nhead=self.nhead,
                activation=F.relu
            )
        self.patch_en_layers = nn.Sequential(en_layers)
        self.en_ln = nn.LayerNorm(self.d_model)

        heads_layers = OrderedDict()

        heads_layers["decoder_flatten"] = nn.Flatten(start_dim=-2)
        heads_layers["decoder_header"] = nn.Linear((self.seq_length // self.seg_len)*self.d_model, self.pred_len)
        heads_layers["detection_flatten"] = nn.Flatten(start_dim=-2)
        heads_layers["layer_norm"] = nn.LayerNorm(self.enc_feature)
        heads_layers["detection_header"] = nn.Linear(self.enc_feature, self.pred_len)
        heads_layers["dropout"] = nn.Dropout(0.)

        self.heads = nn.Sequential(heads_layers)

        for layer in self.heads:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal(layer.weight)

        self.statistical_feature = nn.Sequential(
            nn.Linear(self.num_globalfeature, self.num_globalfeature//2),
            nn.LayerNorm(self.num_globalfeature//2),
            nn.ReLU(inplace=True),
            nn.Linear(self.num_globalfeature // 2, self.pred_len),
        )

        self.sigmoid = nn.Sigmoid()

        self.fusion_linear = nn.Linear(2, self.pred_len)

    def forward(self, waveform, statistical_feature):
        torch._assert(waveform.dim() == 3, f"src: Expected (batch_size, seq_length, hidden_dim) got {waveform.shape}")
        torch._assert(statistical_feature.dim() == 2, f"src: Expected (batch_size, dim) got {statistical_feature.shape}")

        src = self.en_encoding(waveform)
        src += self.enc_pos_embedding
        src = self.pre_norm(src)

        for encoder in self.patch_en_layers:
            src = encoder(src)
        src = self.en_ln(src)
        res_waveform = self.heads(src)

        res_statistical = self.statistical_feature(statistical_feature).reshape(-1, 1)

        output = self.sigmoid(self.fusion_linear(torch.hstack((res_waveform, res_statistical))))

        if self.output_attention:
            return output, None
        else:
            return output
