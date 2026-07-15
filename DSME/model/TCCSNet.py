from torch import nn
import torch
import math
class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        ).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return self.pe[:, : x.size(1)]
    
class SelfAttentionBlock(nn.Module):
    def __init__(self, num_channels) -> None:
        super(SelfAttentionBlock, self).__init__()
        self.ln1 = nn.LayerNorm(normalized_shape=num_channels)
        # multihead attention + dropout 
        num_heads = 2
        if num_channels // 2 != 0:
            num_heads = 1 

        self.msa = nn.MultiheadAttention(embed_dim=num_channels, num_heads=num_heads, dropout=0.1, batch_first=True) 
        self.ln2 = nn.LayerNorm(normalized_shape=num_channels)
        
        self.ffn = nn.Sequential(
            nn.Linear(in_features=num_channels, out_features=32),
            nn.ReLU(False),
            nn.Linear(in_features=32, out_features=num_channels),
            nn.ReLU(False),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # add skip-connect
        ln1_x = self.ln1(x)
        out = self.msa(ln1_x, ln1_x, ln1_x)[0] + x
        # add skip-connect
        out = self.ffn(self.ln2(out)) + out
        return out
    

class TCCSNet(nn.Module):
    def __init__(self, window_len, num_channels, num_classes) -> None:
        super(TCCSNet, self).__init__()
        self.num_channels  = window_len
        self.num_classes = num_channels
        self.window_len = num_classes
        self.omiga = nn.Parameter(torch.tensor(0.5, requires_grad=True))
        # time
        self.time_conv_block1 = nn.Sequential(
            # Conv Block k=3 F=32,64
            nn.Conv2d(in_channels=num_channels, out_channels=32, kernel_size=(3,1), padding=(1,0)),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,1), padding=(1,0)),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=(1,1), stride=1),
        )
        self.time_dropout1 = nn.Dropout()
        self.position_embedding = PositionalEmbedding(d_model= 64)
        self.time_sa1 = SelfAttentionBlock(64)
        self.time_sa2 = SelfAttentionBlock(64)
        
        # channel 
        self.channel_conv_block1 = nn.Sequential(
            # Conv Block k=3 F=32,64
            nn.Conv2d(in_channels=window_len, out_channels=32, kernel_size=(3,1), padding=(1,0)),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,1), padding=(1,0)),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=(1,1), stride=1),
        ) 
        
        self.channel_dropout1 = nn.Dropout()
        self.channel_sa1 = SelfAttentionBlock(64)
        self.channel_sa2 = SelfAttentionBlock(64)
        
        # after concatenate
        self.conv_block2 = nn.Sequential(
            # Conv Block k=3 F=64
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,1), padding=(1,0)),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,1), padding=(1,0)),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=(3,1),stride=1)
        )

        self.fc = nn.Sequential(
            nn.Linear(in_features=64*(self.window_len+self.num_channels), out_features=512),
            nn.ReLU(False),
            nn.Dropout(0.4),
            nn.Linear(in_features=512, out_features=self.num_classes),
        )
    def forward(self, x):

        x = x.permute(0, 2, 1)

        time_x = self.time_conv_block1(x)
        time_x = self.time_dropout1(time_x)
        time_x = time_x.transpose(1, 2)
        pos = self.position_embedding(time_x)
        time_x = time_x + pos
        time_x = self.time_sa1(time_x)
        time_x = self.time_sa2(time_x)
        
        channel_x = self.channel_conv_block1(x)
        channel_x = self.channel_dropout1(channel_x)
        channel_x = channel_x.transpose(1, 2)
        channel_x = self.channel_sa1(channel_x)
        channel_x = self.channel_sa2(channel_x)
        


        omiga = torch.sigmoid(self.omiga)
        total_x = torch.cat([omiga * time_x ,(1 - omiga) * channel_x ], dim=1)
        
        total_x = total_x.transpose(1, 2)
        total_x = self.conv_block2(total_x)
        
        total_x = total_x.reshape(total_x.shape[0], -1)
        
        out = self.fc(total_x)      
        return out