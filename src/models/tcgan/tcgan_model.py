import torch
import torch.nn as nn

from ..utils import GeneratorTransformerEncoder


class TCGANModel(nn.Module):
    def __init__(self, seq_len=150, channels=3, num_classes=9, latent_dim=100, data_embed_dim=10,
                 label_embed_dim=10, depth=3, num_heads=5,
                 forward_drop_rate=0.5, attn_drop_rate=0.5, init_type='normal'):
        super(TCGANModel, self).__init__()
        self.seq_len = seq_len
        self.channels = channels
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.data_embed_dim = data_embed_dim
        self.label_embed_dim = label_embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.attn_drop_rate = attn_drop_rate
        self.forward_drop_rate = forward_drop_rate

        self.l1 = nn.Linear(self.latent_dim + self.label_embed_dim, self.seq_len * self.data_embed_dim)
        self.label_embedding = nn.Embedding(self.num_classes, self.label_embed_dim)

        self.blocks = GeneratorTransformerEncoder(
            depth=self.depth,
            emb_size=self.data_embed_dim,
            num_heads=self.num_heads,
            drop_p=self.attn_drop_rate,
            forward_drop_p=self.forward_drop_rate
        )

        self.deconv = nn.Sequential(
            nn.Conv2d(self.data_embed_dim, self.channels, 1, 1, 0)
        )
        self.initialize_weights(init_type)

    def initialize_weights(self, init_type='normal'):
        for m in self.modules():
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, 0.02)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=1.0)
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=1.0)
                else:
                    raise NotImplementedError(f'Initialization method [{init_type}] not implemented')
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            elif classname.find('BatchNorm2d') != -1:
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0.0)

    def forward(self, z, labels):
        c = self.label_embedding(labels)
        x = torch.cat([z, c], 1)
        x = self.l1(x)
        x = x.view(-1, self.seq_len, self.data_embed_dim)
        H, W = 1, self.seq_len
        x = self.blocks(x)
        x = x.reshape(x.shape[0], 1, x.shape[1], x.shape[2])
        output = self.deconv(x.permute(0, 3, 1, 2))
        return output
