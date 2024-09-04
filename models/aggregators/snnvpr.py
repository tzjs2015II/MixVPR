import torch
import torch.nn.functional as F
import torch.nn as nn

import numpy as np

from spikingjelly.clock_driven.neuron import MultiStepParametricLIFNode, MultiStepLIFNode,LIFNode



class FeatureMixerLayer(nn.Module):
    def __init__(self, in_dim, mlp_ratio=1,in_channels=1024):
        super().__init__()
        self.mix = nn.Sequential(
            nn.LayerNorm(in_dim),#归一化
            nn.Linear(in_dim, int(in_dim * mlp_ratio)),
            nn.ReLU(),
            nn.Linear(int(in_dim * mlp_ratio), in_dim),

        )
        # 脉冲链接模块
        self.mlif_1=nn.Sequential(
            MultiStepLIFNode(tau=2.0, detach_reset=False, backend='torch'),# 理论上torch的后端比较慢
            nn.Conv2d(out_channels=in_dim,in_channels=in_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(in_dim),
        )
        self.mlif_2=nn.Sequential(
            MultiStepLIFNode(tau=2.0, detach_reset=False, backend='torch'),# 理论上torch的后端比较慢
            nn.Conv2d(out_channels=in_channels,in_channels=in_dim, kernel_size=1, stride=1),
            nn.BatchNorm2d(in_channels),
        ) 

        # self.SNN_MLIF = MultiStepLIFNode(tau=2.0, detach_reset=False, backend='torch')      
        
        for m in self.modules():
            if isinstance(m, (nn.Linear)): 
                nn.init.trunc_normal_(m.weight, std=0.02)# 权重初始化为标准差为0.02
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
    def forward(self, x):
        # x_1 = x + self.mix(x)
        # x_1=x_1.permute(2, 1, 0).unsqueeze(-1)
        # x_SNN_1= self.mlif_1(x_1)
        # x_SNN_2= self.mlif_2(x_SNN_1)
        # x_2=x_SNN_2.permute(2, 1, 0, 3).flatten(2,3)
        # x=x_1+x_2
        # return  x 
        
        # 内存爆炸占用的核心是中间变量复制太多次，导致如果输入特征很大，那么中间变量也会变的非常大
        x_1 = x + self.mix(x)
        x=x.permute(2, 1, 0).unsqueeze(-1)
        x= self.mlif_1(x)
        x= self.mlif_2(x)
        x=x.permute(2, 1, 0, 3).flatten(2,3)
        x=x_1+x
        return  x 

        # return  x + self.mix(x)+self.SNN_MLIF(x)

class SNNVPR(nn.Module):
    def __init__(self,
                 in_channels=1024,
                 in_h=20,
                 in_w=20,
                 out_channels=512,
                 mix_depth=1,
                 mlp_ratio=1,
                 out_rows=4,
                 ) -> None:
        super().__init__()

        self.in_h = in_h # height of input feature maps
        self.in_w = in_w # width of input feature maps
        self.in_channels = in_channels # depth of input feature maps
        
        self.out_channels = out_channels # depth wise projection dimension
        self.out_rows = out_rows # row wise projection dimesion

        self.mix_depth = mix_depth # L the number of stacked FeatureMixers
        self.mlp_ratio = mlp_ratio # ratio of the mid projection layer in the mixer block

        hw = in_h*in_w
        self.mix = nn.Sequential(*[
            FeatureMixerLayer(in_dim=hw, 
                              mlp_ratio=mlp_ratio,
                              in_channels=self.in_channels)
            for _ in range(self.mix_depth)
        ])
        self.channel_proj = nn.Linear(in_channels, out_channels)
        self.row_proj = nn.Linear(hw, out_rows)

    def forward(self, x):
        x = x.flatten(2)
        x = self.mix(x)
        x = x.permute(0, 2, 1)
        x = self.channel_proj(x)
        x = x.permute(0, 2, 1)
        x = self.row_proj(x)
        x = F.normalize(x.flatten(1), p=2, dim=-1)
        return x


# -------------------------------------------------------------------------------

def print_nb_params(m):
    model_parameters = filter(lambda p: p.requires_grad, m.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(f'Trainable parameters: {params/1e6:.3}M')


def main():
    x = torch.randn(1, 256, 20, 20)
    agg = SNNVPR(
        in_channels=256,
        in_h=20,
        in_w=20,
        out_channels=256,
        mix_depth=4,
        mlp_ratio=1,
        out_rows=4)

    print_nb_params(agg)
    output = agg(x)
    print(output.shape)


if __name__ == '__main__':
    main()
