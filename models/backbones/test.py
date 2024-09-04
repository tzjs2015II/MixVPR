import torch
import torch.nn as nn
import timm

class CustomEfficientNet(nn.Module):
    def __init__(self, model_name='efficientnet_b0', pretrained=False, custom_layer=None):
        super().__init__()
        # 使用timm库中的create_model函数加载EfficientNet模型
        self.model = timm.create_model(model_name=model_name, pretrained=pretrained, features_only=True)
        
        # 这里可以修改已有的层或添加新层
        if custom_layer:
            self.model.blocks[5] = custom_layer  # 用自定义层替换第5个块
        
        # 示例：用自定义卷积层替换stem卷积层
        self.model.conv_stem = nn.Conv2d(3, self.model.conv_stem.out_channels, kernel_size=5, stride=2, padding=2)

    def forward(self, x):
        x = self.model(x)
        return x

# 测试
if __name__ == '__main__':
    x = torch.randn(4, 3, 224, 224)
    model = CustomEfficientNet(model_name='efficientnet_b0', pretrained=False, custom_layer=None)
    r = model(x)
    print(f'Input shape: {x.shape}')
    print(f'Output shape: {r[-1].shape}')  # r[-1]指的是最终的输出特征图
