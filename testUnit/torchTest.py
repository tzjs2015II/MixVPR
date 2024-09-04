import torch

if __name__ == '__main__':
    print(torch.__version__)
    print(torch.cuda.is_available())
    print(torch.version.cuda)
    print(torch.cuda.current_device())
    print(torch.cuda.get_device_name(0))
    print(torch.backends.cudnn.version())    

    print(torch.cuda.memory_summary())



    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = torch.randn(3, 3).to(device)
    print(x)
