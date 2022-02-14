from typing import Union
import torch.nn as nn

def get_device(module : Union[nn.Module, nn.Parameter]):
    if not (isinstance(module, nn.Module) \
         or isinstance(module, nn.Parameter)):
        raise RuntimeError("module is not a instance of torch.nn.Module")
    if hasattr(module, 'device'):
        return module.device
    else:
        params_devices = [p.device for p in module.parameters()]
        if len(params_devices) == 0:
            return None
        elif len(set(params_devices))==1:
            return params_devices[0]
        else:
            raise RuntimeError("The module is paralleled acrossed device, please get device in a inner module")


# unitest, should be removed later
if __name__ == "__main__":
    import torch
    import torch.nn as nn

    a = nn.Parameter(torch.randn(3,5))

    class MyNet(nn.Module):
        def __init__(self):
            super().__init__()

    class MyNet2(nn.Module):
        def __init__(self):
            super().__init__()
            self.l1 = nn.Linear(3,5).to('cuda:2')
            self.l2 = nn.Linear(3,5).to('cuda:2')

    class MyNet3(nn.Module):
        def __init__(self):
            super().__init__()
            self.l1 = nn.Linear(3,5).to('cuda:3')
            self.l2 = nn.Linear(3,5).cuda()

    class MyNet4:
        pass

    b = MyNet()
    c = MyNet2()
    d = MyNet3()
    e = MyNet4()

    print(get_device(a))
    print(get_device(b))
    print(get_device(c))
    print(get_device(e))
    print(get_device(d))



