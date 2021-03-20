import torch
from models.DCCRN import dccrn

if __name__ == '__main__':
    torch.manual_seed(10)
    torch.autograd.set_detect_anomaly(True)
    inputs = torch.randn([10, 16000 * 4]).clamp_(-1, 1)
    labels = torch.randn([10, 16000 * 4]).clamp_(-1, 1)

    # DCCRN-E
    # model = dccrn('E')
    # DCCRN-R
    # model = dccrn('R')
    # DCCRN-C
    # model = dccrn('C')
    # DCCRN-CL
    model = dccrn('CL')

    outputs = model(inputs)[1]
    loss = model.loss(outputs, labels, loss_mode='SI-SNR')
    print(loss)
