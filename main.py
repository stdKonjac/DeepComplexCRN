from models.DCCRN import dccrn


def train(mode='CL'):
    model = dccrn(mode)

    model.to(device)


if __name__ == '__main__':
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
