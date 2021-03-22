import os
import librosa
import soundfile as sf
import time

import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from torchnet.meter import AverageValueMeter

from models.DCCRN import dccrn
from models.loss import SISNRLoss

from dataloader.THCHS30 import THCHS30
from config import opt


def train(mode='CL'):
    model = dccrn(mode)
    model.to(opt.device)

    train_data = THCHS30(phase='train')
    train_loader = DataLoader(train_data,
                              batch_size=opt.batch_size,
                              num_workers=opt.num_workers,
                              shuffle=True)

    optimizer = Adam(model.parameters(), lr=opt.lr)
    scheduler = MultiStepLR(optimizer,
                            milestones=[int(opt.max_epoch * 0.5),
                                        int(opt.max_epoch * 0.7),
                                        int(opt.max_epoch * 0.9)],
                            gamma=opt.lr_decay)
    criterion = SISNRLoss()

    loss_meter = AverageValueMeter()

    for epoch in range(0, opt.max_epoch):
        loss_meter.reset()
        for i, (data, label) in enumerate(train_loader):
            data = data.to(opt.device)
            label = label.to(opt.device)

            spec, wav = model(data)

            optimizer.zero_grad()
            loss = criterion(wav, label)
            loss.backward()
            optimizer.step()

            loss_meter.add(loss.item())

            if (i + 1) % opt.verbose_inter == 0:
                print('epoch', epoch + 1, 'batch', i + 1,
                      'SI-SNR', -loss_meter.value()[0])
        if (epoch + 1) % opt.save_inter == 0:
            print('save model at epoch {0} ...'.format(epoch + 1))
            save_path = os.path.join(opt.checkpoint_root,
                                     'DCCRN_{0}_{1}.pth'.format(mode, epoch + 1))
            torch.save(model.state_dict(), save_path)

        scheduler.step()

    save_path = os.path.join(opt.checkpoint_root,
                             'DCCRN_{0}.pth'.format(mode))
    torch.save(model.state_dict(), save_path)


# when denoising, use cpu
def denoise(mode, speech_file, save_dir, pth=None):
    assert os.path.exists(speech_file), 'speech file does not exist!'

    assert speech_file.endswith('.wav'), 'non-supported speech format!'

    if not os.path.exists(save_dir):
        print('warning: save directory does not exist, it will be created automatically!')
        os.makedirs(save_dir)

    model = dccrn(mode)
    if pth is not None:
        model.load_state_dict(torch.load(pth), strict=True)

    noisy_wav, _ = librosa.load(speech_file, sr=16000)

    noisy_wav = torch.Tensor(noisy_wav).reshape(1, -1)

    torch.cuda.synchronize()
    start = time.time()

    _, denoised_wav = model(noisy_wav)

    torch.cuda.synchronize()
    end = time.time()

    print('process time {0}s on device {1}'.format(end - start, 'cpu'))

    speech_name = os.path.basename(speech_file)[:-4]

    noisy_path = os.path.join(save_dir, speech_name + '_' + 'noisy' + '.wav')
    denoised_path = os.path.join(save_dir, speech_name + '_' + 'denoised' + '.wav')

    noisy_wav = noisy_wav.data.numpy().flatten()
    denoised_wav = denoised_wav.data.numpy().flatten()

    sf.write(noisy_path, noisy_wav, 16000)
    sf.write(denoised_path, denoised_wav, 16000)


if __name__ == '__main__':
    # train('E')

    test_speech_base = os.path.join(opt.data_root, 'THCHS-30', 'data_synthesized/test/noisy')
    test_speech = os.path.join(test_speech_base, 'D11_752_car.wav')

    save_dir = os.path.join(opt.sample_root, 'THCHS-30')
    pth = os.path.join(opt.checkpoint_root, 'DCCRN_E.pth')

    denoise('E', test_speech, save_dir, pth=pth)

    pass
