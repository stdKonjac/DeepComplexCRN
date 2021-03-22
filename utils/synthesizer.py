from scipy.io import wavfile
import numpy as np
import soundfile as sf
import librosa
import random
import os
from config import opt


# split origin noise file to gain better generalization performance
def split_noise(noise_file, save_dir, prop=0.5):
    assert os.path.exists(noise_file), 'noise file does not exist!'

    assert noise_file.endswith('.wav'), 'non-supported noise format!'

    if not os.path.exists(save_dir):
        print('warning: save directory does not exist, it will be created automatically.')
        os.makedirs(save_dir)

    sample_rate, sig = wavfile.read(noise_file)

    train_len = sig.shape[0] * prop

    train_noise = sig[:int(train_len)]
    test_noise = sig[int(train_len):]

    # remove .wav
    noise_name = os.path.basename(noise_file)[:-4]

    train_noise_dir = os.path.join(save_dir, 'train')
    test_noise_dir = os.path.join(save_dir, 'test')

    if not os.path.exists(train_noise_dir):
        os.makedirs(train_noise_dir)
    if not os.path.exists(test_noise_dir):
        os.makedirs(test_noise_dir)

    train_noise_path = os.path.join(train_noise_dir, noise_name + '.wav')
    test_noise_path = os.path.join(test_noise_dir, noise_name + '.wav')

    sf.write(train_noise_path, train_noise, sample_rate)
    sf.write(test_noise_path, test_noise, sample_rate)


def synthesize_noisy_speech(speech_file, noise_file, save_dir, snr=0):
    assert os.path.exists(speech_file), 'speech file does not exist!'
    assert os.path.exists(noise_file), 'noise file does not exist!'

    assert speech_file.endswith('.wav'), 'non-supported speech format!'
    assert noise_file.endswith('.wav'), 'non-supported noise format!'

    if not os.path.exists(save_dir):
        print('warning: save directory does not exist, it will be created automatically.')
        os.makedirs(save_dir)

    speech_name = os.path.basename(speech_file)[:-4]
    noise_name = os.path.basename(noise_file)[:-4]

    # 原始语音
    a, a_sr = librosa.load(speech_file, sr=16000)
    # 噪音
    b, b_sr = librosa.load(noise_file, sr=16000)
    # 随机取一段噪声，保证长度和纯净语音长度一致，保证不会越界
    start = random.randint(0, b.shape[0] - a.shape[0])
    # 切片
    n_b = b[int(start):int(start) + a.shape[0]]

    # 平方求和
    sum_s = np.sum(a ** 2)
    sum_n = np.sum(n_b ** 2)
    # 信噪比为snr时的权重
    x = np.sqrt(sum_s / (sum_n * pow(10, snr)))

    noise = x * n_b
    noisy_speech = a + noise

    noisy_dir = os.path.join(save_dir, '{0}dB'.format(snr), 'noisy')
    clean_dir = os.path.join(save_dir, '{0}dB'.format(snr), 'clean')

    if not os.path.exists(noisy_dir):
        os.makedirs(noisy_dir)
    if not os.path.exists(clean_dir):
        os.makedirs(clean_dir)

    noisy_speech_path = os.path.join(noisy_dir, speech_name + '_' + noise_name + '.wav')
    clean_speech_path = os.path.join(clean_dir, speech_name + '.wav')

    sf.write(noisy_speech_path, noisy_speech, 16000)
    sf.write(clean_speech_path, a, 16000)


# split noise for train and test
def generate_noise_dataset(noise_base, save_dir):
    print('noise base directory: ', noise_base)
    print('output directory: ', save_dir)
    # find all noise file and split them with custom proportion
    for dir in os.listdir(noise_base):
        noise_dir = os.path.join(noise_base, dir)
        for file in os.listdir(noise_dir):
            if file.endswith('.wav'):
                noise_file = os.path.join(noise_dir, file)
                split_noise(noise_file, save_dir, prop=0.5)
    print('succesfully generated noise dataset!')


def generate_noisy_dataset(speech_base, noise_base, save_dir):
    print('speech base directory: ', speech_base)
    print('output directory: ', save_dir)
    noise_files = []
    for file in os.listdir(noise_base):
        if file.endswith('.wav'):
            noise_files.append(os.path.join(noise_base, file))
    for file in os.listdir(speech_base):
        if file.endswith('.wav'):
            speech_file = os.path.join(speech_base, file)
            noise_file = random.choice(noise_files)
            synthesize_noisy_speech(speech_file, noise_file, save_dir=save_dir, snr=0)
    print('successfully generate noisy dataset!')


if __name__ == '__main__':
    # origin speech data path
    noise_base = os.path.join(opt.data_root, 'THCHS-30', 'test-noise/noise')
    train_speech_base = os.path.join(opt.data_root, 'THCHS-30', 'data_thchs30/train')
    test_speech_base = os.path.join(opt.data_root, 'THCHS-30', 'data_thchs30/test')

    # synthesized speech data path
    noise_dir = os.path.join(opt.data_root, 'THCHS-30', 'data_synthesized/noise')
    train_dir = os.path.join(opt.data_root, 'THCHS-30', 'data_synthesized/train')
    test_dir = os.path.join(opt.data_root, 'THCHS-30', 'data_synthesized/test')

    # split origin noise for train and test
    # generate_noise_dataset(noise_base=noise_base, save_dir=noise_dir)

    # generate train noisy speech
    generate_noisy_dataset(speech_base=train_speech_base,
                           noise_base=os.path.join(noise_dir, 'train'),
                           save_dir=train_dir)

    # generate test noisy speech
    generate_noisy_dataset(speech_base=test_speech_base,
                           noise_base=os.path.join(noise_dir, 'test'),
                           save_dir=test_dir)
