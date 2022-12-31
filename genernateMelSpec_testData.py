import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

matplotlib.use('TkAgg')

from config import data_source

during_time = 20


def main():
    music_path = '../../../LibriSpeech-SI/test'
    melSpecImag_path = '../../../data_set/' + data_source
    melSpecImag_path = os.path.join(melSpecImag_path, 'test')
    config_value = data_source.split('_')
    n_fft = int(config_value[2])
    hop_length = int(config_value[3])
    n_mels = int(config_value[4])

    if os.path.exists(melSpecImag_path):
        pass
    else:
        os.makedirs(melSpecImag_path)

    elements = os.listdir(music_path)
    for element in elements:
        music_file = os.path.join(music_path, element)
        y, sr = librosa.load(music_file)
        fs = sr

        # data process  to check
        pre_len = len(y)
        if pre_len < 20 * sr:
            cicle_time = 20 * sr / pre_len + 1
            y_temp = y
            for i in range(int(cicle_time)):
                y_temp = np.concatenate([y, y_temp])
            y = y_temp
        y = y[0:20 * sr]

        melspec = librosa.feature.melspectrogram(y, sr=fs, n_fft=n_fft, hop_length=hop_length,
                                                 n_mels=n_mels)  # (128,856)
        logmelspec = librosa.power_to_db(melspec)  # (128,856)

        plt.figure()
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)

        librosa.display.specshow(logmelspec, sr=fs)
        name = element.split('.')[0]
        name = name + '.png'
        save_file = os.path.join(melSpecImag_path, name)
        plt.savefig(save_file)
        # plt.show()
        plt.close('all')


if __name__ == '__main__':
    main()
