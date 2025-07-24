import os
import shutil
import time

import matplotlib.pyplot as plt
import numpy as np
import torch

import art.attacks.evasion as evasion_attack
from art.estimators.classification import PyTorchClassifier
from utils.audio import AudioMNISTDataset, PreprocessRaw
from model.RawAudioCNN import RawAudioCNN
from model.ResNet18_audio import ResNet18
from model.VGG16_audio import VGG16
import random
import argparse


def display_waveform(waveform, title, wav_file, wave_form_path, sr=8000):
    """Display waveform plot and audio play UI."""
    plt.figure()
    plt.title(title)
    plt.plot(waveform)
    plt.savefig(wave_form_path)
    plt.close()
    vec2wav(waveform, wav_file, framerate=sr)


def vec2wav(pcm_vec, wav_file, framerate=16000):
    """
    将numpy数组转为单通道wav文件
    :param pcm_vec: 输入的numpy向量
    :param wav_file: wav文件名
    :param framerate: 采样率
    :return:
    """
    import wave
    if np.max(np.abs(pcm_vec)) > 1.0:
        pcm_vec *= 32767 / max(0.01, np.max(np.abs(pcm_vec)))
    else:
        pcm_vec = pcm_vec * 32768
    pcm_vec = pcm_vec.astype(np.int16)
    wave_out = wave.open(wav_file, 'wb')
    wave_out.setnchannels(1)
    wave_out.setsampwidth(2)
    wave_out.setframerate(framerate)
    wave_out.writeframes(pcm_vec)


def audio_attack(dataset_name, model_name, wav_path, wave_form_path, epsilon, device):
    # load AudioMNIST test set
    audiomnist_test = AudioMNISTDataset(root_dir="../data/" + dataset_name + "/test/", transform=PreprocessRaw())

    # load pretrained model
    if model_name == "RawAudioCNN":
        model = RawAudioCNN()
    elif model_name == "ResNet18":
        model = ResNet18()
    else:
        model = VGG16()
    model.load_state_dict(torch.load("../model-weights/audio/" + model_name + ".pt", map_location="cpu"))
    model.eval()

    # wrap model in an ART classifier
    classifier_art = PyTorchClassifier(
        model=model,
        loss=torch.nn.CrossEntropyLoss(),
        optimizer=None,
        input_shape=[1, 8000],
        nb_classes=10,
        clip_values=(-2 ** 15, 2 ** 15 - 1)
    )

    # 清空之前保存的语音对抗样本
    for item in ["raw/", "adv/"]:
        shutil.rmtree(wav_path+item, ignore_errors=True)
        os.makedirs(wav_path+item)
        shutil.rmtree(wave_form_path+item, ignore_errors=True)
        os.makedirs(wave_form_path+item)

    attack_success = 0
    pred_acc = 0
    attack_time = 0
    random.seed(123)
    total = 50
    for i in range(total):
        sample = audiomnist_test[random.randint(0, 5999)]

        waveform = sample['input']  # torch.Size([1, 8000])
        label = sample['digit']  # torch.Size([1])

        # craft adversarial example with PGD
        start_time = time.time()
        pgd = evasion_attack.ProjectedGradientDescent(classifier_art, eps=epsilon)

        adv_waveform = pgd.generate(x=torch.unsqueeze(waveform, 0).numpy())

        # evaluate the classifier on the adversarial example
        with torch.no_grad():
            _, pred = torch.max(model(torch.unsqueeze(waveform.to(device), 0)), 1)
            _, pred_adv = torch.max(model(torch.from_numpy(adv_waveform).to(device)), 1)
        end_time = time.time()
        attack_time += end_time - start_time

        pred_raw = f"{pred.tolist()[0]}"
        pred_adv = f"{pred_adv.tolist()[0]}"
        if pred_raw == str(label):
            pred_acc += 1
            if pred_adv != pred_raw:
                attack_success += 1

                print("raw-{}(ground truth):{}({})".format(attack_success, pred_raw, label))
                title = "raw-{}(correctly classified as {})".format(attack_success, pred_raw)
                display_waveform(waveform.numpy()[0, :], title=title,
                                 wav_file=wav_path + "raw/" + str(attack_success) + ".wav",
                                 wave_form_path=wave_form_path + "raw/" + str(attack_success) + ".png")

                print("adv-{}:\t\t{}".format(attack_success, pred_adv))
                title = "adv-{}(classified as {} instead of {})".format(attack_success, pred_adv, pred_raw)
                display_waveform(adv_waveform[0, 0, :], title=title,
                                 wav_file=wav_path + "adv/" + str(attack_success) + ".wav",
                                 wave_form_path=wave_form_path + "adv/" + str(attack_success) + ".png")
                print("--------------------")

    print("epsilon:{:.3f}  success rate:{:.3f}%  time(s)/batch:{:.3f}\nwav files saved in: {:}\nwaveform images saved in: {:}".
          format(epsilon, attack_success / pred_acc * 100, attack_time / total, wav_path, wave_form_path))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--method_name",
                        type=str,
                        default="PGD",
                        help="Which method to use.")
    parser.add_argument("--model_name",
                        type=str,
                        default=None,
                        help="Which model to attack.")
    parser.add_argument("--dataset_name",
                        type=str,
                        default="AudioMNIST",
                        help="Which dataset to attack.")
    parser.add_argument("--epsilon",
                        type=float,
                        required=True,
                        help="What epsilon to use.")
    args = parser.parse_args()
    # print(args.method_name, args.model_name, args.dataset_name, args.epsilon
    wav_path = '../adv-audio/' + args.dataset_name + '/' + args.model_name + '/' + args.method_name + '/epsilon={:.3f}'.format(args.epsilon) + '/wav/'
    wave_form_path = '../adv-audio/' + args.dataset_name + '/' + args.model_name + '/' + args.method_name + '/epsilon={:.3f}'.format(args.epsilon) + '/wave_form/'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    audio_attack(args.dataset_name, args.model_name, wav_path, wave_form_path, args.epsilon, device)


if __name__ == '__main__':
    main()
