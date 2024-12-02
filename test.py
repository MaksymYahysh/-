import os
import torch
import torch.nn as nn
import torchaudio
import tkinter as tk
from tkinter import filedialog

# Гіперпараметри
SAMPLE_RATE = 22050
NUM_SAMPLES = SAMPLE_RATE * 2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Mel-спектрограма
mel_spectrogram = torchaudio.transforms.MelSpectrogram(
    sample_rate=SAMPLE_RATE, n_mels=64, n_fft=1024, hop_length=512
)

# Архітектура моделі
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.skip(x)
        return self.relu(out)

class AudioClassifier(nn.Module):
    def __init__(self):
        super(AudioClassifier, self).__init__()
        self.layer1 = ResidualBlock(1, 16, stride=2)
        self.layer2 = ResidualBlock(16, 32, stride=2)
        self.layer3 = ResidualBlock(32, 64, stride=2)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, 2)  # 2 класи: "Шахед" і "Шум"

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

def predict(model, audio_path):
    model.eval()
    try:
        signal, sr = torchaudio.load(audio_path)
    except Exception as e:
        print(f"Помилка завантаження аудіофайлу: {str(e)}")
        return

    print(f"Завантажено аудіофайл: {audio_path} з частотою дискретизації {sr}")

    # Пересемплювання
    if sr != SAMPLE_RATE:
        signal = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(signal)
        print("Аудіофайл пересемпльовано до SAMPLE_RATE.")

    # Усереднення каналів (якщо стерео)
    signal = signal.mean(dim=0, keepdim=True)

    # Перевірка довжини сигналу та доповнення/обрізка до NUM_SAMPLES
    if signal.shape[-1] < NUM_SAMPLES:
        padding = NUM_SAMPLES - signal.shape[-1]
        signal = nn.functional.pad(signal, (0, padding))
    else:
        signal = signal[:, :NUM_SAMPLES]

    print(f"Форма сигналу перед перетворенням: {signal.shape}")

    # Застосування Mel-спектрограми
    signal = mel_spectrogram(signal)
    print(f"Форма після Mel-спектрограми: {signal.shape}")

    # Додавання розмірності пакету та перенесення на пристрій
    signal = signal.unsqueeze(0).to(device)
    print(f"Форма сигналу після додавання пакету: {signal.shape}")

    with torch.no_grad():
        output = model(signal)

    probabilities = torch.softmax(output, dim=1)
    print(f"Ймовірності: {probabilities}")
    return probabilities

def open_file():
    file_path = filedialog.askopenfilename()
    if file_path:
        print(f"Обраний файл: {file_path}")
        probabilities = predict(model, file_path)
        if probabilities is not None:
            print(f"Ймовірність того, що це 'Шахед': {probabilities[0][1].item() * 100:.2f}%")
            print(f"Ймовірність того, що це 'Шум': {probabilities[0][0].item() * 100:.2f}%")

# Ініціалізація Tkinter
root = tk.Tk()
root.withdraw()

# Створення моделі та завантаження ваг
model = AudioClassifier().to(device)
try:
    state_dict = torch.load("models/model.pth", map_location=device)
    model.load_state_dict(state_dict, strict=True)
    print("Модель успішно завантажена.")
except RuntimeError as e:
    print(f"Помилка завантаження моделі: {str(e)}")
    exit()

# Відкриття файлу та виконання прогнозу
open_file()
