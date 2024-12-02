import os
import asyncio
import time

import websockets
import threading
import webbrowser
from http.server import HTTPServer, SimpleHTTPRequestHandler
import tkinter as tk
from tkinter import ttk, Canvas, scrolledtext
import pyaudio
import numpy as np
import re
import torch
import torch.nn as nn
import torchaudio
from datetime import datetime

# Ініціалізація PyAudio
p = pyaudio.PyAudio()

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 2048

connected_users = {}
user_widgets = {}
active_user = None

# Гіперпараметри для моделі
SAMPLE_RATE = 22050
NUM_SAMPLES = SAMPLE_RATE * 2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Mel-спектрограма
mel_spectrogram = torchaudio.transforms.MelSpectrogram(
    sample_rate=SAMPLE_RATE, n_mels=64, n_fft=1024, hop_length=512
)


# Визначення моделі
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
        self.fc = nn.Linear(64, 2)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


# Завантаження моделі
model = AudioClassifier().to(device)
try:
    state_dict = torch.load("models/model.pth", map_location=device)
    model.load_state_dict(state_dict, strict=True)
    print("Модель успішно завантажена.")
except RuntimeError as e:
    print(f"Помилка завантаження моделі: {str(e)}")
    exit()


def parse_user_agent(user_agent):
    """Парсинг User-Agent для відображення імені браузера та ОС."""
    browser_match = re.search(r'(Chrome|Firefox|Safari|Opera|Edg|MSIE)\/[\d.]+', user_agent)
    os_match = re.search(r'\(([^)]+)\)', user_agent)

    browser = browser_match.group(0) if browser_match else "Невідомий браузер"
    os_info = os_match.group(1) if os_match else "Невідома ОС"
    return f"{browser} на {os_info}"


def play_audio_stream_in_thread(audio_data):
    """Програвання аудіо в окремому потоці для зменшення затримок."""

    def play():
        try:
            stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, output=True)
            stream.write(audio_data)
            stream.stop_stream()
            stream.close()
        except Exception as e:
            print(f"Помилка відтворення аудіо: {e}")

    threading.Thread(target=play, daemon=True).start()


def update_volume_meter(canvas, volume):
    """Оновлення індикатора гучності."""
    canvas.delete("volume")
    height = int((volume / 32768) * 100)
    canvas.create_rectangle(10, 100 - height, 40, 100, fill="lime", tags="volume")


def process_audio_and_predict(audio_data):
    """Обробка аудіо-даних та передбачення."""
    signal = torch.frombuffer(audio_data, dtype=torch.int16).float().to(device)
    signal = signal.view(1, -1)

    # Пересемплінг, якщо необхідно
    if signal.shape[-1] != SAMPLE_RATE:
        signal = torchaudio.transforms.Resample(RATE, SAMPLE_RATE)(signal)
    signal = signal.mean(dim=0, keepdim=True)

    # Застосування Mel-спектрограми
    signal = mel_spectrogram(signal)
    signal = signal.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(signal)
        probabilities = torch.softmax(output, dim=1)
        shahed_prob = probabilities[0][1].item() * 100
        noise_prob = probabilities[0][0].item() * 100

    return shahed_prob, noise_prob


def display_active_users():
    """Виводить список всіх активних користувачів з ймовірностями."""
    log_text.config(state=tk.NORMAL)
    log_text.delete(1.0, tk.END)
    log_text.insert(tk.END, "Зараз прослуховуються користувачі:\n\n", 'header')
    for user, data in connected_users.items():
        if 'probabilities' in user_widgets[user]:
            shahed_prob, noise_prob = user_widgets[user]['probabilities']
            log_text.insert(tk.END,
                            f"{user}: Ймовірність 'Шахед' = {shahed_prob:.2f}%, Ймовірність 'Шум' = {noise_prob:.2f}%\n",
                            'error' if shahed_prob > 50 else 'info')
    log_text.config(state=tk.DISABLED)
    log_text.see(tk.END)


async def audio_handler(websocket, path):
    """Обробник WebSocket-з'єднань."""
    user_agent = websocket.request_headers.get('User-Agent', 'Невідомий пристрій')
    username = parse_user_agent(user_agent)
    coordinates = path.strip("/")

    if username not in connected_users:
        connected_users[username] = {'socket': websocket, 'coordinates': coordinates}
        log(f"{username} підключений з координатами {coordinates}")
        add_user_card(username, coordinates)

    try:
        async for message in websocket:
            if isinstance(message, bytes):
                volume = np.frombuffer(message, np.int16).astype(np.float32).max()
                update_volume_meter(user_widgets[username]['canvas'], volume)

                if user_widgets[username]['listen_var'].get():
                    play_audio_stream_in_thread(message)

                shahed_prob, noise_prob = process_audio_and_predict(message)
                user_widgets[username]['probabilities'] = (shahed_prob, noise_prob)

                log_prediction(username, shahed_prob, noise_prob)
                display_active_users()
    except websockets.exceptions.ConnectionClosed:
        log(f"{username} відключений")
        remove_user_card(username)


def log_prediction(username, shahed_prob, noise_prob):
    """Логування ймовірностей у консоль та інтерфейс."""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    message = f"[{timestamp}] {username}: Ймовірність 'Шахед' = {shahed_prob:.2f}%, Ймовірність 'Шум' = {noise_prob:.2f}%"
    print(message)
    log_text.config(state=tk.NORMAL)
    log_text.insert(tk.END, message + "\n", 'error' if shahed_prob > 50 else 'info')
    log_text.config(state=tk.DISABLED)
    log_text.see(tk.END)


def add_user_card(username, coordinates):
    """Додавання картки користувача в інтерфейс."""
    frame = ttk.Frame(scrollable_frame, padding=5)
    frame.grid(row=len(user_widgets) // 3, column=len(user_widgets) % 3, padx=10, pady=10)

    label = ttk.Label(frame, text=username)
    label.pack()

    canvas = Canvas(frame, width=50, height=100, bg="black")
    canvas.pack()

    toggle_var = tk.BooleanVar(value=False)
    toggle = ttk.Checkbutton(frame, text="Прослуховування", variable=toggle_var)
    toggle.pack()

    location_button = ttk.Button(frame, text="Переглянути локацію", command=lambda: open_location(coordinates))
    location_button.pack()

    user_widgets[username] = {'frame': frame, 'canvas': canvas, 'listen_var': toggle_var}


def open_location(coordinates):
    """Відкриває Google Maps з координатами."""
    base_url = "https://www.google.com/maps?q="
    webbrowser.open(f"{base_url}{coordinates}")


def remove_user_card(username):
    """Видалення картки користувача."""
    if username in user_widgets:
        user_widgets[username]['frame'].destroy()
        del user_widgets[username]
        del connected_users[username]


def log(message):
    """Виведення повідомлень у лог."""
    log_text.config(state=tk.NORMAL)
    log_text.insert(tk.END, f"{message}\n")
    log_text.config(state=tk.DISABLED)
    log_text.see(tk.END)


async def start_websocket_server():
    """Запуск WebSocket сервера."""
    async with websockets.serve(audio_handler, "localhost", 8765):
        log("WebSocket сервер запущено на ws://localhost:8765")
        await asyncio.Future()


def run_websocket_server():
    """Фоновий запуск WebSocket сервера."""
    asyncio.run(start_websocket_server())


def start_http_server():
    """Запуск HTTP сервера."""
    httpd = HTTPServer(("localhost", 8080), SimpleHTTPRequestHandler)
    httpd.serve_forever()


def start_servers():
    """Запуск серверів у потоках."""
    threading.Thread(target=start_http_server, daemon=True).start()
    webbrowser.open("http://localhost:8080/index.html")
    threading.Thread(target=run_websocket_server, daemon=True).start()


# Створення інтерфейсу
root = tk.Tk()
root.title("WebSocket і HTTP Сервер")
root.geometry("800x600")

# Основний фрейм
main_frame = ttk.Frame(root, padding=10)
main_frame.pack(fill=tk.BOTH, expand=True)

# Прокручувана область для карток
canvas = Canvas(main_frame)
scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
scrollable_frame = ttk.Frame(canvas)

scrollable_frame.bind(
    "<Configure>",
    lambda e: canvas.configure(
        scrollregion=canvas.bbox("all")
    )
)

canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
canvas.configure(yscrollcommand=scrollbar.set)

canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

# Лог-фрейм
log_frame = ttk.Frame(root, padding=10)
log_frame.pack(fill=tk.X)

log_text = scrolledtext.ScrolledText(log_frame, height=8, state='disabled')
log_text.tag_config('error', foreground='red')
log_text.tag_config('info', foreground='black')
log_text.tag_config('header', font='TkDefaultFont 10 bold')
log_text.pack(fill=tk.X, expand=True)

# Запуск серверів і основного циклу
start_servers()
root.mainloop()
