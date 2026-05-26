import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SimpleForgeryCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class SimpleForgeryLSTM(nn.Module):
    def __init__(self, sequence_length=8):
        super().__init__()
        self.conv = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.lstm = nn.LSTM(input_size=16 * 32 * 32, hidden_size=64, batch_first=True)
        self.classifier = nn.Linear(64, 2)
        self.sequence_length = sequence_length

    def forward(self, x):
        batch_size = x.size(0)
        features = F.relu(self.conv(x))
        features = F.max_pool2d(features, 2)
        features = features.view(batch_size, self.sequence_length, -1)
        output, _ = self.lstm(features)
        return self.classifier(output[:, -1, :])


def _prepare_tensor(image: np.ndarray, size=(128, 128)) -> torch.Tensor:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, size)
    normalized = resized.astype(np.float32) / 255.0
    tensor = torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0)
    return tensor


def predict_deepfake(image: np.ndarray, threshold: float = 0.5) -> dict:
    tensor = _prepare_tensor(image)

    cnn = SimpleForgeryCNN()
    cnn.eval()
    with torch.no_grad():
        logits = cnn(tensor)
        scores = torch.softmax(logits, dim=1).cpu().numpy()[0]

    cnn_probability = float(scores[1])

    lstm = SimpleForgeryLSTM(sequence_length=1)
    lstm.eval()
    with torch.no_grad():
        logits = lstm(tensor)
        scores = torch.softmax(logits, dim=1).cpu().numpy()[0]

    lstm_probability = float(scores[1])

    return {
        "cnn": {
            "probability": cnn_probability,
            "is_suspicious": cnn_probability >= threshold,
        },
        "lstm": {
            "probability": lstm_probability,
            "is_suspicious": lstm_probability >= threshold,
        },
        "threshold": threshold,
        "message": "AI model sonuçları, eğitimli modellere göre optimize edilebilir."
    }
