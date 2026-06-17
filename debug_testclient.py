import cv2
import numpy as np
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)
img = np.zeros((100, 100, 3), dtype=np.uint8)
_, buf = cv2.imencode('.png', img)
files = {
    'file': ('test.png', buf.tobytes(), 'image/png')
}
data = {'threshold': '0.5'}

try:
    response = client.post('/upload/', files=files, data=data)
    print('status', response.status_code)
    print(response.text)
except Exception as e:
    print('exception', repr(e))
