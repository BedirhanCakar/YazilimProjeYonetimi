import cv2
import numpy as np
import requests

img = np.zeros((100, 100, 3), dtype=np.uint8)
cv2.imwrite('test_upload_real.png', img)
with open('test_upload_real.png', 'rb') as f:
    resp = requests.post(
        'http://127.0.0.1:8000/upload/',
        files={'file': ('test_upload_real.png', f, 'image/png')},
        data={'threshold': '0.5'}
    )
print('status', resp.status_code)
try:
    print(resp.json())
except Exception:
    print(resp.text)
