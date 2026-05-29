import base64
import urllib.request
import urllib.error
import uuid

png_base64 = 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGNgYAAAAAMAASsJTYQAAAAASUVORK5CYII='
content = base64.b64decode(png_base64)
boundary = '----WebKitFormBoundary' + uuid.uuid4().hex
body = []
body.append('--' + boundary)
body.append('Content-Disposition: form-data; name="file"; filename="test.png"')
body.append('Content-Type: image/png')
body.append('')
body.append(content)
body.append('--' + boundary)
body.append('Content-Disposition: form-data; name="threshold"')
body.append('')
body.append('0.5')
body.append('--' + boundary + '--')
body.append('')
body_bytes = b'\r\n'.join(x if isinstance(x, bytes) else x.encode('utf-8') for x in body)
req = urllib.request.Request('http://127.0.0.1:8000/upload/', data=body_bytes, method='POST')
req.add_header('Content-Type', 'multipart/form-data; boundary=' + boundary)

try:
    with urllib.request.urlopen(req, timeout=20) as r:
        print('status', r.status)
        print(r.read().decode('utf-8'))
except urllib.error.HTTPError as e:
    print('status', e.code)
    print(e.read().decode('utf-8'))
except Exception as e:
    print('error', e)
