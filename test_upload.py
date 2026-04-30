import requests

url = 'http://127.0.0.1:8000/predict'
file_path = 'chest_xray/val/NORMAL/NORMAL2-IM-0383-0001.jpeg'

with open(file_path, 'rb') as f:
    files = {'file': (file_path, f, 'image/jpeg')}
    response = requests.post(url, files=files)

print("Response Status Code:", response.status_code)
print("Response JSON:", response.json())
