import requests

response = requests.get('https://api.github.com')
print(response.json())
if response.status_code == 200:
    print("Request was successful!")
else:
    print("Request failed!")
