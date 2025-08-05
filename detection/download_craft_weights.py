import requests

url = "https://huggingface.co/amitesh863/craft/resolve/main/craft_mlt_25k.pth"
file_path = "weights/craft_mlt_25k.pth"

print("Downloading CRAFT weights...")

response = requests.get(url, stream=True)
if response.status_code == 200:
    with open(file_path, 'wb') as f:
        for chunk in response.iter_content(1024):
            f.write(chunk)
    print("Download complete!")
else:
    print("Failed to download. Status code:", response.status_code)
