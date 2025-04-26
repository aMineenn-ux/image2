from PIL import Image
# Load an image to test
IM = Image.open("IMG_0450.jpg")
w, h = IM.size  # This will get the width and height
print(w, h)