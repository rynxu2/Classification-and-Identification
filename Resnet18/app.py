import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import torch
from torch import nn
from torchvision import models, transforms

model = models.resnet18()
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 7)

state_dict = torch.load('Resnet18\\animal_classifier.pth')
model.load_state_dict(state_dict)

model.eval()

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

class_names = ['Antelope', 'Badger', 'Bat', 'Bear', 'Bee', 'Beetle', 'Bison']


def predict_image():
    file_path = filedialog.askopenfilename()

    img = Image.open(file_path)

    img_tk = ImageTk.PhotoImage(img)
    img_label.configure(image=img_tk)
    img_label.image = img_tk

    img_t = transform(img)
    img_t = img_t.unsqueeze(0)

    with torch.no_grad():
        output = model(img_t)
        predicted_class = output.argmax(dim=1).item()

    predicted_label = class_names[predicted_class]

    result_label.configure(text=f'Predicted label: {predicted_label}')
    print(predicted_label)


root = tk.Tk()
root.title('Animal Classifier')

select_button = tk.Button(root, text='Select Image', command=predict_image)
select_button.pack()

img_label = tk.Label(root)
img_label.pack()

result_label = tk.Label(root)
result_label.pack()

root.mainloop()
