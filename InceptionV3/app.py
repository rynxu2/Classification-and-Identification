import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

MODEL_PATH ='InceptionV3\\animal.h5'

model = load_model(MODEL_PATH)

def predict_image():
    file_path = filedialog.askopenfilename()
    
    img = Image.open(file_path)
    img_tk = ImageTk.PhotoImage(image=img)
    img_label.configure(image=img_tk)
    img_label.image = img_tk

    img = image.load_img(file_path, target_size=(224, 224))
    x = image.img_to_array(img)
    
    x=x/255
    x = np.expand_dims(x, axis=0)

    preds = model.predict(x)
    preds=np.argmax(preds, axis=1)
    if preds==0:
        preds="Antelope"
    elif preds==1:
        preds="Badger"
    elif preds==2:
        preds="Bat"
    elif preds==3:
        preds="Bear"
    elif preds==4:
        preds="Bee"
    elif preds==5:
        preds="Beetle"
    elif preds==6:
        preds="Bison"

    result_label.configure(text=f'Predicted label: {preds}')
    print(preds)

root = tk.Tk()
root.title('Animal Classifier')

select_button = tk.Button(root, text='Select Image', command=predict_image)
select_button.pack()

img_label = tk.Label(root)
img_label.pack()

result_label = tk.Label(root)
result_label.pack()

root.mainloop()