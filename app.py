from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing import image
import numpy as np 

app = Flask(__name__)

class_mapping = {
    0: 'circle',
    1: 'square',
    2: 'triangle',
    3: 'rectangle',
    4: 'star',
}


model = load_model('models/model.h5')

model.make_predict_function()

#def predict_label(img_path):
	#i = image.load_img(img_path, target_size=(224,224))
	#i = image.img_to_array(i)/255.0
	#i = i.reshape(224,224,3)
	#p = model.predict_classes(i)
	#return dic[p[0]]

# routes
def predict_label(img, model, class_mapping):
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Agregar una dimensión para el lote (batch)

    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions[0])

    predicted_label = class_mapping[predicted_class_index]

    return predicted_label

@app.route("/submit", methods=['GET', 'POST'])
def get_output():
    if request.method == 'POST':
        img = request.files['my_image']

        img_path = "static/" + img.filename
        img.save(img_path)

        img = image.load_img(img_path, target_size=(224, 224))
        p = predict_label(img, model, class_mapping)

        return render_template("index2.html", prediction=p, img_path=img_path)

    return render_template("index2.html")

if __name__ == '__main__':
    app.run(debug=True)