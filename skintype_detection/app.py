from flask import Flask, request, render_template, send_from_directory
import onnxruntime as ort
from PIL import Image
import numpy as np
import os
import uuid
from skincare_suggestions import get_skincare_suggestions

app = Flask(__name__)
model_path = "model_resnet50.onnx"
session = ort.InferenceSession(model_path)

# Preprocessing image (sesuaikan dengan input model Anda)
def preprocess_image(image):
    image = image.resize((224, 224))  # ukuran sesuai model ResNet50
    image = np.array(image).astype('float32') / 255.0
    if image.shape[-1] == 4:  # Jika gambar RGBA
        image = image[:, :, :3]
    image = np.transpose(image, (2, 0, 1))  # channel-first
    image = np.expand_dims(image, axis=0)
    return image

# Mapping label index ke tipe kulit
label_map = {
    0: "Normal",
    1: "Berminyak",
    2: "Kering",
    3: "Kombinasi",
    4: "Berjerawat"
}

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    suggestions = None
    uploaded_image_url = None

    if request.method == "POST":
        image_file = request.files["image"]
        if image_file:
            image = Image.open(image_file).convert("RGB")
            input_data = preprocess_image(image)

            # Simpan gambar yang diupload untuk ditampilkan
            filename = f"{uuid.uuid4().hex}.png"
            save_path = os.path.join("static", "uploads", filename)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            image.save(save_path)
            uploaded_image_url = f"/static/uploads/{filename}"

            input_name = session.get_inputs()[0].name
            output = session.run(None, {input_name: input_data})
            pred_index = int(np.argmax(output[0]))
            skin_type = label_map.get(pred_index, "Tidak Dikenal")
            suggestions = get_skincare_suggestions(skin_type)
            result = f"Tipe kulit Anda: {skin_type}"

    return render_template("index.html", result=result, suggestions=suggestions, uploaded_image=uploaded_image_url)

if __name__ == "__main__":
    app.run(debug=True)
