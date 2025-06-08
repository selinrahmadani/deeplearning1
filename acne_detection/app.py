from flask import Flask, render_template, request
import onnxruntime as ort
from PIL import Image
import numpy as np
import os
import uuid

app = Flask(__name__)

# Pastikan folder upload ada
UPLOAD_FOLDER = os.path.join("static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Inisialisasi model
model_path = "model/acne-yv8m.onnx"
session = ort.InferenceSession(model_path)

# Label dan saran skincare
labels = {
    0: ("Kulit Normal", ["Gunakan pelembab ringan", "SPF 30+", "Gentle Cleanser"]),
    1: ("Kulit Berminyak", ["Gunakan salicylic acid", "Oil-free moisturizer", "Clay mask"]),
    2: ("Kulit Kering", ["Gunakan hyaluronic acid", "Rich moisturizer", "Hydrating toner"]),
    3: ("Kulit Berjerawat", ["Gunakan benzoyl peroxide", "Niacinamide", "Non-comedogenic products"]),
    4: ("Kulit Sensitif", ["Gunakan ceramide", "Fragrance-free", "Hypoallergenic cleanser"]),
}

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess(image):
    image = image.resize((640, 640))  # Ukuran input sesuai model
    image = np.array(image).astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))  # CHW
    image = np.expand_dims(image, axis=0)  # Tambah batch dimensi
    return image

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    advice = None
    image_path = None

    if request.method == "POST":
        try:
            file = request.files["image"]
            if file and allowed_file(file.filename):
                # Simpan dengan nama unik
                ext = file.filename.rsplit(".", 1)[1].lower()
                unique_filename = f"{uuid.uuid4().hex}.{ext}"
                upload_path = os.path.join(UPLOAD_FOLDER, unique_filename)

                # Buka dan simpan gambar
                img = Image.open(file.stream).convert("RGB")
                img.save(upload_path)

                # Proses & prediksi
                input_tensor = preprocess(img)
                input_name = session.get_inputs()[0].name
                outputs = session.run(None, {input_name: input_tensor})
                pred_idx = int(np.argmax(outputs[0]))

                # Ambil label & saran
                if pred_idx in labels:
                    prediction, advice = labels[pred_idx]
                else:
                    prediction = "Tidak Dikenali"
                    advice = ["Silakan coba gambar lain."]

                image_path = os.path.join("static", "uploads", unique_filename)

            else:
                prediction = "File tidak valid"
                advice = ["Silakan unggah file dengan format JPG, JPEG, atau PNG."]

        except Exception as e:
            prediction = "Terjadi kesalahan dalam memproses gambar."
            advice = [str(e)]

    return render_template("index.html", prediction=prediction, advice=advice, image_path=image_path)

if __name__ == "__main__":
    app.run(debug=True)
