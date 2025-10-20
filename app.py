from flask import Flask, request, render_template, send_file
import onnxruntime as rt
import numpy as np
from PIL import Image
import os, time, mimetypes, glob, hashlib

# =========================================================
# Konfigurasi Aplikasi Flask
# =========================================================
app = Flask(__name__)

UPLOAD_FOLDER = os.path.join('static', 'uploads')
RESULTS_FOLDER = os.path.join('static', 'results')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)
mimetypes.add_type('image/png', '.png')

# =========================================================
# Inisialisasi (Lazy Load + Cache)
# =========================================================
ONNX_LOADED = False
session_pre = None
session_end = None
model_cache = {}  # Cache hasil enhancement


def load_onnx_model():
    """Muat model ONNX hanya saat pertama kali dipanggil"""
    global session_pre, session_end, ONNX_LOADED
    if not ONNX_LOADED:
        try:
            print("⏳ Memuat model ONNX (lazy load)...")
            providers = ['CPUExecutionProvider']
            opt = rt.SessionOptions()
            opt.intra_op_num_threads = 1  # Optimasi CPU single-core (Railway)

            session_pre = rt.InferenceSession("esrgan-small-pre.onnx", sess_options=opt, providers=providers)
            session_end = rt.InferenceSession("esrgan-small-end.onnx", sess_options=opt, providers=providers)

            ONNX_LOADED = True
            print("✅ Model ONNX berhasil dimuat.")
        except Exception as e:
            print(f"⚠️ Gagal memuat model ONNX: {e}")


# =========================================================
# Fungsi Peningkatan Citra (dengan cache & penghematan RAM)
# =========================================================
def enhance_image_onnx(img):
    global model_cache

    if not ONNX_LOADED:
        load_onnx_model()

    # Hash unik untuk caching
    img_hash = hashlib.md5(img.tobytes()).hexdigest()
    if img_hash in model_cache:
        print("🧠 Cache hit: hasil sebelumnya digunakan.")
        return model_cache[img_hash]

    # Proses citra ke tensor
    img = img.convert('RGB')
    img_np = np.array(img).astype(np.float32) / 255.0
    img_tensor = np.transpose(img_np, (2, 0, 1))
    img_tensor = np.expand_dims(img_tensor, axis=0)

    # --- Model Pre ---
    input_pre_name = session_pre.get_inputs()[0].name
    output_pre = session_pre.run(None, {input_pre_name: img_tensor})[0]

    # --- Model End ---
    input_end_names = [inp.name for inp in session_end.get_inputs()]
    input_end_data = {
        input_end_names[0]: img_tensor,
        input_end_names[1]: output_pre
    }

    output_final = session_end.run(None, input_end_data)[0]

    # Postprocess
    output_final = np.clip(output_final[0] * 255.0, 0, 255).astype(np.uint8)
    output_final = np.transpose(output_final, (1, 2, 0))
    enhanced_img = Image.fromarray(output_final)

    # Simpan ke cache (maksimal 3 entri untuk hemat RAM)
    model_cache[img_hash] = enhanced_img
    if len(model_cache) > 3:
        oldest = next(iter(model_cache))
        del model_cache[oldest]
        print("🧹 Cache lama dihapus untuk hemat memori.")

    print("💾 Cache baru disimpan.")
    return enhanced_img


# =========================================================
# ROUTES
# =========================================================
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files or not request.files['file'].filename:
        return render_template('index.html', error="⚠️ Tidak ada file yang dipilih.")

    file = request.files['file']

    try:
        # Buka citra dan kurangi resolusi jika terlalu besar
        img = Image.open(file.stream).convert('RGB')
        file.stream.seek(0, os.SEEK_END)
        file_size = file.stream.tell()
        file.stream.seek(0)

        if file_size > 1_000_000:  # >1 MB
            print(f"⚠️ File besar ({round(file_size/1024/1024,2)} MB), sedang diresize agar hemat RAM.")
            max_size = (720, 720)
            img.thumbnail(max_size, Image.LANCZOS)

        # Bersihkan hasil lama
        for old_file in glob.glob(os.path.join(app.config['RESULTS_FOLDER'], 'enhanced_*.png')):
            try: os.remove(old_file)
            except: pass
        for old_file in glob.glob(os.path.join(app.config['UPLOAD_FOLDER'], 'original_*.png')):
            try: os.remove(old_file)
            except: pass

        # Proses peningkatan
        start_time = time.perf_counter()
        enhanced_img = enhance_image_onnx(img)
        process_time = round(time.perf_counter() - start_time, 2)

        # Simpan hasil
        unique_id = str(int(time.time()))
        original_filename = f'original_{unique_id}.png'
        enhanced_filename = f'enhanced_{unique_id}.png'

        original_path = os.path.join(app.config['UPLOAD_FOLDER'], original_filename)
        enhanced_path = os.path.join(app.config['RESULTS_FOLDER'], enhanced_filename)

        img.save(original_path)
        enhanced_img.save(enhanced_path)

        return render_template(
            'index.html',
            original_image=f'uploads/{original_filename}',
            enhanced_image=f'results/{enhanced_filename}',
            process_time=process_time,
            upscale_factor=4
        )

    except Exception as e:
        print(f"❌ Error saat memproses gambar: {e}")
        return render_template('index.html', error=f"Terjadi kesalahan: {str(e)}")


@app.route('/download/<filename>')
def download(filename):
    if 'enhanced_' not in filename or '..' in filename:
        return "File tidak valid.", 404

    path = os.path.join(app.config['RESULTS_FOLDER'], filename)
    try:
        return send_file(path, as_attachment=True, download_name="Enhanced_Image_SR.png")
    except FileNotFoundError:
        return "File hasil tidak ditemukan.", 404


# =========================================================
# ENTRY POINT (Railway & Lokal)
# =========================================================
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
