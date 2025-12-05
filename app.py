from flask import Flask, request, render_template, send_file
import onnxruntime as rt
import numpy as np
from PIL import Image
import os, time, mimetypes, glob, hashlib, gc
import threading 

# =========================================================
# Konfigurasi Aplikasi Flask
# =========================================================
app = Flask(__name__)

# Simpan semua file statis yang dibuat (asli & hasil) di satu folder 'results'
RESULTS_FOLDER = os.path.join('static', 'results')
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER

os.makedirs(RESULTS_FOLDER, exist_ok=True)
mimetypes.add_type('image/png', '.png')
MAX_FILE_SIZE = 2 * 1024 * 1024  # 2MB

# =========================================================
# Lazy Load Model ONNX dan Cache RAM
# =========================================================
ONNX_LOADED = False
session_pre = None
session_end = None
model_cache = {}

def load_onnx_model():
    """Muat model hanya sekali saat pertama digunakan"""
    global session_pre, session_end, ONNX_LOADED
    if not ONNX_LOADED:
        try:
            print("⏳ Memuat model ONNX (lazy load)...")
            providers = ['CPUExecutionProvider']
            opt = rt.SessionOptions()
            opt.intra_op_num_threads = 1
            opt.execution_mode = rt.ExecutionMode.ORT_SEQUENTIAL

            session_pre = rt.InferenceSession("esrgan-small-pre.onnx", sess_options=opt, providers=providers)
            session_end = rt.InferenceSession("esrgan-small-end.onnx", sess_options=opt, providers=providers)

            ONNX_LOADED = True
            print("✅ Model ONNX berhasil dimuat.")
        except Exception as e:
            print(f"⚠️ Gagal memuat model ONNX: {e}") 

def cleanup_cache():
    """Hapus cache lama (lebih dari 60 detik) dari memori"""
    current_time = time.time()
    expired = [key for key, (_, ts) in model_cache.items() if current_time - ts > 60]
    if expired:
        for key in expired:
            del model_cache[key]
        print(f"🧹 Cache memori ({len(expired)} entri) dihapus otomatis.")

def enhance_image_onnx(img):
    global model_cache

    if not ONNX_LOADED:
        load_onnx_model()

    if not ONNX_LOADED:
         raise RuntimeError("Model ONNX tidak tersedia atau gagal dimuat.")

    # Hash unik untuk caching
    img_hash = hashlib.md5(img.tobytes()).hexdigest()
    if img_hash in model_cache:
        print("🧠 Cache hit: hasil sebelumnya digunakan.")
        model_cache[img_hash] = (model_cache[img_hash][0], time.time()) 
        return model_cache[img_hash][0]

    img = img.convert('RGB')
    img_np = np.array(img).astype(np.float32) / 255.0
    img_tensor = np.transpose(img_np, (2, 0, 1))
    img_tensor = np.expand_dims(img_tensor, axis=0)

    # --- Proses Model ---
    input_pre_name = session_pre.get_inputs()[0].name
    output_pre = session_pre.run(None, {input_pre_name: img_tensor})[0]

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

    # Simpan cache
    model_cache[img_hash] = (enhanced_img, time.time())
    cleanup_cache()

    # Bersihkan tensor dari RAM
    del img_np, img_tensor, output_pre, output_final
    gc.collect()

    return enhanced_img


# =========================================================
# DISK CLEANUP WORKER (Dual Rules: Time & Count)
# =========================================================
def cleanup_worker(max_files_to_keep=3, hours_to_keep=0.5):
    """Menghapus file dari RESULTS_FOLDER berdasarkan waktu (30 menit) dan jumlah (max 3)."""
    
    cutoff_time = time.time() - (hours_to_keep * 3600)
    folder = app.config['RESULTS_FOLDER']
    
    while True:
        deleted_count = 0
        files_to_delete = []
        
        try:
            # 1. AMBIL SEMUA FILE DAN URUTKAN BERDASARKAN WAKTU MODIFIKASI
            files = glob.glob(os.path.join(folder, '*.png'))
            files_with_time = [(os.path.getmtime(f), f) for f in files]
            files_with_time.sort() # Urutkan dari yang PALING LAMA ke yang TERBARU
            
            current_file_count = len(files_with_time)
            
            # 2. HAPUS BERDASARKAN WAKTU (File yang sangat lama, >30 menit)
            for file_mod_time, file_path in files_with_time:
                if file_mod_time < cutoff_time:
                    files_to_delete.append(file_path)
                    
            # 3. HAPUS BERDASARKAN JUMLAH (Hanya simpan 3 file terbaru)
            if current_file_count > max_files_to_keep:
                # Ambil file terlama, kecuali yang max_files_to_keep terbaru
                files_to_keep = [path for _, path in files_with_time[-max_files_to_keep:]]
                
                for _, file_path in files_with_time:
                    # Tambahkan ke list hapus jika tidak termasuk file terbaru yang harus dipertahankan
                    if file_path not in files_to_delete and file_path not in files_to_keep:
                        files_to_delete.append(file_path)


            # Hapus file unik
            for file_path in set(files_to_delete):
                os.remove(file_path)
                deleted_count += 1
                    
        except Exception as e:
            print(f"❌ Error saat membersihkan disk: {e}")
        
        if deleted_count > 0:
            print(f"🗑️ Disk Cleanup: {deleted_count} file lama dihapus.")
        
        gc.collect()
        time.sleep(5 * 60) # Tidur selama 5 menit

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
        # Pengecekan ukuran file yang lebih efisien
        if request.content_length is not None and request.content_length > MAX_FILE_SIZE:
             return render_template('index.html', error=f"⚠️ File terlalu besar. Maksimal {MAX_FILE_SIZE / 1024 / 1024:.0f} MB.")

        img = Image.open(file.stream).convert('RGB')

        # === MODIFIKASI AGRESIF UNTUK MENGHEMAT RAM RAILWAY FREE TIER ===
        MAX_DIMENSION = 500  # DIUBAH DARI 720 menjadi 500 untuk hemat RAM
        if max(img.size) > MAX_DIMENSION:
            print(f"⚠️ Resolusi besar {img.size}, auto resize ke {MAX_DIMENSION}px.")
            img.thumbnail((MAX_DIMENSION, MAX_DIMENSION), Image.LANCZOS)
        # === END MODIFIKASI AGRESIF ===

        # Proses
        start = time.perf_counter()
        enhanced_img = enhance_image_onnx(img)
        duration = round(time.perf_counter() - start, 2)

        # Simpan hasil (Original dan Enhanced disimpan di RESULTS_FOLDER)
        ts = str(int(time.time()))
        original_filename = f'original_{ts}.png'
        enhanced_filename = f'enhanced_{ts}.png'

        img.save(os.path.join(app.config['RESULTS_FOLDER'], original_filename))
        enhanced_img.save(os.path.join(app.config['RESULTS_FOLDER'], enhanced_filename))

        # Bersihkan RAM setelah selesai
        del img, enhanced_img
        gc.collect()

        return render_template(
            'index.html',
            original_image=f'results/{original_filename}', 
            enhanced_image=f'results/{enhanced_filename}',
            process_time=duration,
            upscale_factor=4
        )

    except Exception as e:
        print(f"❌ Error saat memproses gambar: {e}")
        gc.collect() 
        return render_template('index.html', error="Terjadi kesalahan saat memproses gambar. Coba lagi atau pastikan format file benar.")


@app.route('/download/<filename>')
def download(filename):
    if 'enhanced_' not in filename or '..' in filename:
        return "File tidak valid.", 404

    path = os.path.join(app.config['RESULTS_FOLDER'], filename)
    try:
        return send_file(path, as_attachment=True, download_name="Enhanced_OptimaPic.png")
    except FileNotFoundError:
        return "File hasil tidak ditemukan.", 404


# =========================================================
# ENTRY POINT (Railway & Lokal)
# =========================================================
if __name__ == '__main__':
    # Start background cleanup thread saat aplikasi dimulai
    print("⏳ Memulai background disk cleanup worker (Max 3 files, 5 menit cek)...")
    cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
    cleanup_thread.start()
    
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
