document.addEventListener("DOMContentLoaded", function () {
    const uploadForm = document.getElementById("uploadForm");
    const fileInput = document.getElementById("file");
    const dropArea = document.getElementById("dropArea");
    const loading = document.getElementById("loadingOverlay");

    // Pastikan semua elemen penting ada
    if (!uploadForm || !fileInput || !dropArea || !loading) {
        // console.error("Elemen HTML penting tidak ditemukan.");
        return;
    }

    function showLoading(show) {
        if (show) {
            loading.classList.remove("hidden");
        } else {
            loading.classList.add("hidden");
        }
    }

    function validateFile(file) {
        if (!file) return false;
        const maxSize = 2 * 1024 * 1024; // 2 MB
        if (file.size > maxSize) {
            alert("⚠️ Ukuran file terlalu besar! Maksimal 2 MB.");
            // Reset input agar user bisa upload lagi tanpa refresh
            fileInput.value = ""; 
            return false;
        }
        return true;
    }

    // Saat user pilih file dari input
    fileInput.addEventListener("change", function () {
        if (fileInput.files.length > 0) {
            const file = fileInput.files[0];
            if (!validateFile(file)) return;
            
            showLoading(true);
            // Submit form setelah validasi
            setTimeout(() => uploadForm.submit(), 100); 
        }
    });

    // Drag & Drop Area
    dropArea.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropArea.classList.add("hover");
    });

    dropArea.addEventListener("dragleave", () => {
        dropArea.classList.remove("hover");
    });

    dropArea.addEventListener("drop", (e) => {
        e.preventDefault();
        dropArea.classList.remove("hover");
        const files = e.dataTransfer.files;
        if (files && files.length > 0) {
            const file = files[0];
            if (!validateFile(file)) return;
            
            // Pindahkan file ke input dan submit
            fileInput.files = files;
            showLoading(true);
            setTimeout(() => uploadForm.submit(), 100);
        }
    });

    // Setelah halaman reload (hasil tampil), sembunyikan loading screen
    window.addEventListener("load", function () {
        setTimeout(() => showLoading(false), 400); 
    });
});
