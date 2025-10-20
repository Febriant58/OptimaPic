document.addEventListener("DOMContentLoaded", function () {
    const uploadForm = document.getElementById("uploadForm");
    const fileInput = document.getElementById("file");
    const dropArea = document.getElementById("dropArea");
    const loading = document.getElementById("loadingOverlay");

    function showLoading(show) {
        if (!loading) return;
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
            return false;
        }
        return true;
    }

    // Saat user submit form
    if (uploadForm) {
        uploadForm.addEventListener("submit", function (e) {
            const file = fileInput.files[0];
            if (!validateFile(file)) {
                e.preventDefault();
                return;
            }
            showLoading(true);
        });
    }

    // Saat user pilih file dari input
    if (fileInput) {
        fileInput.addEventListener("change", function () {
            if (fileInput.files.length > 0) {
                const file = fileInput.files[0];
                if (!validateFile(file)) return;
                showLoading(true);
                setTimeout(() => uploadForm.submit(), 300); // delay halus
            }
        });
    }

    // Drag & Drop Area
    if (dropArea) {
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
                fileInput.files = files;
                showLoading(true);
                setTimeout(() => uploadForm.submit(), 300);
            }
        });
    }

    // Setelah halaman reload (hasil tampil)
    window.addEventListener("load", function () {
        setTimeout(() => showLoading(false), 400);
    });
});
