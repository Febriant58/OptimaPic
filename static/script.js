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

    // Saat user submit form (klik upload)
    if (uploadForm) {
        uploadForm.addEventListener("submit", function () {
            showLoading(true);
        });
    }

    // Saat user pilih file langsung
    if (fileInput) {
        fileInput.addEventListener("change", function () {
            if (fileInput.files.length > 0) {
                showLoading(true);
                uploadForm.submit();
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
                fileInput.files = files;
                showLoading(true);
                uploadForm.submit();
            }
        });
    }

    // Setelah halaman selesai dimuat ulang (misal hasil sudah tampil)
    window.addEventListener("load", function () {
        setTimeout(() => showLoading(false), 400);
    });
});
