document.addEventListener('DOMContentLoaded', () => {
    // DOM references
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const uploadPrompt = document.getElementById('upload-prompt');
    const previewContainer = document.getElementById('preview-container');
    const imagePreview = document.getElementById('image-preview');
    const changeImageBtn = document.getElementById('change-image');
    const fileName = document.getElementById('file-name');
    const analyzeBtn = document.getElementById('analyze-btn');
    const btnText = document.getElementById('btn-text');
    const spinner = document.getElementById('spinner');

    const resultsPlaceholder = document.getElementById('results-placeholder');
    const resultsLoading = document.getElementById('results-loading');
    const resultsSuccess = document.getElementById('results-success');
    const resultsInvalid = document.getElementById('results-invalid');

    const diagnosisBadge = document.getElementById('diagnosis-badge');
    const diagnosisIcon = document.getElementById('diagnosis-icon');
    const diagnosisValue = document.getElementById('diagnosis-value');
    const confidenceText = document.getElementById('confidence-text');
    const confidenceBar = document.getElementById('confidence-bar');
    const infoBanner = document.getElementById('info-banner');
    const infoText = document.getElementById('info-text');
    const invalidMessage = document.getElementById('invalid-message');

    let currentFile = null;

    // ---- Drag & Drop ----
    ['dragenter', 'dragover'].forEach(evt => {
        dropZone.addEventListener(evt, (e) => {
            e.preventDefault();
            dropZone.classList.add('dragover');
        });
    });

    ['dragleave', 'drop'].forEach(evt => {
        dropZone.addEventListener(evt, (e) => {
            e.preventDefault();
            dropZone.classList.remove('dragover');
        });
    });

    dropZone.addEventListener('drop', (e) => {
        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            handleFile(e.dataTransfer.files[0]);
        }
    });

    // ---- File Input ----
    fileInput.addEventListener('change', (e) => {
        if (e.target.files && e.target.files[0]) {
            handleFile(e.target.files[0]);
        }
    });

    // ---- Change Image ----
    changeImageBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        fileInput.click();
    });

    function handleFile(file) {
        if (!file.type.startsWith('image/')) {
            alert('Please upload a valid image file.');
            return;
        }

        currentFile = file;
        fileName.textContent = file.name;

        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            uploadPrompt.classList.add('hidden');
            previewContainer.classList.remove('hidden');
            analyzeBtn.disabled = false;
            resetResults();
        };
        reader.readAsDataURL(file);
    }

    function resetResults() {
        resultsPlaceholder.classList.remove('hidden');
        resultsLoading.classList.add('hidden');
        resultsSuccess.classList.add('hidden');
        resultsInvalid.classList.add('hidden');
        confidenceBar.style.width = '0%';
    }

    function showLoading() {
        resultsPlaceholder.classList.add('hidden');
        resultsLoading.classList.remove('hidden');
        resultsSuccess.classList.add('hidden');
        resultsInvalid.classList.add('hidden');
    }

    function showSuccess(data) {
        resultsLoading.classList.add('hidden');
        resultsSuccess.classList.remove('hidden');

        const isNormal = data.prediction === 'Normal';
        const cls = isNormal ? 'normal' : 'pneumonia';

        diagnosisBadge.className = 'diagnosis-badge ' + cls;
        diagnosisIcon.textContent = isNormal ? '✓' : '⚠';
        diagnosisValue.textContent = data.prediction;
        confidenceText.textContent = data.confidence + '%';

        confidenceBar.className = 'confidence-fill ' + cls;
        setTimeout(() => {
            confidenceBar.style.width = data.confidence + '%';
        }, 100);

        infoText.textContent = isNormal
            ? 'No signs of pneumonia detected. This AI tool is for educational purposes only. Always consult a medical professional.'
            : 'Signs consistent with pneumonia detected. This AI tool is for educational purposes only. Please consult a medical professional immediately.';
    }

    function showInvalid(message) {
        resultsLoading.classList.add('hidden');
        resultsInvalid.classList.remove('hidden');
        invalidMessage.textContent = message;
    }

    // ---- Analyze ----
    analyzeBtn.addEventListener('click', async () => {
        if (!currentFile) return;

        // Set loading state
        analyzeBtn.disabled = true;
        btnText.textContent = 'Analyzing...';
        spinner.classList.remove('hidden');
        showLoading();

        const formData = new FormData();
        formData.append('file', currentFile);

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.detail || 'Server error during analysis.');
            }

            if (!data.is_xray) {
                showInvalid(data.message);
            } else {
                showSuccess(data);
            }

        } catch (error) {
            showInvalid(error.message || 'An unexpected error occurred. Please try again.');
        } finally {
            analyzeBtn.disabled = false;
            btnText.textContent = 'Analyze Scan';
            spinner.classList.add('hidden');
        }
    });
});
