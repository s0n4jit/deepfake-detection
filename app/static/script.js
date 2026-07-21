document.addEventListener("DOMContentLoaded", () => {
    const dropZone = document.getElementById("drop-zone");
    const fileInput = document.getElementById("file-input");
    const uploadPrompt = document.getElementById("upload-prompt");
    const previewContainer = document.getElementById("preview-container");
    const imagePreview = document.getElementById("image-preview");
    const scanCanvas = document.getElementById("scan-canvas");
    const scanBtn = document.getElementById("scan-btn");
    const scanProgress = document.getElementById("scan-progress");
    const resultsPanel = document.getElementById("results-panel");
    const errorBanner = document.getElementById("error-banner");
    const errorText = document.getElementById("error-text");
    const resetBtn = document.getElementById("reset-btn");
     
    // Result elements
    const verdictText = document.getElementById("verdict-text");
    const verdictConfidence = document.getElementById("verdict-confidence");
    const valCnn = document.getElementById("val-cnn");
    const modelVersionChip = document.getElementById("model-version-chip");
    const cnnModelNameLabel = document.getElementById("cnn-model-name-label");

    let selectedFile = null;
    let selectedModel = "cnn"; // Default model parameter
    let detectedBox = null;     // Bounding box from API response

    // Fetch and populate model details dynamically
    async function loadModelInfo() {
        try {
            const response = await fetch("/api/models");
            if (response.ok) {
                const data = await response.json();
                if (data.cnn) {
                    modelVersionChip.innerText = data.cnn.version || "cnn_v1";
                    cnnModelNameLabel.innerText = data.cnn.name;
                }
            }
        } catch (error) {
            console.error("Error loading model info:", error);
        }
    }
    loadModelInfo();

    // Handle Drag & Drop behavior
    dropZone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropZone.classList.add("dragover");
    });

    dropZone.addEventListener("dragleave", () => {
        dropZone.classList.remove("dragover");
    });

    dropZone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropZone.classList.remove("dragover");
        if (e.dataTransfer.files.length > 0) {
            handleFile(e.dataTransfer.files[0]);
        }
    });

    // Handle Click behavior to trigger input
    dropZone.addEventListener("click", (e) => {
        // Prevent click trigger if clicking preview image or canvas
        if (e.target !== dropZone && e.target !== uploadPrompt && !e.target.classList.contains("browse-link")) {
            return;
        }
        fileInput.click();
    });

    fileInput.addEventListener("change", () => {
        if (fileInput.files.length > 0) {
            handleFile(fileInput.files[0]);
        }
    });

    // Reset button functionality
    resetBtn.addEventListener("click", (e) => {
        e.stopPropagation(); // Avoid triggering dropZone click
        selectedFile = null;
        fileInput.value = "";
        imagePreview.src = "";
        previewContainer.style.display = "none";
        uploadPrompt.style.display = "block";
        scanBtn.setAttribute("disabled", "true");
        detectedBox = null;
        clearCanvas();
        resultsPanel.style.display = "none";
        errorBanner.style.display = "none";
    });

    // Preview File logic
    function handleFile(file) {
        selectedFile = file;
        
        // Reset canvas & states
        detectedBox = null;
        clearCanvas();
        resultsPanel.style.display = "none";
        errorBanner.style.display = "none";
        
        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            uploadPrompt.style.display = "none";
            previewContainer.style.display = "block";
            scanBtn.removeAttribute("disabled");
        };
        reader.readAsDataURL(file);
    }

    // Clear Canvas
    function clearCanvas() {
        const ctx = scanCanvas.getContext("2d");
        ctx.clearRect(0, 0, scanCanvas.width, scanCanvas.height);
    }

    // Scan Image action
    scanBtn.addEventListener("click", async () => {
        if (!selectedFile) return;

        // Reset UI states
        resultsPanel.style.display = "none";
        errorBanner.style.display = "none";
        scanProgress.style.display = "block";
        scanBtn.setAttribute("disabled", "true");
        clearCanvas();

        const formData = new FormData();
        formData.append("file", selectedFile);
        formData.append("model", selectedModel);

        try {
            const response = await fetch("/api/scan", {
                method: "POST",
                body: formData
            });

            if (!response.ok) {
                const data = await response.json();
                throw new Error(data.detail || "Scanning failed.");
            }

            const data = await response.json();
            displayResults(data);

        } catch (error) {
            showError(error.message);
        } finally {
            scanProgress.style.display = "none";
            scanBtn.removeAttribute("disabled");
        }
    });

    // Display Scan results
    function displayResults(data) {
        if (!data.face_detected) {
            showError("No face detected in this image. Try a clearer, front-facing photo.");
            return;
        }

        // Draw bounding box overlay
        if (data.box) {
            detectedBox = data.box;
            drawBoxOverlay(data.box);
        }

        // Parse verdict states
        const results = data.results;
        let finalVerdict = "real";
        let finalConf = 0.0;
        
        if (results.cnn) {
            valCnn.innerText = `${results.cnn.verdict.toUpperCase()} (${(results.cnn.confidence * 100).toFixed(1)}%)`;
            finalVerdict = results.cnn.verdict;
            finalConf = results.cnn.confidence;
        }

        const verdictCard = document.querySelector(".verdict-card");
        verdictCard.className = "verdict-card"; // Reset classes
        
        if (finalVerdict === "real") {
            verdictCard.classList.add("state-real");
            verdictText.innerText = "REAL";
        } else {
            verdictCard.classList.add("state-fake");
            verdictText.innerText = "FAKE";
        }

        verdictConfidence.innerText = `${(finalConf * 100).toFixed(1)}%`;
        resultsPanel.style.display = "block";
    }

    // Bounding Box Camera Focus Overlay Drawer
    function drawBoxOverlay(box) {
        // Resize canvas to match display size of the image
        const imgWidth = imagePreview.clientWidth;
        const imgHeight = imagePreview.clientHeight;
        
        scanCanvas.width = imgWidth;
        scanCanvas.height = imgHeight;

        // Scale factors
        const scaleX = imgWidth / imagePreview.naturalWidth;
        const scaleY = imgHeight / imagePreview.naturalHeight;

        const x = box.x * scaleX;
        const y = box.y * scaleY;
        const w = box.w * scaleX;
        const h = box.h * scaleY;

        const ctx = scanCanvas.getContext("2d");
        ctx.strokeStyle = "#2C5F7C"; // Steel Blue Accent
        ctx.lineWidth = 1;
        
        // Draw Corner Brackets (Camera Focus Reticle style)
        const len = Math.min(w, h) * 0.15; // 15% length
        
        ctx.beginPath();
        // Top Left
        ctx.moveTo(x + len, y);
        ctx.lineTo(x, y);
        ctx.lineTo(x, y + len);
        
        // Top Right
        ctx.moveTo(x + w - len, y);
        ctx.lineTo(x + w, y);
        ctx.lineTo(x + w, y + len);
        
        // Bottom Left
        ctx.moveTo(x, y + h - len);
        ctx.lineTo(x, y + h);
        ctx.lineTo(x + len, y + h);
        
        // Bottom Right
        ctx.moveTo(x + w - len, y + h);
        ctx.lineTo(x + w, y + h);
        ctx.lineTo(x + w, y + h - len);
        
        ctx.stroke();
    }

    // Redraw bounding box on window resize
    window.addEventListener("resize", () => {
        if (detectedBox && imagePreview.style.display !== "none") {
            drawBoxOverlay(detectedBox);
        }
    });

    // Helper to display errorsFact
    function showError(message) {
        errorText.innerText = message;
        errorBanner.style.display = "block";
    }
});
