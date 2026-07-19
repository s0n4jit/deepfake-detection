document.addEventListener("DOMContentLoaded", () => {
    const dropZone = document.getElementById("drop-zone");
    const fileInput = document.getElementById("file-input");
    const uploadPrompt = document.getElementById("upload-prompt");
    const previewContainer = document.getElementById("preview-container");
    const imagePreview = document.getElementById("image-preview");
    const scanCanvas = document.getElementById("scan-canvas");
    const scanBtn = document.getElementById("scan-btn");
    const toggleBtns = document.querySelectorAll(".toggle-btn");
    const scanProgress = document.getElementById("scan-progress");
    const resultsPanel = document.getElementById("results-panel");
    const errorBanner = document.getElementById("error-banner");
    const errorText = document.getElementById("error-text");
    
    // Result elements
    const verdictText = document.getElementById("verdict-text");
    const verdictConfidence = document.getElementById("verdict-confidence");
    const rowClassical = document.getElementById("row-classical");
    const valClassical = document.getElementById("val-classical");
    const rowCnn = document.getElementById("row-cnn");
    const valCnn = document.getElementById("val-cnn");

    let selectedFile = null;
    let selectedModel = "both"; // Default model parameter
    let detectedBox = null;     // Bounding box from API response

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

    // Toggle button selectors
    toggleBtns.forEach(btn => {
        btn.addEventListener("click", () => {
            toggleBtns.forEach(b => b.classList.remove("active"));
            btn.classList.add("active");
            selectedModel = btn.getAttribute("data-model");
        });
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
        // If both models are run, compute an averaged summary, otherwise use the single active verdict
        const results = data.results;
        let finalVerdict = "real";
        let finalConf = 0.0;
        
        // Default rows visibility
        rowClassical.style.display = "none";
        rowCnn.style.display = "none";
        
        if (results.classical && results.cnn) {
            rowClassical.style.display = "flex";
            rowCnn.style.display = "flex";
            
            valClassical.innerText = `${results.classical.verdict.toUpperCase()} (${(results.classical.confidence * 100).toFixed(1)}%)`;
            valCnn.innerText = `${results.cnn.verdict.toUpperCase()} (${(results.cnn.confidence * 100).toFixed(1)}%)`;
            
            // If they disagree, show the higher confidence verdict
            if (results.classical.verdict === results.cnn.verdict) {
                finalVerdict = results.classical.verdict;
                finalConf = (results.classical.confidence + results.cnn.confidence) / 2.0;
            } else {
                if (results.classical.confidence > results.cnn.confidence) {
                    finalVerdict = results.classical.verdict;
                    finalConf = results.classical.confidence;
                } else {
                    finalVerdict = results.cnn.verdict;
                    finalConf = results.cnn.confidence;
                }
            }
        } else if (results.classical) {
            rowClassical.style.display = "flex";
            valClassical.innerText = `${results.classical.verdict.toUpperCase()} (${(results.classical.confidence * 100).toFixed(1)}%)`;
            finalVerdict = results.classical.verdict;
            finalConf = results.classical.confidence;
        } else if (results.cnn) {
            rowCnn.style.display = "flex";
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
