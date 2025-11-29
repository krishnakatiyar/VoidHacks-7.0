import { GoogleGenerativeAI } from "@google/generative-ai";

// Configuration
const API_KEY = "AIzaSyBKnAuySubi-OA6J3dfz3GIgGdEwlvZcMc"; // WARNING: Exposed key for demo
const MODEL_URL = "./Model/";

// DOM Elements
const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const previewContainer = document.getElementById('previewContainer');
const imagePreview = document.getElementById('imagePreview');
const removeBtn = document.getElementById('removeImage');
const generateBtn = document.getElementById('generateBtn');
const loading = document.getElementById('loading');
const loadingText = document.getElementById('loadingText');
const results = document.getElementById('results');
const probabilityBars = document.getElementById('probabilityBars');
const reportContent = document.getElementById('reportContent');

// State
let model, maxPredictions;
let uploadedImage = null;

// Initialize
async function init() {
    try {
        const modelURL = MODEL_URL + "model.json";
        const metadataURL = MODEL_URL + "metadata.json";

        model = await tmImage.load(modelURL, metadataURL);
        maxPredictions = model.getTotalClasses();
        console.log("Teachable Machine Model Loaded");
    } catch (error) {
        console.error("Error loading model:", error);
        alert("Failed to load model. Please ensure model files are in ./Model/");
    }
}

// Event Listeners
dropZone.addEventListener('click', () => fileInput.click());
dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.style.borderColor = 'var(--primary)';
});
dropZone.addEventListener('dragleave', () => {
    dropZone.style.borderColor = 'var(--border)';
});
dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.style.borderColor = 'var(--border)';
    handleFile(e.dataTransfer.files[0]);
});
fileInput.addEventListener('change', (e) => handleFile(e.target.files[0]));
removeBtn.addEventListener('click', resetUpload);
generateBtn.addEventListener('click', processImage);

function handleFile(file) {
    if (!file || !file.type.startsWith('image/')) return;

    const reader = new FileReader();
    reader.onload = (e) => {
        imagePreview.src = e.target.result;
        uploadedImage = imagePreview; // Use the img element for prediction

        dropZone.style.display = 'none';
        previewContainer.style.display = 'inline-block';
        generateBtn.disabled = false;
        results.style.display = 'none';
    };
    reader.readAsDataURL(file);
}

function resetUpload() {
    fileInput.value = '';
    uploadedImage = null;
    dropZone.style.display = 'block';
    previewContainer.style.display = 'none';
    generateBtn.disabled = true;
    results.style.display = 'none';
}

async function processImage() {
    if (!model || !uploadedImage) return;

    setLoading(true, "Analyzing MRI Scan...");

    try {
        // 1. Run TM Model Prediction
        const prediction = await model.predict(uploadedImage);
        displayProbabilities(prediction);

        // 2. Generate Caption with Gemini (Vision)
        setLoading(true, "Generating Clinical Caption...");
        const caption = await generateCaption(uploadedImage);

        // 3. Generate Full Report with Gemini
        setLoading(true, "Compiling Wellness Report...");
        const report = await generateReport(prediction, caption);

        // 4. Display Results
        displayReport(report);
        setLoading(false);
        results.style.display = 'block';
        results.scrollIntoView({ behavior: 'smooth' });

    } catch (error) {
        console.error("Processing error:", error);
        alert("An error occurred during processing. See console for details.");
        setLoading(false);
    }
}

function displayProbabilities(prediction) {
    probabilityBars.innerHTML = '';
    prediction.forEach(p => {
        const percentage = (p.probability * 100).toFixed(1);
        const row = document.createElement('div');
        row.className = 'prob-row';
        row.innerHTML = `
            <div class="prob-label">
                <span>${p.className}</span>
                <span>${percentage}%</span>
            </div>
            <div class="prob-bar-bg">
                <div class="prob-bar-fill" style="width: ${percentage}%"></div>
            </div>
        `;
        probabilityBars.appendChild(row);
    });
}

async function generateCaption(imgElement) {
    const genAI = new GoogleGenerativeAI(API_KEY);
    const model = genAI.getGenerativeModel({ model: "gemini-2.0-flash-lite" });

    // Convert image to base64 for API
    const base64Data = imgElement.src.split(',')[1];
    const imagePart = {
        inlineData: {
            data: base64Data,
            mimeType: "image/jpeg" // Assuming jpeg/png
        }
    };

    const prompt = "Describe this MRI scan in technical medical terms, focusing on visible structures and any potential anomalies related to cognitive health. Keep it concise.";

    const result = await model.generateContent([prompt, imagePart]);
    const response = await result.response;
    return response.text();
}

async function generateReport(prediction, caption) {
    const genAI = new GoogleGenerativeAI(API_KEY);

    // Fetch system instructions
    const systemInstructionRes = await fetch('system_instructions.md');
    let systemInstruction = await systemInstructionRes.text();

    // Clean up system instruction if it has frontmatter or extra text (optional)
    // For now, assuming raw markdown is fine or simple text

    const model = genAI.getGenerativeModel({
        model: "gemini-2.0-flash-lite",
        systemInstruction: systemInstruction
    });

    // Format prediction for prompt
    const mriOutput = prediction.map(p => `${p.className}: ${(p.probability * 100).toFixed(1)}%`).join('\n');

    const prompt = `
MRI_MODEL_OUTPUT:
${mriOutput}

IMAGE_CAPTION:
"${caption}"
`;

    const result = await model.generateContent(prompt);
    const response = await result.response;
    return response.text();
}

function displayReport(markdownText) {
    reportContent.innerHTML = marked.parse(markdownText);
}

function setLoading(isLoading, text) {
    loading.style.display = isLoading ? 'block' : 'none';
    loadingText.textContent = text;
    generateBtn.disabled = isLoading;
}

// Start
init();
