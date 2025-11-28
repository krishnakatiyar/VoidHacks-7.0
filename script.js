// Model URL - assuming files are in ./Model/
const URL = "./Model/";

let model, webcam, ctx, labelContainer, maxPredictions;

async function init() {
    const startBtn = document.getElementById('start-btn');
    startBtn.innerHTML = 'Loading Model...';
    startBtn.disabled = true;

    const modelURL = URL + "model.json";
    const metadataURL = URL + "metadata.json";

    try {
        // Load the model and metadata
        model = await tmPose.load(modelURL, metadataURL);
        maxPredictions = model.getTotalClasses();

        // Convenience function to setup a webcam
        const size = 400; // Match CSS size roughly
        const flip = true; // whether to flip the webcam
        webcam = new tmPose.Webcam(size, size, flip); // width, height, flip
        await webcam.setup(); // request access to the webcam
        await webcam.play();
        window.requestAnimationFrame(loop);

        // append/get elements to the DOM
        const canvas = document.getElementById("webcam-container");
        canvas.innerHTML = ''; // Clear loading placeholder
        canvas.appendChild(webcam.canvas);

        // Setup label container
        labelContainer = document.getElementById("label-container");
        labelContainer.innerHTML = ''; // Clear previous
        for (let i = 0; i < maxPredictions; i++) { // and class labels
            const div = document.createElement("div");
            div.className = "label-item";
            div.id = `label-${i}`;
            div.innerHTML = `<span>${model.getClassLabels()[i]}</span> <span class="prob">0%</span>`;
            labelContainer.appendChild(div);
        }

        // Update UI state
        startBtn.innerHTML = 'Session Active';
        startBtn.style.background = 'linear-gradient(45deg, #66bb6a, #43a047)';
        // startBtn.disabled = false; // Optional: keep disabled to prevent re-init

    } catch (error) {
        console.error(error);
        startBtn.innerHTML = 'Error Loading';
        startBtn.disabled = false;
        alert('Error accessing camera or loading model. Please ensure you are running on a secure context (HTTPS or localhost) and have granted camera permissions.');
    }
}

async function loop(timestamp) {
    webcam.update(); // update the webcam frame
    await predict();
    window.requestAnimationFrame(loop);
}

async function predict() {
    // Prediction #1: run input through posenet
    // estimatePose can take in an image, video or canvas html element
    const { pose, posenetOutput } = await model.estimatePose(webcam.canvas);
    // Prediction 2: run input through teachable machine classification model
    const prediction = await model.predict(posenetOutput);

    let highestProb = 0;
    let bestClassIndex = -1;

    for (let i = 0; i < maxPredictions; i++) {
        const probability = prediction[i].probability.toFixed(2);
        const percentage = Math.round(probability * 100) + '%';

        const labelItem = document.getElementById(`label-${i}`);
        labelItem.querySelector('.prob').innerText = percentage;

        // Highlight active pose
        if (prediction[i].probability > highestProb) {
            highestProb = prediction[i].probability;
            bestClassIndex = i;
        }

        // Reset styles
        labelItem.classList.remove('active');
        labelItem.style.background = 'white';
        labelItem.style.borderColor = 'transparent';
    }

    // Highlight the best match if confidence is high enough
    if (highestProb > 0.75 && bestClassIndex !== -1) {
        const activeItem = document.getElementById(`label-${bestClassIndex}`);
        activeItem.classList.add('active');
        activeItem.style.background = '#e0f7fa';
        activeItem.style.borderColor = '#80deea';
    }
}

// Bind start button
document.getElementById('start-btn').addEventListener('click', init);
