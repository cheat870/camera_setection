let streamActive = false;

// DOM Elements
const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');
const videoFeed = document.getElementById('videoFeed');
const statusDiv = document.getElementById('status');
const detectionInfo = document.getElementById('detectionInfo');

// Load model info on page load
async function loadModelInfo() {
    try {
        const response = await fetch('/model_info');
        const data = await response.json();
        if (data.status === 'success') {
            document.getElementById('modelInfo').innerHTML = `
                🤖 ${data.model} | ${data.device.toUpperCase()} | ${data.classes} classes
            `;
        }
    } catch (error) {
        console.error('Error loading model info:', error);
    }
}

// Start camera
async function startCamera() {
    try {
        const response = await fetch('/start_camera');
        const data = await response.json();
        
        if (data.status === 'success') {
            // Set video feed source
            videoFeed.src = '/video_feed?' + Date.now();
            streamActive = true;
            
            // Update UI
            startBtn.disabled = true;
            stopBtn.disabled = false;
            statusDiv.className = 'status active';
            statusDiv.textContent = '🟢 Camera Active - Detecting Objects';
            detectionInfo.innerHTML = '🔍 Camera streaming with YOLOv8n detection...';
            
            // Add event listener for when feed loads
            videoFeed.onload = () => {
                console.log('Video feed loaded');
            };
            
            videoFeed.onerror = () => {
                console.error('Error loading video feed');
                stopCamera();
            };
        } else {
            showError('Could not start camera. Make sure camera is connected.');
        }
    } catch (error) {
        console.error('Error starting camera:', error);
        showError('Error connecting to server');
    }
}

// Stop camera
async function stopCamera() {
    try {
        const response = await fetch('/stop_camera');
        const data = await response.json();
        
        if (data.status === 'success') {
            videoFeed.src = '';
            streamActive = false;
            
            // Update UI
            startBtn.disabled = false;
            stopBtn.disabled = true;
            statusDiv.className = 'status inactive';
            statusDiv.textContent = '⚫ Camera Stopped';
            detectionInfo.innerHTML = '💤 Camera is stopped. Click "Start Camera" to begin detection.';
        }
    } catch (error) {
        console.error('Error stopping camera:', error);
    }
}

// Show error message
function showError(message) {
    statusDiv.className = 'status inactive';
    statusDiv.textContent = '❌ ' + message;
    detectionInfo.innerHTML = '⚠️ ' + message;
    
    setTimeout(() => {
        if (!streamActive) {
            statusDiv.textContent = '⚫ Camera Stopped';
            detectionInfo.innerHTML = '💤 Camera is stopped. Click "Start Camera" to begin detection.';
        }
    }, 3000);
}

// Update detection info periodically
setInterval(async () => {
    if (streamActive) {
        // You can add more detailed detection stats here
        detectionInfo.innerHTML = `
            🎯 YOLOv8n Detection Active<br>
            📍 Real-time object detection<br>
            🟢 Detecting multiple objects simultaneously
        `;
    }
}, 2000);

// Event listeners
startBtn.addEventListener('click', startCamera);
stopBtn.addEventListener('click', stopCamera);

// Load model info when page loads
loadModelInfo();

// Clean up on page unload
window.addEventListener('beforeunload', () => {
    if (streamActive) {
        stopCamera();
    }
});