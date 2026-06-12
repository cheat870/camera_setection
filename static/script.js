let streamActive = false;
let resultInterval = null;

// DOM Elements
const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');
const videoFeed = document.getElementById('videoFeed');
const statusDiv = document.getElementById('status');
const liveResults = document.getElementById('liveResults');
const statisticsDiv = document.getElementById('statistics');
const refreshBtn = document.getElementById('refreshBtn');

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

// Fetch and display results
async function fetchResults() {
    if (!streamActive) return;
    
    try {
        const response = await fetch('/get_results');
        const data = await response.json();
        
        if (data.status === 'success' && data.results) {
            displayResults(data.results);
        }
    } catch (error) {
        console.error('Error fetching results:', error);
    }
}

// Display detection results
function displayResults(results) {
    // Display live results
    if (results.objects && results.objects.length > 0) {
        liveResults.innerHTML = results.objects.map(obj => `
            <div class="result-item">
                <strong>${obj.name}</strong><br>
                <span class="confidence">Confidence: ${(obj.confidence * 100).toFixed(1)}%</span>
            </div>
        `).join('');
    } else {
        liveResults.innerHTML = '<div class="loading">No objects detected yet...</div>';
    }
    
    // Display statistics
    if (results.object_summary && Object.keys(results.object_summary).length > 0) {
        const totalObjects = results.total_objects;
        const objectTypes = Object.keys(results.object_summary).length;
        
        let statsHtml = `
            <div class="stat-item">
                <span class="stat-label">Total Objects:</span>
                <span class="stat-value">${totalObjects}</span>
            </div>
            <div class="stat-item">
                <span class="stat-label">Object Types:</span>
                <span class="stat-value">${objectTypes}</span>
            </div>
            <div class="stat-item">
                <span class="stat-label">Last Update:</span>
                <span class="stat-value">${new Date(results.timestamp * 1000).toLocaleTimeString()}</span>
            </div>
            <div style="margin-top: 10px;">
                <strong>Detected Objects:</strong>
                <ul class="object-list">
                    ${Object.entries(results.object_summary).map(([name, count]) => `
                        <li><span>${name}:</span> <strong>${count}</strong></li>
                    `).join('')}
                </ul>
            </div>
        `;
        statisticsDiv.innerHTML = statsHtml;
    } else {
        statisticsDiv.innerHTML = '<p>No statistics available yet</p>';
    }
}

// Start camera
async function startCamera() {
    try {
        const response = await fetch('/start_camera');
        const data = await response.json();
        
        if (data.status === 'success') {
            videoFeed.src = '/video_feed?' + Date.now();
            streamActive = true;
            
            startBtn.disabled = true;
            stopBtn.disabled = false;
            statusDiv.className = 'status active';
            statusDiv.textContent = '🟢 Camera Active - Detecting Objects';
            liveResults.innerHTML = '<div class="loading">Detecting objects...</div>';
            
            // Start fetching results every second
            if (resultInterval) clearInterval(resultInterval);
            resultInterval = setInterval(fetchResults, 1000);
            
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
            
            // Stop fetching results
            if (resultInterval) {
                clearInterval(resultInterval);
                resultInterval = null;
            }
            
            startBtn.disabled = false;
            stopBtn.disabled = true;
            statusDiv.className = 'status inactive';
            statusDiv.textContent = '⚫ Camera Stopped';
            liveResults.innerHTML = '<div class="loading">Camera is stopped. Click "Start Camera" to begin.</div>';
            statisticsDiv.innerHTML = '<p>No data available</p>';
        }
    } catch (error) {
        console.error('Error stopping camera:', error);
    }
}

// Show error message
function showError(message) {
    statusDiv.className = 'status inactive';
    statusDiv.textContent = '❌ ' + message;
    liveResults.innerHTML = '<div class="loading">Error: ' + message + '</div>';
    
    setTimeout(() => {
        if (!streamActive) {
            statusDiv.textContent = '⚫ Camera Stopped';
            liveResults.innerHTML = '<div class="loading">Camera is stopped. Click "Start Camera" to begin.</div>';
        }
    }, 3000);
}

// Refresh button click
refreshBtn.addEventListener('click', () => {
    if (streamActive) {
        fetchResults();
    }
});

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