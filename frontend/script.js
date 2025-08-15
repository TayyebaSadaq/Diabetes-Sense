const API_URL = 'http://127.0.0.1:5000';

function switchTab(tabName) {
    // Remove active class from all tabs and contents
    document.querySelectorAll('.tab').forEach(tab => tab.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));
    
    // Add active class to clicked tab and corresponding content
    event.target.classList.add('active');
    document.getElementById(tabName).classList.add('active');
}

document.getElementById('predictionForm').addEventListener('submit', async function(e) {
    e.preventDefault();
    
    const formData = new FormData(e.target);
    const data = {};
    
    // Convert form data to object
    for (let [key, value] of formData.entries()) {
        data[key] = parseFloat(value);
    }
    
    // Show loading state
    document.getElementById('loading').classList.add('show');
    document.getElementById('predictBtn').disabled = true;
    document.getElementById('predictBtn').textContent = 'Analyzing...';
    document.getElementById('results').innerHTML = '';
    
    try {
        const response = await fetch(`${API_URL}/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data)
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const results = await response.json();
        displayResults(results);
        
    } catch (error) {
        console.error('Error:', error);
        document.getElementById('results').innerHTML = `
            <div class="model-result" style="border-left-color: #e53e3e;">
                <h3>❌ Error</h3>
                <p>Failed to get prediction. Please make sure the backend server is running on port 5000.</p>
                <p><strong>Error details:</strong> ${error.message}</p>
            </div>
        `;
    } finally {
        // Hide loading state
        document.getElementById('loading').classList.remove('show');
        document.getElementById('predictBtn').disabled = false;
        document.getElementById('predictBtn').textContent = '🔮 Predict Diabetes Risk';
    }
});

function displayResults(results) {
    const resultsContainer = document.getElementById('results');
    let html = '<h2>🎯 Prediction Results</h2>';
    
    const modelNames = {
        'logistic_regression': 'Logistic Regression',
        'random_forest': 'Random Forest',
        'gradient_boosting': 'Gradient Boosting'
    };
    
    for (const [modelKey, result] of Object.entries(results)) {
        const modelName = modelNames[modelKey] || modelKey;
        const predictionClass = result.prediction === 'Diabetic' ? 'diabetic' : 'not-diabetic';
        const predictionIcon = result.prediction === 'Diabetic' ? '⚠️' : '✅';
        
        html += `
            <div class="model-result">
                <h3>${predictionIcon} ${modelName}</h3>
                
                <div class="prediction-info">
                    <div class="info-item">
                        <div class="label">Prediction</div>
                        <div class="value prediction ${predictionClass}">
                            ${result.prediction}
                        </div>
                    </div>
                    <div class="info-item">
                        <div class="label">Confidence</div>
                        <div class="value">${(result.confidence * 100).toFixed(1)}%</div>
                    </div>
                    <div class="info-item">
                        <div class="label">Model Accuracy</div>
                        <div class="value">${(result.accuracy * 100).toFixed(1)}%</div>
                    </div>
                </div>
                
                <div class="explanation">
                    <h4>📝 Explanation:</h4>
                    <p>${result.text_explanation}</p>
                    
                    ${result.lime_explanation_image ? `
                        <h4 style="margin-top: 20px;">📊 Feature Importance Analysis:</h4>
                        <img src="data:image/png;base64,${result.lime_explanation_image}" 
                             alt="LIME Explanation for ${modelName}" 
                             class="lime-image">
                    ` : ''}
                </div>
            </div>
        `;
    }
    
    resultsContainer.innerHTML = html;
}
