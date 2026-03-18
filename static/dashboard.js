// Multi-Domain LSTM Forecasting System - Dashboard JavaScript

// Global state
let currentDomain = null;
let currentDataset = null;
let trainingChart = null;
let forecastChart = null;

// Initialize on page load
document.addEventListener('DOMContentLoaded', function() {
    initializeNavigation();
    initializeUploadZone();
    initializeEventListeners();
    updateArchitectureDiagram();
});

// Navigation
function initializeNavigation() {
    const navTabs = document.querySelectorAll('.nav-tab');
    const sections = document.querySelectorAll('.content-section');
    
    navTabs.forEach(tab => {
        tab.addEventListener('click', () => {
            const targetSection = tab.dataset.section;
            
            // Update active tab
            navTabs.forEach(t => t.classList.remove('active'));
            tab.classList.add('active');
            
            // Update active section
            sections.forEach(s => s.classList.remove('active'));
            document.getElementById(targetSection).classList.add('active');
        });
    });
}


// Display domain info
// Display file info
function displayFileInfo(file) {
    const domainInfo = document.getElementById('domainInfo');
    const card = document.getElementById('selectedDomainCard');
    card.style.display = 'block';
    
    // Format file size
    const size = (file.size / 1024).toFixed(2) + ' KB';
    
    domainInfo.innerHTML = `
        <div class="stat-item">
            <span class="stat-label">File Name</span>
            <span class="stat-value" style="font-size: 1.2rem">${file.name}</span>
        </div>
        <div class="stat-item">
            <span class="stat-label">File Size</span>
            <span class="stat-value" style="font-size: 1.2rem">${size}</span>
        </div>
        <div class="stat-item">
            <span class="stat-label">Type</span>
            <span class="stat-value" style="font-size: 1.2rem">Custom CSV</span>
        </div>
        <div class="stat-item">
            <span class="stat-label">Status</span>
            <span class="stat-value" style="font-size: 1.2rem; color: var(--success-color)">Uploaded</span>
        </div>
    `;
}

// Display dataset statistics
function displayDatasetStats(stats) {
    document.getElementById('totalRecords').textContent = stats.total_records;
    document.getElementById('totalFeatures').textContent = Object.keys(stats.features).length;
    document.getElementById('dateRange').textContent = 
        `${stats.date_range.start.split('T')[0]} to ${stats.date_range.end.split('T')[0]}`;
}

// Display quality report
function displayQualityReport(report) {
    const qualityReport = document.getElementById('qualityReport');
    qualityReport.innerHTML = `
        <div class="quality-item">
            <span class="quality-label">Total Records</span>
            <span class="quality-value">${report.total_records}</span>
        </div>
        <div class="quality-item">
            <span class="quality-label">Total Features</span>
            <span class="quality-value">${report.total_features}</span>
        </div>
        <div class="quality-item">
            <span class="quality-label">Missing Values</span>
            <span class="quality-value">${Object.values(report.missing_values).reduce((a, b) => a + b, 0)}</span>
        </div>
        <div class="quality-item">
            <span class="quality-label">Duplicate Rows</span>
            <span class="quality-value">${report.duplicate_rows}</span>
        </div>
        <div class="quality-item">
            <span class="quality-label">Temporal Order</span>
            <span class="quality-value">${report.temporal_order_valid ? '✓ Valid' : '✗ Invalid'}</span>
        </div>
    `;
}

// Display data preview
function displayDataPreview(preview) {
    const dataPreview = document.getElementById('dataPreview');
    
    if (!preview || preview.length === 0) {
        dataPreview.innerHTML = '<p>No data to preview</p>';
        return;
    }
    
    const columns = Object.keys(preview[0]);
    let tableHTML = '<table><thead><tr>';
    columns.forEach(col => {
        tableHTML += `<th>${col}</th>`;
    });
    tableHTML += '</tr></thead><tbody>';
    
    preview.forEach(row => {
        tableHTML += '<tr>';
        columns.forEach(col => {
            let value = row[col];
            if (typeof value === 'number') {
                value = value.toFixed(4);
            }
            tableHTML += `<td>${value}</td>`;
        });
        tableHTML += '</tr>';
    });
    
    tableHTML += '</tbody></table>';
    dataPreview.innerHTML = tableHTML;
}

// Initialize upload zone
function initializeUploadZone() {
    const uploadZone = document.getElementById('uploadZone');
    const fileInput = document.getElementById('fileInput');
    
    uploadZone.addEventListener('click', () => fileInput.click());
    
    uploadZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadZone.classList.add('dragover');
    });
    
    uploadZone.addEventListener('dragleave', () => {
        uploadZone.classList.remove('dragover');
    });
    
    uploadZone.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadZone.classList.remove('dragover');
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFileUpload(files[0]);
        }
    });
    
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFileUpload(e.target.files[0]);
        }
    });
}

// Handle file upload
async function handleFileUpload(file) {
    try {
        updateStatus('Uploading dataset...');
        
        const formData = new FormData();
        formData.append('file', file);
        formData.append('domain_id', 'custom');
        
        const response = await fetch('/api/upload_dataset', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (data.success) {
            currentDomain = 'custom';
            currentDataset = data;
            
            displayFileInfo(file);
            
            displayDatasetStats(data.statistics);
            displayQualityReport(data.quality_report);
            displayDataPreview(data.preview);
            
            updateStatus('Dataset uploaded successfully');
            showNotification('Dataset uploaded successfully', 'success');
        } else {
            showNotification(data.error, 'error');
        }
    } catch (error) {
        console.error('Error uploading file:', error);
        showNotification('Error uploading file', 'error');
    }
}

// Initialize event listeners
function initializeEventListeners() {
    // Preprocessing
    document.getElementById('preprocessBtn').addEventListener('click', preprocessData);
    
    // Model configuration updates
    document.getElementById('lstmUnits').addEventListener('input', updateArchitectureDiagram);
    document.getElementById('dropoutRate').addEventListener('input', updateArchitectureDiagram);
    
    // Training
    document.getElementById('trainBtn').addEventListener('click', trainModel);
    
    // Forecasting
    document.getElementById('generateForecastBtn').addEventListener('click', generateForecast);
    
    // Evaluation
    document.getElementById('evaluateBtn').addEventListener('click', evaluateModel);
}

// Preprocess data
async function preprocessData() {
    try {
        if (!currentDataset) {
            showNotification('Please load a dataset first', 'warning');
            return;
        }
        
        updateStatus('Preprocessing data...');
        
        const config = {
            missing_value_strategy: document.getElementById('missingValueStrategy').value,
            scaling_method: document.getElementById('scalingMethod').value,
            create_lag_features: document.getElementById('createLagFeatures').checked,
            create_rolling_features: document.getElementById('createRollingFeatures').checked
        };
        
        const response = await fetch('/api/preprocess', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(config)
        });
        
        const data = await response.json();
        
        if (data.success) {
            updateStatus('Data preprocessed successfully');
            showNotification('Preprocessing completed', 'success');
        } else {
            showNotification(data.error, 'error');
        }
    } catch (error) {
        console.error('Error preprocessing data:', error);
        showNotification('Error preprocessing data', 'error');
    }
}

// Update architecture diagram
function updateArchitectureDiagram() {
    const lstmUnits = parseInt(document.getElementById('lstmUnits').value) || 100;
    const dropoutRate = parseFloat(document.getElementById('dropoutRate').value) || 0.2;
    
    document.getElementById('lstm1Info').textContent = `Units: ${lstmUnits}`;
    document.getElementById('lstm2Info').textContent = `Units: ${Math.floor(lstmUnits / 2)}`;
    document.getElementById('dropout1Info').textContent = `Rate: ${dropoutRate}`;
    document.getElementById('dropout2Info').textContent = `Rate: ${dropoutRate}`;
}

// Train model
async function trainModel() {
    try {
        if (!currentDataset) {
            showNotification('Please load and preprocess a dataset first', 'warning');
            return;
        }
        
        updateStatus('Training model...');
        document.getElementById('trainBtn').disabled = true;
        
        const config = {
            lookback: parseInt(document.getElementById('lookback').value),
            lstm_units: parseInt(document.getElementById('lstmUnits').value),
            dropout_rate: parseFloat(document.getElementById('dropoutRate').value),
            learning_rate: parseFloat(document.getElementById('learningRate').value),
            epochs: parseInt(document.getElementById('epochs').value),
            batch_size: parseInt(document.getElementById('batchSize').value)
        };
        
        const response = await fetch('/api/train_model', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(config)
        });
        
        const data = await response.json();
        
        if (data.success) {
            displayTrainingResults(data);
            updateStatus('Model trained successfully');
            showNotification('Training completed', 'success');
        } else {
            showNotification(data.error, 'error');
        }
    } catch (error) {
        console.error('Error training model:', error);
        showNotification('Error training model', 'error');
    } finally {
        document.getElementById('trainBtn').disabled = false;
    }
}

// Display training results
function displayTrainingResults(data) {
    const history = data.training_history;
    const metrics = data.test_metrics;
    
    // Update progress
    document.getElementById('trainingProgress').style.width = '100%';
    document.getElementById('currentEpoch').textContent = `Epoch: ${history.loss.length}/${history.loss.length}`;
    
    // Update metrics
    document.getElementById('trainLoss').textContent = history.loss[history.loss.length - 1].toFixed(4);
    document.getElementById('valLoss').textContent = history.val_loss[history.val_loss.length - 1].toFixed(4);
    
    // Display test metrics
    const testMetrics = document.getElementById('testMetrics');
    testMetrics.innerHTML = `
        <div class="metric-card">
            <span class="metric-label">RMSE</span>
            <span class="metric-value">${metrics.rmse.toFixed(4)}</span>
        </div>
        <div class="metric-card">
            <span class="metric-label">MAE</span>
            <span class="metric-value">${metrics.mae.toFixed(4)}</span>
        </div>
        <div class="metric-card">
            <span class="metric-label">MAPE</span>
            <span class="metric-value">${metrics.mape.toFixed(2)}%</span>
        </div>
        <div class="metric-card">
            <span class="metric-label">R² Score</span>
            <span class="metric-value">${metrics.r2_score.toFixed(4)}</span>
        </div>
    `;
    
    document.getElementById('testResultsCard').style.display = 'block';
    
    // Create training chart
    createTrainingChart(history);
}

// Create training chart
function createTrainingChart(history) {
    const ctx = document.getElementById('trainingChart').getContext('2d');
    
    if (trainingChart) {
        trainingChart.destroy();
    }
    
    trainingChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: Array.from({length: history.loss.length}, (_, i) => i + 1),
            datasets: [{
                label: 'Training Loss',
                data: history.loss,
                borderColor: '#00ff88',
                backgroundColor: 'rgba(0, 255, 136, 0.1)',
                tension: 0.4
            }, {
                label: 'Validation Loss',
                data: history.val_loss,
                borderColor: '#ff6b6b',
                backgroundColor: 'rgba(255, 107, 107, 0.1)',
                tension: 0.4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                legend: {
                    labels: {color: '#ffffff'}
                }
            },
            scales: {
                x: {
                    ticks: {color: '#ffffff'},
                    grid: {color: 'rgba(255, 255, 255, 0.1)'}
                },
                y: {
                    ticks: {color: '#ffffff'},
                    grid: {color: 'rgba(255, 255, 255, 0.1)'}
                }
            }
        }
    });
}

// Generate forecast
async function generateForecast() {
    try {
        updateStatus('Generating forecast...');
        
        const config = {
            horizon_type: document.getElementById('forecastHorizon').value,
            steps: parseInt(document.getElementById('forecastSteps').value),
            lookback: parseInt(document.getElementById('lookback').value)
        };
        
        const response = await fetch('/api/forecast', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(config)
        });
        
        const data = await response.json();
        
        if (data.success) {
            displayForecastResults(data.forecast);
            updateStatus('Forecast generated successfully');
            showNotification('Forecast completed', 'success');
        } else {
            showNotification(data.error, 'error');
        }
    } catch (error) {
        console.error('Error generating forecast:', error);
        showNotification('Error generating forecast', 'error');
    }
}

// Display forecast results
function displayForecastResults(forecast) {
    document.getElementById('forecastResultsCard').style.display = 'block';
    
    // Create forecast table
    const forecastTable = document.getElementById('forecastTable');
    let tableHTML = '<table><thead><tr><th>Date</th><th>Forecast</th></tr></thead><tbody>';
    
    forecast.dates.forEach((date, i) => {
        const value = forecast.predictions[i];
        tableHTML += `<tr><td>${date}</td><td>${typeof value === 'number' ? value.toFixed(4) : value[0].toFixed(4)}</td></tr>`;
    });
    
    tableHTML += '</tbody></table>';
    forecastTable.innerHTML = tableHTML;
    
    // Create forecast chart
    createForecastChart(forecast);
}

// Create forecast chart
function createForecastChart(forecast) {
    const ctx = document.getElementById('forecastChart').getContext('2d');
    
    if (forecastChart) {
        forecastChart.destroy();
    }
    
    const predictions = forecast.predictions.map(p => typeof p === 'number' ? p : p[0]);
    
    forecastChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: forecast.dates,
            datasets: [{
                label: 'Forecast',
                data: predictions,
                borderColor: '#00ff88',
                backgroundColor: 'rgba(0, 255, 136, 0.1)',
                tension: 0.4,
                fill: true
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                legend: {
                    labels: {color: '#ffffff'}
                }
            },
            scales: {
                x: {
                    ticks: {color: '#ffffff'},
                    grid: {color: 'rgba(255, 255, 255, 0.1)'}
                },
                y: {
                    ticks: {color: '#ffffff'},
                    grid: {color: 'rgba(255, 255, 255, 0.1)'}
                }
            }
        }
    });
}

// Evaluate model
async function evaluateModel() {
    try {
        updateStatus('Evaluating model...');
        
        const config = {
            lookback: parseInt(document.getElementById('lookback').value)
        };
        
        const response = await fetch('/api/evaluate', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(config)
        });
        
        const data = await response.json();
        
        if (data.success) {
            displayEvaluationResults(data);
            updateStatus('Evaluation completed');
            showNotification('Evaluation completed', 'success');
        } else {
            showNotification(data.error, 'error');
        }
    } catch (error) {
        console.error('Error evaluating model:', error);
        showNotification('Error evaluating model', 'error');
    }
}

// Display evaluation results
function displayEvaluationResults(data) {
    const metrics = data.metrics;
    const errorDist = data.error_distribution;
    
    // Display performance metrics
    const performanceMetrics = document.getElementById('performanceMetrics');
    performanceMetrics.innerHTML = `
        <div class="metric-card">
            <span class="metric-label">RMSE</span>
            <span class="metric-value">${metrics.rmse.toFixed(4)}</span>
        </div>
        <div class="metric-card">
            <span class="metric-label">MAE</span>
            <span class="metric-value">${metrics.mae.toFixed(4)}</span>
        </div>
        <div class="metric-card">
            <span class="metric-label">MAPE</span>
            <span class="metric-value">${metrics.mape.toFixed(2)}%</span>
        </div>
        <div class="metric-card">
            <span class="metric-label">R² Score</span>
            <span class="metric-value">${metrics.r2_score.toFixed(4)}</span>
        </div>
        <div class="metric-card">
            <span class="metric-label">Directional Accuracy</span>
            <span class="metric-value">${metrics.directional_accuracy.toFixed(2)}%</span>
        </div>
    `;
    
    document.getElementById('metricsCard').style.display = 'block';
    
    // Display error distribution
    const errorDistribution = document.getElementById('errorDistribution');
    errorDistribution.innerHTML = `
        <div class="stat-item">
            <span class="stat-label">Mean Error</span>
            <span class="stat-value">${errorDist.mean_error.toFixed(4)}</span>
        </div>
        <div class="stat-item">
            <span class="stat-label">Std Error</span>
            <span class="stat-value">${errorDist.std_error.toFixed(4)}</span>
        </div>
        <div class="stat-item">
            <span class="stat-label">Min Error</span>
            <span class="stat-value">${errorDist.min_error.toFixed(4)}</span>
        </div>
        <div class="stat-item">
            <span class="stat-label">Max Error</span>
            <span class="stat-value">${errorDist.max_error.toFixed(4)}</span>
        </div>
    `;
    
    document.getElementById('errorAnalysisCard').style.display = 'block';
}

// Update status
function updateStatus(message) {
    document.getElementById('systemStatus').querySelector('.status-text').textContent = message;
}

// Show notification
function showNotification(message, type = 'info') {
    // Log to console with appropriate level
    if (type === 'error') {
        console.error(`[${type.toUpperCase()}] ${message}`);
    } else if (type === 'warning') {
        console.warn(`[${type.toUpperCase()}] ${message}`);
    } else {
        console.log(`[${type.toUpperCase()}] ${message}`);
    }
    
    // Only show alert for errors (you can implement a toast notification system instead)
    if (type === 'error') {
        alert(`ERROR: ${message}`);
    }
}
