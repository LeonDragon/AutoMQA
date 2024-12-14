// Add these functions at the beginning of the file
function goBack(currentStage) {
    const previousStage = currentStage - 1;
    if (previousStage >= 1) {
        showStage(previousStage);
        
        // Clear data from current stage
        if (currentStage === 3) {
            // Clear all images
            const columnsContainer = document.getElementById('columns-container');
            const headerImage = document.getElementById('header-image');
            if (columnsContainer) {
                columnsContainer.innerHTML = '';
            }
            if (headerImage) {
                headerImage.src = '';
            }
            window.columnData = null;
            document.getElementById('columns-section').style.display = 'none';
        } else if (currentStage === 4) {
            const geminiResults = document.getElementById('gemini-results');
            if (geminiResults) {
                geminiResults.innerHTML = '';
            }
        }
    }
}

// Add back buttons to each stage's HTML through JavaScript
document.addEventListener('DOMContentLoaded', function() {
    // Add back buttons to stages 2, 3, and 4
    const stages = [2, 3, 4];
    stages.forEach(stageNum => {
        const stage = document.getElementById(`stage${stageNum}`);
        if (stage) {
            const cardHeader = stage.querySelector('.card-header');
            if (cardHeader) {
                // Create back button
                const backButton = document.createElement('button');
                backButton.className = 'btn btn-outline-light btn-sm float-end';
                backButton.innerHTML = '<i class="fas fa-arrow-left me-2"></i>Back';
                backButton.onclick = () => goBack(stageNum);
                
                // Add button to card header
                cardHeader.appendChild(backButton);
            }
        }
    });
});

// Update the showStage function to handle stage transitions
function showStage(stageNumber) {
    // Hide all stages
    document.querySelectorAll('.stage').forEach(stage => {
        stage.classList.remove('active');
    });
    
    // Show the selected stage
    const currentStage = document.getElementById(`stage${stageNumber}`);
    if (currentStage) {
        currentStage.classList.add('active');
    }
    
    // Update progress steps
    document.querySelectorAll('.step').forEach((step, index) => {
        if (index + 1 < stageNumber) {
            step.classList.add('completed');
            step.classList.remove('active');
        } else if (index + 1 === stageNumber) {
            step.classList.add('active');
            step.classList.remove('completed');
        } else {
            step.classList.remove('completed', 'active');
        }
    });
    
    // Update progress bar
    const progressBar = document.getElementById('progress-bar');
    if (progressBar) {
        const progress = ((stageNumber - 1) / 3) * 100;
        progressBar.style.width = `${progress}%`;
        progressBar.setAttribute('aria-valuenow', progress);
    }
}

// Handle answer key upload
document.getElementById('answer-key-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    const formData = new FormData();
    const file = document.getElementById('answer-key').files[0];
    formData.append('answer_key', file);

    try {
        const response = await fetch('/process_answer_key', {
            method: 'POST',
            body: formData
        });
        const data = await response.json();

        if (data.success) {
            const resultsDiv = document.getElementById('answer-key-results');
            resultsDiv.innerHTML = '<div class="alert alert-success">Answer key processed successfully!</div>';
            showStage(2);
        } else {
            throw new Error(data.error);
        }
    } catch (error) {
        document.getElementById('answer-key-results').innerHTML = 
            `<div class="alert alert-danger">Error: ${error.message}</div>`;
    }
});

// Handle student sheet processing
document.getElementById('student-sheet-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    // Get references to DOM elements
    const processedImage = document.getElementById('processed-image');
    const warpedImage = document.getElementById('warped-image');
    const reviewControls = document.getElementById('review-controls');
    
    // Clear previous results
    processedImage.src = '';
    warpedImage.src = '';
    reviewControls.style.display = 'none';

    // Validate file input
    const fileInput = document.getElementById('student-sheet');
    if (!fileInput || fileInput.files.length === 0) {
        alert('Please select a file first');
        return;
    }

    // Create form data with settings
    const formData = new FormData();
    formData.append('student_sheet', fileInput.files[0]);
    
    // Add processing settings
    Object.entries(settings).forEach(([key, value]) => {
        formData.append(key, value);
    });

    try {
        const response = await fetch('/process_student_sheet', {
            method: 'POST',
            body: formData
        });
        const data = await response.json();
        
        if (data.success) {
            console.log('=== Stage 2 Data Received ===');
            console.log('Raw data:', data);  // Log the entire response
            console.log('Columns received:', data.columns?.length);
            console.log('Header received:', !!data.header);

            // Display stage 2 images
            if (data.processed_image) {
                processedImage.src = `data:image/jpeg;base64,${data.processed_image}`;
            }
            if (data.warped_image) {
                warpedImage.src = `data:image/jpeg;base64,${data.warped_image}`;
            }

            // Store data for stage 3
            if (data.columns && data.columns.length > 0) {
                window.columnData = data.columns;
                console.log('Column data stored in window:', window.columnData.length);
            } else {
                console.warn('No column data in response');
            }
            if (data.header) {
                window.headerData = data.header;
                console.log('Header data stored in window:', !!window.headerData);
            } else {
                console.warn('No header data in response');
            }

            // Show review controls
            reviewControls.style.display = 'block';
        } else {
            throw new Error(data.error || 'Failed to process image');
        }
    } catch (error) {
        console.error('Error:', error);
        alert('Error processing image: ' + error.message);
    }
});

function displayColumns(columns) {
    console.log('=== Display Columns Called ===');
    console.log('Columns to display:', columns?.length);
    
    const container = document.getElementById('columns-container');
    console.log('Container found:', !!container);
    if (!container || !columns) {
        console.error('Missing container or columns data');
        return;
    }
    
    container.innerHTML = '';
    
    const row = document.createElement('div');
    row.className = 'row';
    
    columns.forEach((column, index) => {
        console.log(`Creating column ${index + 1}`);
        const colDiv = document.createElement('div');
        colDiv.className = 'col-md-3';
        
        colDiv.innerHTML = `
            <div class="column-preview">
                <h6>Column ${index + 1}</h6>
                <img src="data:image/jpeg;base64,${column}" 
                     class="img-fluid"
                     onload="console.log('Column ${index + 1} image loaded')"
                     onerror="console.error('Column ${index + 1} image failed to load')">
                <div class="column-results mt-2"></div>
            </div>
        `;
        row.appendChild(colDiv);
    });
    
    container.appendChild(row);
    console.log('Display columns completed');
}

// Handle Gemini processing
document.getElementById('gemini-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    const modelName = document.getElementById('model-select').value;

    try {
        const response = await fetch('/process_with_gemini', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                model_name: modelName,
                columns: columnData
            })
        });
        const data = await response.json();

        if (data.success) {
            console.log('Received columns:', data.columns?.length);
            console.log('Received header:', data.header ? 'yes' : 'no');
            let resultsHtml = '<div class="alert alert-success"><h5>Results:</h5>';
            for (const [testCode, score] of Object.entries(data.scores)) {
                resultsHtml += `<p>Test Code ${testCode}: ${score !== null ? score.toFixed(2) + '%' : 'Failed to calculate'}</p>`;
            }
            resultsHtml += '</div>';
            document.getElementById('gemini-results').innerHTML = resultsHtml;
        } else {
            throw new Error(data.error);
        }
    } catch (error) {
        document.getElementById('gemini-results').innerHTML = 
            `<div class="alert alert-danger">Error: ${error.message}</div>`;
    }
});

// Add click handler for processing all columns
document.getElementById('process-all-columns').addEventListener('click', async function() {
    console.log('Button clicked - Starting Gemini processing');
    
    if (!columnData) {
        console.error('No column data available');
        alert('No columns available to process. Please process a student sheet first.');
        return;
    }

    const modelName = document.getElementById('gemini-model-select').value;
    console.log('Selected model:', modelName);
    console.log('Column data available:', columnData ? 'Yes' : 'No');
    console.log('Number of columns:', columnData ? columnData.length : 0);
    
    const button = this;
    
    // Show loading state
    button.disabled = true;
    const originalContent = button.innerHTML;
    button.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Processing all columns...';
    
    try {
        console.log('Preparing API request to /process_all_columns');
        console.log('Request payload:', {
            model_name: modelName,
            columns: columnData
        });
        
        const response = await fetch('/process_all_columns', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                model_name: modelName,
                columns: columnData
            })
        });

        console.log('Response received:', response.status, response.statusText);

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const result = await response.json();
        console.log('Processing result:', result);

        // Update UI with results
        if (result.success) {
            result.column_results.forEach((columnResult, index) => {
                const columnDiv = document.querySelectorAll('.column-results')[index];
                if (columnDiv) {
                    let answersHtml = '<div class="alert alert-info mt-2"><small>';
                    columnResult.forEach((answer, i) => {
                        answersHtml += `Q${i + 1}: ${answer}<br>`;
                    });
                    answersHtml += '</small></div>';
                    columnDiv.innerHTML = answersHtml;
                }
            });

            // Show success message
            const successDiv = document.createElement('div');
            successDiv.className = 'alert alert-success mt-4';
            successDiv.innerHTML = '<i class="fas fa-check-circle me-2"></i>All columns processed successfully!';
            document.getElementById('columns-container').appendChild(successDiv);
        }
    } catch (error) {
        console.error('Error:', error);
        // Show error message
        document.querySelectorAll('.column-results').forEach(div => {
            div.innerHTML = `
                <div class="alert alert-danger mt-2">
                    <i class="fas fa-exclamation-circle me-2"></i>
                    <small>${error.message}</small>
                </div>
            `;
        });
    } finally {
        // Restore button state
        button.disabled = false;
        button.innerHTML = originalContent;
    }
});

// Update the proceedToStage3 function
function proceedToStage3() {
    console.log('=== Stage 3 Transition Started ===');
    console.log('Column data available:', window.columnData ? 'Yes' : 'No');
    console.log('Number of columns:', window.columnData ? window.columnData.length : 0);
    console.log('Header data available:', window.headerData ? 'Yes' : 'No');

    // First show stage 3
    showStage(3);

    // Get and show the columns section
    const columnsSection = document.getElementById('columns-section');
    if (columnsSection) {
        // Make sure the section is visible
        columnsSection.style.display = 'block';

        // Get the containers for columns and header
        const columnsContainer = document.getElementById('columns-container');
        const headerImage = document.getElementById('header-image');

        // Clear any existing content
        columnsContainer.innerHTML = '';

        // Display columns if we have data
        if (window.columnData && window.columnData.length > 0) {
            // Create each column
            window.columnData.forEach((column, index) => {
                const colDiv = document.createElement('div');
                colDiv.className = 'col-md-3'; // Adjust column width for 4 columns

                colDiv.innerHTML = `
                    <div class="column-preview">
                        <h6>Column ${index + 1}</h6>
                        <img src="data:image/jpeg;base64,${column}" class="img-fluid">
                        <div class="column-results mt-2"></div>
                    </div>
                `;
                columnsContainer.appendChild(colDiv);
            });
        }

        // Display header image if we have data
        if (headerImage && window.headerData) {
            headerImage.src = `data:image/jpeg;base64,${window.headerData}`;
        }
    }
}

document.addEventListener('DOMContentLoaded', function() {
    const continueButton = document.querySelector("#review-controls .btn-primary");
    if (continueButton) {
        continueButton.addEventListener('click', proceedToStage3);
    }
});