// Add these functions at the beginning of the file
function goBack(currentStage) {
    const previousStage = currentStage - 1;
    if (previousStage >= 1) {
        showStage(previousStage);
        
        // Clear data from current stage
        if (currentStage === 3) {
            clearStage2State();
            showStage(2);
        } else if (currentStage === 2) {
            clearStage2State();
            showStage(1);
        }
    }
}

// Function to clear state when moving away from stage 2
function clearStage2State() {
    processingState.reset();
    const processedImage = document.getElementById('processed-image');
    const warpedImage = document.getElementById('warped-image');
    const reviewControls = document.getElementById('review-controls');
    if (processedImage) processedImage.src = '';
    if (warpedImage) warpedImage.src = '';
    if (reviewControls) reviewControls.style.display = 'none';
}

// Store processing data in a module-level object instead of window
const processingState = {
    columnData: null,
    headerData: null,
    reset() {
        this.columnData = null;
        this.headerData = null;
        console.log('Processing state reset');
    }
};

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
    
    // Clear previous state before processing new sheet
    clearStage2State();
    
    // Get references to DOM elements
    const processedImage = document.getElementById('processed-image');
    const warpedImage = document.getElementById('warped-image');
    const reviewControls = document.getElementById('review-controls');

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
            console.log('Columns received:', data.columns?.length);
            console.log('Header received:', !!data.header);

            // Display stage 2 images
            if (data.processed_image) {
                processedImage.src = `data:image/jpeg;base64,${data.processed_image}`;
            }
            if (data.warped_image) {
                warpedImage.src = `data:image/jpeg;base64,${data.warped_image}`;
            }

            // Store data in our state management object
            if (data.columns && data.columns.length > 0) {
                processingState.columnData = data.columns;
                console.log('Column data stored:', processingState.columnData.length);
            } else {
                console.warn('No column data in response');
            }
            if (data.header) {
                processingState.headerData = data.header;
                console.log('Header data stored:', !!processingState.headerData);
            } else {
                console.warn('No header data in response');
            }

            // Show review controls
            reviewControls.style.display = 'block';
            
            // Set up the continue button with our new function
            setupContinueButton();

        } else {
            throw new Error(data.error || 'Failed to process image');
        }
    } catch (error) {
        console.error('Error:', error);
        alert('Error processing image: ' + error.message);
    }
});

// Define the click handler function outside so we can reference it for both adding and removing
function handleContinueToStage3(e) {
    console.log('=== Continue Button Clicked ===');
    e.preventDefault();
    
    if (!processingState.columnData || processingState.columnData.length === 0) {
        console.error('No column data available. Please process a student sheet first.');
        alert('Please process a student sheet first to get the column data.');
        return;
    }

    // First call the base stage transition
    proceedToStage3();
    
    // Then handle our specific column display logic
    console.log('Setting up column display...');
    setTimeout(() => {
        const columnsSection = document.getElementById('columns-section');
        if (columnsSection) {
            // Display columns using the dedicated function
            displayColumns(processingState.columnData);

            // Handle header image
            const headerImage = document.getElementById('header-image');
            if (headerImage && processingState.headerData) {
                console.log('Setting header image source');
                headerImage.src = `data:image/jpeg;base64,${processingState.headerData}`;
            }
        }
    }, 100); // Small delay to ensure DOM is ready
}

// Add a function to set up the continue button
function setupContinueButton() {
    console.log('Setting up continue button...');
    const continueButton = document.getElementById('continue-to-stage3');
    if (continueButton) {
        console.log('Found continue button, adding click handler');
        // Remove any existing listeners first
        continueButton.removeEventListener('click', handleContinueToStage3);
        continueButton.addEventListener('click', handleContinueToStage3);
        
        // Also add an onclick attribute as a backup
        continueButton.onclick = handleContinueToStage3;
        console.log('Click handler added to continue button');
    } else {
        console.error('Continue button not found');
    }
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
                columns: processingState.columnData
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

    console.log('++ process_all_columns');
    
    if (!processingState.columnData) {
        console.error('No column data available');
        alert('No columns available to process. Please process a student sheet first.');
        return;
    }

    const modelName = document.getElementById('gemini-model-select').value;
    console.log('Selected model:', modelName);
    console.log('Column data available:', processingState.columnData ? 'Yes' : 'No');
    console.log('Number of columns:', processingState.columnData ? processingState.columnData.length : 0);
    
    const button = this;
    
    // Show loading state
    button.disabled = true;
    const originalContent = button.innerHTML;
    button.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Processing all columns...';
    
    try {
        console.log('Preparing API request to /process_all_columns');
        console.log('Request payload:', {
            model_name: modelName,
            columns: processingState.columnData
        });
        
        const response = await fetch('/process_all_columns', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                model_name: modelName,
                columns: processingState.columnData
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
            // Function to update column results
            const updateColumnResults = () => {
                // Display all_response (if you're using the previous suggestion)
                if (result.all_response) {
                    const allResponseDiv = document.createElement('div');
                    allResponseDiv.className = 'alert alert-secondary mt-4';
                    allResponseDiv.innerHTML = `<h6>All Responses:</h6><pre>${JSON.stringify(result.all_response, null, 2)}</pre>`;
                    document.getElementById('columns-container').appendChild(allResponseDiv);
                }

                result.column_results.forEach((columnResult, index) => {
                    const columnDivs = document.querySelectorAll('.column-results');
                    if (columnDivs.length > index && columnDivs[index]) {
                        let answersHtml = '<div class="alert alert-info mt-2"><small>';
                        columnResult.forEach((answer, i) => {
                            answersHtml += `Q${i + 1}: ${answer}<br>`;
                        });
                        answersHtml += '</small></div>';
                        columnDivs[index].innerHTML = answersHtml;
                    } else {
                        console.error(`Could not find .column-results element at index ${index}`);
                    }
                });
            };

            // Use MutationObserver to wait for .column-results to be added to the DOM
            const observer = new MutationObserver((mutationsList, observer) => {
                for (const mutation of mutationsList) {
                    if (mutation.addedNodes.length) {
                        const columnResults = document.querySelectorAll('.column-results');
                        if (columnResults.length === processingState.columnData.length) {
                            updateColumnResults();
                            observer.disconnect(); // Stop observing once we've updated the results
                            break;
                        }
                    }
                }
            });

            // Start observing the columns-container for added nodes
            const columnsContainer = document.getElementById('columns-container');
            observer.observe(columnsContainer, { childList: true, subtree: true });

            // Optional: Fallback in case MutationObserver doesn't work as expected
            setTimeout(() => {
                if (document.querySelectorAll('.column-results').length === processingState.columnData.length) {
                updateColumnResults();
                observer.disconnect();
                }
            }, 2000); // 2 second fallback

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

// Update proceedToStage3 to use the processingState
function proceedToStage3() {
    console.log('=== Stage 3 Transition Started ===');
    console.log('Column data available:', !!processingState.columnData);
    console.log('Number of columns:', processingState.columnData ? processingState.columnData.length : 0);
    console.log('Header data available:', !!processingState.headerData);

    // Check if we have the required data
    if (!processingState.columnData || processingState.columnData.length === 0) {
        console.error('No column data available. Please process a student sheet first.');
        alert('Please process a student sheet first to get the column data.');
        return;
    }

    // First show stage 3
    showStage(3);

    // Important: Wait for a moment to ensure DOM is updated after stage change
    setTimeout(() => {
        // Get and show the columns section
        const columnsSection = document.getElementById('columns-section');
        console.log('Columns section found:', columnsSection ? 'Yes' : 'No');
        
        if (columnsSection) {
            console.log('Making columns section visible');
            columnsSection.style.display = 'block';

            // Display columns using the dedicated function
            displayColumns(processingState.columnData);

            // Handle header image separately
            const headerImage = document.getElementById('header-image');
            if (headerImage && processingState.headerData) {
                console.log('Setting header image source');
                // Create a new image to preload
                const tempImg = new Image();
                tempImg.onload = function() {
                    headerImage.src = this.src;
                    console.log('Header image loaded successfully');
                };
                tempImg.onerror = function() {
                    console.error('Failed to load header image');
                };
                tempImg.src = `data:image/jpeg;base64,${processingState.headerData}`;
            } else {
                console.log('Cannot set header image:', !headerImage ? 'element not found' : 'no header data');
            }
        } else {
            console.error('Could not find columns-section element');
        }
    }, 100); // Small delay to ensure DOM is ready
}

// Update displayColumns function to properly handle image loading
function displayColumns(columns) {
    console.log('=== Display Columns Called ===');
    console.log('Columns to display:', columns?.length);
    
    const container = document.getElementById('columns-container');
    console.log('Container found:', !!container);
    if (!container || !columns) {
        console.error('Missing container or columns data');
        return;
    }
    
    // Clear existing content
    container.innerHTML = '';
    
    // Create row element
    const row = document.createElement('div');
    row.className = 'row';
    container.appendChild(row);
    
    // Track loaded images
    let loadedImages = 0;
    const totalImages = columns.length;
    
    columns.forEach((column, index) => {
        console.log(`Creating column ${index + 1}`);
        const colDiv = document.createElement('div');
        colDiv.className = 'col-md-3';
        
        // Create column preview container
        const previewDiv = document.createElement('div');
        previewDiv.className = 'column-preview';
        
        // Add heading
        const heading = document.createElement('h6');
        heading.textContent = `Column ${index + 1}`;
        previewDiv.appendChild(heading);
        
        // Create and set up image
        const img = document.createElement('img');
        img.className = 'img-fluid';
        
        // Set up image loading handlers
        img.onload = function() {
            console.log(`Column ${index + 1} image loaded successfully`);
            loadedImages++;
            if (loadedImages === totalImages) {
                console.log('All column images loaded');
            }
        };
        
        img.onerror = function() {
            console.error(`Failed to load column ${index + 1} image`);
            loadedImages++;
        };
        
        // Set image source after setting up handlers
        img.src = `data:image/jpeg;base64,${column}`;
        previewDiv.appendChild(img);
        
        // Add results container
        const resultsDiv = document.createElement('div');
        resultsDiv.className = 'column-results mt-2';
        previewDiv.appendChild(resultsDiv);
        
        // Add everything to the column
        colDiv.appendChild(previewDiv);
        row.appendChild(colDiv);
    });
    
    console.log('Display columns setup completed');
}