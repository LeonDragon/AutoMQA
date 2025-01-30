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
    answerKeyData: null,
    reset() {
        this.columnData = null;
        this.headerData = null;
        this.answerKeyData = null;
        console.log('Processing state reset');
    }
};

// Add back buttons to each stage's HTML through JavaScript
document.addEventListener('DOMContentLoaded', function() {
    // Debug check for Check Score button
    const checkScoreBtn = document.getElementById('check-score');
    if (!checkScoreBtn) {
        console.error('Check Score button not found in DOM!');
    } else {
        console.log('Check Score button found in DOM');
    }

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
            
            // Store the answer key data in processingState
            processingState.answerKeyData = data.answer_keys;
            console.log('Answer key data stored:', processingState.answerKeyData);
            
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
            displayColumns(processingState.columnData, data.vertical_groups || []);

            // Handle header image only if it doesn't already exist
            const existingHeader = document.getElementById('header-image-container');
            if (processingState.headerData && !existingHeader) {
                console.log('Setting header image source');
                
                // Create header container
                const headerContainer = document.createElement('div');
                headerContainer.id = 'header-image-container';
                headerContainer.className = 'header-container';
                
                // Create image element
                const headerImage = document.createElement('img');
                headerImage.id = 'header-image';
                headerImage.src = `data:image/jpeg;base64,${processingState.headerData}`;
                headerImage.alt = 'Processed Header';
                
                // Add image to container
                headerContainer.appendChild(headerImage);
                
                // Insert header at the top of the columns container
                const columnsContainer = document.getElementById('columns-container');
                if (columnsContainer) {
                    columnsContainer.insertBefore(headerContainer, columnsContainer.firstChild);
                } else {
                    console.error('Columns container not found');
                }
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

// Function to update results with Gemini response
function updateGeminiResponse(result) {
    console.log('Updating Gemini response:', result);
    
    if (result && result.all_response) {
        // Remove any existing response first
        const existingResponse = document.querySelector('.gemini-response');
        if (existingResponse) {
            existingResponse.remove();
        }

        const allResponseDiv = document.createElement('div');
        allResponseDiv.className = 'alert alert-secondary mt-4 gemini-response';
        allResponseDiv.innerHTML = `<h6>Gemini Response:</h6><div class="mt-2">${result.all_response}</div>`;
        
        const container = document.getElementById('columns-container');
        if (container) {
            container.insertBefore(allResponseDiv, container.firstChild);
        } else {
            console.error('columns-container not found');
        }
    }

    if (result && result.column_results) {
        result.column_results.forEach((columnResult, index) => {
            const columnDivs = document.querySelectorAll('.column-results');
            if (columnDivs.length > index && columnDivs[index]) {
                let answersHtml = '<div class="alert alert-info mt-2"><small>';
                columnResult.forEach((answer, i) => {
                    answersHtml += `{i + 1}: ${answer}<br>`;
                });
                answersHtml += '</small></div>';
                columnDivs[index].innerHTML = answersHtml;
            }
        });
    }
}

// Handle Gemini processing
document.getElementById('gemini-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    const modelName = document.getElementById('model-select').value;
    
    try {
        // Show loading state
        const button = e.target.querySelector('button[type="submit"]');
        const originalContent = button.innerHTML;
        button.disabled = true;
        button.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Processing...';

        const response = await fetch('/process_gemini', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                model: modelName,
                columns: processingState.columnData,
                headers: processingState.headerData
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
            // Function to calculate score and correct count
            const calculateScore = (llmAnswers, groundTruth) => {
                let correct = 0;
                let total = Object.keys(groundTruth).length;
                
                Object.keys(groundTruth).forEach(qNum => {
                    if (llmAnswers[qNum] && groundTruth[qNum] === llmAnswers[qNum]) {
                        correct++;
                    }
                });
                
                return {
                    score: ((correct / total) * 100).toFixed(2),
                    correct: correct,
                    total: total
                };
            };

            // Function to update column results
            const updateColumnResults = () => {
                console.log('Full result object:', result);
                
                // First, combine all LLM responses into a single object
                const combinedResponses = {};
                result.all_responses.forEach(response => {
                    if (response) {
                        // Each response is already a JSON object, just merge it
                        Object.assign(combinedResponses, response);
                    }
                });
                
                console.log('Combined LLM Responses:', combinedResponses);
                console.log('Answer Key Data:', result.answer_key_data);
                
                // Collect all score results
                const scoreResults = {};
                Object.entries(result.answer_key_data).forEach(([examCode, groundTruth]) => {
                    const scoreResult = calculateScore(combinedResponses, groundTruth);
                    scoreResults[examCode] = {
                        score: scoreResult.score,
                        correct: scoreResult.correct,
                        total: scoreResult.total
                    };
                });

                // Create score messages with processing time
                const processingTime = result.processing_time ? `\nProcessing Time: ${result.processing_time.toFixed(2)} seconds` : '';
                const tokenUsage = result.token_usage ? 
                    `\nToken Usage:\n- Input: ${result.token_usage.input_tokens}\n- Output: ${result.token_usage.output_tokens}` : '';
                
                const scoreMessages = Object.entries(scoreResults)
                    .map(([examCode, result]) => 
                        `Exam ${examCode} Score: ${result.score}% (${result.correct}/${result.total} questions)`
                    )
                    .join('\n') + processingTime + tokenUsage;

                // Log to console
                console.log(scoreMessages);
                
                // Create and show popup dialog with all results
                const dialog = document.createElement('dialog');
                dialog.style.cssText = `
                    position: fixed;
                    top: 50%;
                    left: 50%;
                    transform: translate(-50%, -50%);
                    border: none;
                    border-radius: 12px;
                    box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
                    padding: 0;
                    width: 600px;
                    max-width: 90%;
                    z-index: 1000;
                    background: white;
                `;
                
                dialog.innerHTML = `
                    <div style="
                        padding: 30px;
                        background: white;
                        border-radius: 12px;
                    ">
                        <h3 style="
                            margin: 0 0 25px 0;
                            color: #2c3e50;
                            font-size: 28px;
                            text-align: center;
                            border-bottom: 2px solid #eee;
                            padding-bottom: 15px;
                            font-weight: 600;
                        ">
                            <i class="fas fa-chart-bar me-2"></i>
                            Scoring Results
                        </h3>
                        <div style="
                            background: #f8f9fa;
                            padding: 20px;
                            border-radius: 8px;
                            margin-bottom: 25px;
                            font-family: monospace;
                            font-size: 16px;
                            line-height: 1.6;
                            max-height: 400px;
                            overflow-y: auto;
                        ">
                            ${scoreMessages.split('\n').map(msg => {
                                if (msg.includes('Processing Time') || msg.includes('Token Usage')) {
                                    return `<div style="color: #666; margin-top: 10px;">${msg}</div>`;
                                }
                                return `<div>${msg}</div>`;
                            }).join('')}
                        </div>
                        <div style="text-align: center;">
                            <button onclick="this.closest('dialog').close()" style="
                                background: #007bff;
                                color: white;
                                border: none;
                                padding: 10px 30px;
                                border-radius: 6px;
                                cursor: pointer;
                                font-size: 16px;
                                font-weight: 500;
                                transition: all 0.3s;
                            ">
                                <i class="fas fa-check-circle me-2"></i>
                                Close Results
                            </button>
                        </div>
                    </div>
                `;

                // Add hover effect to the close button
                const closeButton = dialog.querySelector('button');
                closeButton.addEventListener('mouseover', () => {
                    closeButton.style.background = '#0056b3';
                });
                closeButton.addEventListener('mouseout', () => {
                    closeButton.style.background = '#007bff';
                });

                document.body.appendChild(dialog);
                dialog.showModal();
                
                // Display responses in their respective panels
                result.all_responses.forEach((response, index) => {
                    const responsePanel = document.getElementById(`response-panel-${index}`);
                    if (responsePanel) {
                        responsePanel.innerHTML = formatResponse(response);
                    }
                });
            };

            // Update each response panel with the results
            result.all_responses.forEach((response, index) => {
                const responsePanel = document.getElementById(`response-panel-${index}`);
                if (responsePanel) {
                    responsePanel.innerHTML = formatResponse(response);
                }
            });

            // Show the Check Score button with enhanced debug logging
            const checkScoreBtn = document.getElementById('check-score');
            if (checkScoreBtn) {
                console.log('=== Making Check Score button visible ===');
                console.log('Button element:', checkScoreBtn);
                console.log('Initial class list:', checkScoreBtn.classList);
                console.log('Initial computed styles:', {
                    display: window.getComputedStyle(checkScoreBtn).display,
                    visibility: window.getComputedStyle(checkScoreBtn).visibility,
                    opacity: window.getComputedStyle(checkScoreBtn).opacity
                });
                
                // Remove hidden class if present
                checkScoreBtn.classList.remove('hidden');
                
                // Force reflow to ensure CSS transition works
                void checkScoreBtn.offsetHeight;
                
                console.log('After removing hidden class:', {
                    classList: checkScoreBtn.classList,
                    computedStyles: {
                        display: window.getComputedStyle(checkScoreBtn).display,
                        visibility: window.getComputedStyle(checkScoreBtn).visibility,
                        opacity: window.getComputedStyle(checkScoreBtn).opacity
                    }
                });
                
                // Add event listener for transition end
                checkScoreBtn.addEventListener('transitionend', () => {
                    console.log('Check Score button transition complete:', {
                        visibility: window.getComputedStyle(checkScoreBtn).visibility,
                        opacity: window.getComputedStyle(checkScoreBtn).opacity
                    });
                });
            } else {
                console.error('Check Score button not found in DOM');
            }

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
        // Reset button state
        const button = e.target.querySelector('button[type="submit"]');
        button.disabled = false;
        button.innerHTML = originalContent;
    }
});

// Function to calculate and display scores
function updateColumnResults() {
    console.log('=== Calculating Scores ===');
    
    // Get the latest results from the response panels
    const columnDivs = document.querySelectorAll('.column-results');
    const allResponses = [];
    
    columnDivs.forEach(columnDiv => {
        const answers = {};
        const answerElements = columnDiv.querySelectorAll('.response-question');
        answerElements.forEach(el => {
            const qNum = el.querySelector('.badge').textContent;
            const answer = el.querySelector('.response-answer').textContent;
            answers[qNum] = answer;
        });
        allResponses.push(answers);
    });
    
    // Combine all responses into a single object
    const combinedResponses = {};
    allResponses.forEach(response => {
        Object.assign(combinedResponses, response);
    });
    
    console.log('Combined Responses:', combinedResponses);
    
    // Get answer key data from processingState
    if (!processingState.answerKeyData) {
        console.error('No answer key data available');
        console.log('Current processingState:', processingState);
        alert('No answer key data available. Please process an answer key first.');
        return;
    }
    
    // Calculate scores for each exam code
    const scoreResults = {};
    Object.entries(processingState.answerKeyData).forEach(([examCode, groundTruth]) => {
        let correct = 0;
        let total = Object.keys(groundTruth).length;
        
        Object.keys(groundTruth).forEach(qNum => {
            if (combinedResponses[qNum] && groundTruth[qNum] === combinedResponses[qNum]) {
                correct++;
            }
        });
        
        const score = ((correct / total) * 100).toFixed(2);
        scoreResults[examCode] = {
            score: score,
            correct: correct,
            total: total
        };
    });
    
    // Create score messages with processing time
    const processingTime = result.processing_time ? `\nProcessing Time: ${result.processing_time.toFixed(2)} seconds` : '';
    const tokenUsage = result.token_usage ? 
        `\nToken Usage:\n- Input: ${result.token_usage.input_tokens}\n- Output: ${result.token_usage.output_tokens}` : '';
            
    const scoreMessages = Object.entries(scoreResults)
        .map(([examCode, result]) => 
            `Exam ${examCode} Score: ${result.score}% (${result.correct}/${result.total} questions)`
        )
        .join('\n') + processingTime + tokenUsage;
    
    console.log('Score Results:', scoreMessages);
    
    // Create and show popup dialog with all results
    const dialog = document.createElement('dialog');
    dialog.style.cssText = `
        position: fixed;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        border: none;
        border-radius: 12px;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
        padding: 0;
        width: 600px;
        max-width: 90%;
        z-index: 1000;
        background: white;
    `;
    
    dialog.innerHTML = `
        <div style="
            padding: 30px;
            background: white;
            border-radius: 12px;
        ">
            <h3 style="
                margin: 0 0 25px 0;
                color: #2c3e50;
                font-size: 28px;
                text-align: center;
                border-bottom: 2px solid #eee;
                padding-bottom: 15px;
                font-weight: 600;
            ">
                <i class="fas fa-chart-bar me-2"></i>
                Scoring Results
            </h3>
            <div style="
                background: #f8f9fa;
                padding: 20px;
                border-radius: 8px;
                margin-bottom: 25px;
                font-family: monospace;
                font-size: 16px;
                line-height: 1.6;
                max-height: 400px;
                overflow-y: auto;
            ">
                ${scoreMessages.split('\n').map(msg => `<div>${msg}</div>`).join('')}
            </div>
            <div style="text-align: center;">
                <button onclick="this.closest('dialog').close()" style="
                    background: #007bff;
                    color: white;
                    border: none;
                    padding: 10px 30px;
                    border-radius: 6px;
                    cursor: pointer;
                    font-size: 16px;
                    font-weight: 500;
                    transition: all 0.3s;
                ">
                    <i class="fas fa-check-circle me-2"></i>
                    Close Results
                </button>
            </div>
        </div>
    `;

    // Add hover effect to the close button
    const closeButton = dialog.querySelector('button');
    closeButton.addEventListener('mouseover', () => {
        closeButton.style.background = '#0056b3';
    });
    closeButton.addEventListener('mouseout', () => {
        closeButton.style.background = '#007bff';
    });

    document.body.appendChild(dialog);
    dialog.showModal();
}

// Add click handler for Check Score button
document.getElementById('check-score').addEventListener('click', function() {
    updateColumnResults();
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
            // Function to calculate score and correct count
            const calculateScore = (llmAnswers, groundTruth) => {
                let correct = 0;
                let total = Object.keys(groundTruth).length;
                
                Object.keys(groundTruth).forEach(qNum => {
                    if (llmAnswers[qNum] && groundTruth[qNum] === llmAnswers[qNum]) {
                        correct++;
                    }
                });
                
                return {
                    score: ((correct / total) * 100).toFixed(2),
                    correct: correct,
                    total: total
                };
            };

            // Function to update column results
            const updateColumnResults = () => {
                console.log('Full result object:', result);
                
                // First, combine all LLM responses into a single object
                const combinedResponses = {};
                result.all_responses.forEach(response => {
                    if (response) {
                        // Each response is already a JSON object, just merge it
                        Object.assign(combinedResponses, response);
                    }
                });
                
                console.log('Combined LLM Responses:', combinedResponses);
                console.log('Answer Key Data:', result.answer_key_data);
                
                // Collect all score results
                const scoreResults = {};
                Object.entries(result.answer_key_data).forEach(([examCode, groundTruth]) => {
                    const scoreResult = calculateScore(combinedResponses, groundTruth);
                    scoreResults[examCode] = {
                        score: scoreResult.score,
                        correct: scoreResult.correct,
                        total: scoreResult.total
                    };
                });

                // Create score messages
                const processingTime = result.processing_time ? `\nProcessing Time: ${result.processing_time.toFixed(2)} seconds` : '';
                const tokenUsage = result.token_usage ? 
                    `\nToken Usage:\n- Input: ${result.token_usage.input_tokens}\n- Output: ${result.token_usage.output_tokens}` : '';
                
                const scoreMessages = Object.entries(scoreResults)
                    .map(([examCode, result]) => 
                        `Exam ${examCode} Score: ${result.score}% (${result.correct}/${result.total} questions)`
                    )
                    .join('\n') + processingTime + tokenUsage;

                // Log to console
                console.log(scoreMessages);
                
                // Create and show popup dialog with all results
                const dialog = document.createElement('dialog');
                dialog.style.cssText = `
                    border: none;
                    border-radius: 8px;
                    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                    padding: 0;
                    max-width: 500px;
                    width: 90%;
                `;
                
                dialog.innerHTML = `
                    <div style="
                        padding: 20px;
                        background: white;
                        border-radius: 8px;
                    ">
                        <h3 style="
                            margin: 0 0 20px 0;
                            color: #2c3e50;
                            font-size: 24px;
                            text-align: center;
                            border-bottom: 2px solid #eee;
                            padding-bottom: 10px;
                        ">Scoring Results</h3>
                        <div style="
                            background: #f8f9fa;
                            padding: 15px;
                            border-radius: 6px;
                            margin-bottom: 20px;
                            font-family: monospace;
                            font-size: 14px;
                            line-height: 1.6;
                        ">${scoreMessages.split('\n').map(msg => {
                            if (msg.includes('Processing Time') || msg.includes('Token Usage')) {
                                return `<div style="color: #666; margin-top: 10px;">${msg}</div>`;
                            }
                            return `<div>${msg}</div>`;
                        }).join('')}</div>
                        <div style="text-align: center;">
                            <button onclick="this.closest('dialog').close()" style="
                                background: #007bff;
                                color: white;
                                border: none;
                                padding: 8px 20px;
                                border-radius: 4px;
                                cursor: pointer;
                                font-size: 14px;
                                transition: background 0.3s;
                            ">Close</button>
                        </div>
                    </div>
                `;

                // Add hover effect to the close button
                const closeButton = dialog.querySelector('button');
                closeButton.addEventListener('mouseover', () => {
                    closeButton.style.background = '#0056b3';
                });
                closeButton.addEventListener('mouseout', () => {
                    closeButton.style.background = '#007bff';
                });

                document.body.appendChild(dialog);
                dialog.showModal();
                
                // Display original responses in UI
                const columnDivs = document.querySelectorAll('.column-results');
                columnDivs.forEach((columnDiv, index) => {
                    let answersHtml = '<div class="alert alert-info mt-2"><small>';
                    const response = result.all_responses[index];
                    if (response) {
                        const formattedResponse = Object.entries(response)
                            .sort(([a], [b]) => parseInt(a) - parseInt(b))
                            .map(([num, value]) => {
                                return `<div class="mb-1">
                                    <span class="text-primary">${num}:</span>
                                    <span class="text-success">${value}</span>
                                </div>`;
                            }).join('');
                        answersHtml += formattedResponse;
                    } else {
                        answersHtml += 'No response available';
                    }
                    answersHtml += '</small></div>';
                    columnDiv.innerHTML = answersHtml;
                });
            };

            // Immediately update the response panels with the results
            result.all_responses.forEach((response, index) => {
                const responsePanel = document.getElementById(`response-panel-${index}`);
                if (responsePanel) {
                    responsePanel.innerHTML = formatResponse(response);
                }
            });

            // Show the enhanced scoring dialog
            updateColumnResults();

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
function displayColumns(columns, verticalGroups) {
    console.log('=== Display Columns Called ===');
    console.log('Columns to display:', columns?.length);
    console.log('Vertical groups:', verticalGroups?.length);
    
    const container = document.getElementById('columns-container');
    console.log('Container found:', !!container);
    if (!container || !columns) {
        console.error('Missing container or columns data');
        return;
    }
    
    // Clear existing content
    container.innerHTML = '';
    
    // Track loaded images
    let loadedImages = 0;
    const totalImages = columns.length;
    
    // Create pairs of columns
    for (let i = 0; i < columns.length; i += 2) {
        console.log(`Creating columns ${i + 1} and ${i + 2}`);
        
        // Create main row container for the pair
        const rowDiv = document.createElement('div');
        rowDiv.className = 'column-row';
        
        // Create first column pair
        const columnPair1 = createColumnPair(columns[i], i);
        rowDiv.appendChild(columnPair1);
        
        // Create second column pair if exists
        if (columns[i + 1]) {
            const columnPair2 = createColumnPair(columns[i + 1], i + 1);
            rowDiv.appendChild(columnPair2);
        }
        
        // Add row with column pairs to container
        container.appendChild(rowDiv);
        
        // Add vertical groups for these columns
        [i, i + 1].forEach(colIndex => {
            if (colIndex < columns.length) {
                const groupContainer = document.createElement('div');
                groupContainer.className = 'vertical-groups-container mt-3';
                groupContainer.innerHTML = `
                    <div class="group-header">
                        <h6>Vertical Groups - Column ${colIndex + 1}</h6>
                        <button class="btn btn-sm btn-toggle" 
                                onclick="toggleGroupVisibility('group-${colIndex}')">
                            Toggle Groups
                        </button>
                    </div>
                    <div class="row-group-grid" id="group-${colIndex}" style="display: none;">
                        ${verticalGroups
                            .filter(g => g.column === colIndex)
                            .map(g => `
                                <div class="row-group-item" data-column="${g.column}" data-group="${g.group}">
                                    <img src="data:image/jpeg;base64,${g.image}" 
                                         alt="Group ${g.group + 1}">
                                    <button class="btn-process-group" 
                                            onclick="processVerticalGroup(${g.column}, ${g.group})">
                                        Analyze
                                    </button>
                                </div>
                            `).join('')}
                    </div>
                `;
                container.appendChild(groupContainer);
            }
        });
    }
    
    console.log('Display columns setup completed');
}

function createColumnPair(column, index) {
    const pairDiv = document.createElement('div');
    pairDiv.className = 'column-pair';
    
    // Add combined heading
    const heading = document.createElement('h6');
    heading.innerHTML = `
        <span>Column ${index + 1}</span>
        <span>Gemini Analysis</span>
    `;
    pairDiv.appendChild(heading);
    
    // Create column preview section
    const previewDiv = document.createElement('div');
    previewDiv.className = 'column-preview';
    
    // Add image
    const img = document.createElement('img');
    img.src = `data:image/jpeg;base64,${column}`;
    img.alt = `Column ${index + 1}`;
    img.style.maxHeight = '100%';
    img.style.width = 'auto';
    previewDiv.appendChild(img);
    
    // Add controls
    const controlsDiv = document.createElement('div');
    controlsDiv.className = 'column-controls';
    
    const reprocessBtn = document.createElement('button');
    reprocessBtn.className = 'reprocess-btn';
    reprocessBtn.innerHTML = '<i class="fas fa-sync-alt"></i> Reprocess';
    reprocessBtn.onclick = () => reprocessColumn(index, column);
    controlsDiv.appendChild(reprocessBtn);
    
    previewDiv.appendChild(controlsDiv);
    pairDiv.appendChild(previewDiv);
    
    // Create response panel
    const responseDiv = document.createElement('div');
    responseDiv.className = 'response-panel';
    responseDiv.id = `response-panel-${index}`;
    
    // Add initial loading state
    responseDiv.innerHTML = `
        <div class="text-center py-4">
            <div class="spinner-border text-primary" role="status">
                <span class="visually-hidden">Loading...</span>
            </div>
            <p class="text-muted mt-2">Waiting for Gemini response...</p>
        </div>
    `;
    
    pairDiv.appendChild(responseDiv);
    
    return pairDiv;
}

function formatResponse(response) {
    if (!response) {
        return '<div class="alert alert-warning">No response available</div>';
    }

    // Convert response to array if it's an object
    const responses = Array.isArray(response) ? response : Object.entries(response);

    return `
        <div class="response-item">
            ${responses
                .map(([qNum, answer], index) => {
                    // Handle both array and object responses
                    const questionNumber = typeof qNum === 'number' ? index + 1 : qNum;
                    return `
                        <div class="response-question mb-2">
                            <span class="badge bg-primary me-1">${questionNumber}</span>
                            <span class="response-answer">${answer}</span>
                        </div>`;
                })
                .join('')}
        </div>
    `;
}

async function reprocessColumn(index, columnBase64) {
    const btn = document.querySelectorAll('.reprocess-btn')[index];
    const originalContent = btn.innerHTML;
    
    try {
        // Show loading state
        btn.disabled = true;
        btn.classList.add('loading');
        btn.innerHTML = '<span class="spinner"></span> Processing...';
        
        const response = await fetch('/process_single_column', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                model_name: 'gemini-1.5-flash',
                column: columnBase64,
                index: index,
                recheck: true  // Indicate this is a recheck request
            })
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const result = await response.json();
        
        if (result.success) {
            // Update the specific column's results
            const columnResults = document.querySelectorAll('.column-results')[index];
            if (columnResults) {
                let answersHtml = '<div class="alert alert-info mt-2"><small>';
                const response = result.response;
                if (response) {
                    const formattedResponse = Object.entries(response)
                        .sort(([a], [b]) => parseInt(a) - parseInt(b))
                        .map(([num, value]) => {
                            return `<div class="mb-1">
                                <span class="text-primary">${num}:</span>
                                <span class="text-success">${value}</span>
                            </div>`;
                        }).join('');
                    answersHtml += formattedResponse;
                } else {
                    answersHtml += 'No response available';
                }
                answersHtml += '</small></div>';
                columnResults.innerHTML = answersHtml;
            }
            
            // Show success message
            showToast('Column reprocessed successfully!', 'success');
        } else {
            throw new Error(result.error || 'Failed to reprocess column');
        }
    } catch (error) {
        console.error('Error reprocessing column:', error);
        showToast(`Error: ${error.message}`, 'error');
    } finally {
        // Reset button state
        btn.disabled = false;
        btn.classList.remove('loading');
        btn.innerHTML = originalContent;
    }
}

function showToast(message, type = 'info') {
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.textContent = message;
    
    document.body.appendChild(toast);
    
    setTimeout(() => {
        toast.classList.add('show');
    }, 10);
    
    setTimeout(() => {
        toast.classList.remove('show');
        setTimeout(() => {
            toast.remove();
        }, 300);
    }, 3000);
}
