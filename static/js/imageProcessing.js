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
                    answersHtml += `Q${i + 1}: ${answer}<br>`;
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

            // Show scores and processing time
            const scoreMessages = Object.entries(result.answer_key_data)
                .map(([examCode, score]) => 
                    `Exam ${examCode} Score: ${score.toFixed(2)}%`
                )
                .join('\n') + `\nProcessing Time: ${result.processing_time.toFixed(2)} seconds`;

            // Create and show results dialog
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
                <div style="padding: 20px;">
                    <h3 style="margin: 0 0 20px 0; text-align: center;">Scoring Results</h3>
                    <div style="background: #f8f9fa; padding: 15px; border-radius: 6px; margin-bottom: 20px;">
                        ${scoreMessages.split('\n').map(msg => `<div>${msg}</div>`).join('')}
                    </div>
                    <div style="text-align: center;">
                        <button onclick="this.closest('dialog').close()" 
                            style="background: #007bff; color: white; border: none; padding: 8px 20px; border-radius: 4px; cursor: pointer;">
                            Close
                        </button>
                    </div>
                </div>
            `;

            document.body.appendChild(dialog);
            dialog.showModal();

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
    
    // Track loaded images
    let loadedImages = 0;
    const totalImages = columns.length;
    
    columns.forEach((column, index) => {
        console.log(`Creating column ${index + 1}`);
        
        // Create main row container
        const rowDiv = document.createElement('div');
        rowDiv.className = 'row column-row mb-4';
        
        // Column Image Section (Left)
        const imageCol = document.createElement('div');
        imageCol.className = 'col-md-6';
        
        const imageContainer = document.createElement('div');
        imageContainer.className = 'column-preview';
        
        const heading = document.createElement('h6');
        heading.textContent = `Column ${index + 1}`;
        imageContainer.appendChild(heading);
        
        const img = document.createElement('img');
        img.className = 'img-fluid';
        
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
        
        img.src = `data:image/jpeg;base64,${column}`;
        imageContainer.appendChild(img);
        
        // Add reprocess button below image
        const controlsDiv = document.createElement('div');
        controlsDiv.className = 'column-controls mt-2';
        
        const reprocessBtn = document.createElement('button');
        reprocessBtn.className = 'reprocess-btn';
        reprocessBtn.innerHTML = '<i class="fas fa-sync-alt"></i> Reprocess';
        reprocessBtn.onclick = () => reprocessColumn(index, column);
        
        controlsDiv.appendChild(reprocessBtn);
        imageContainer.appendChild(controlsDiv);
        
        imageCol.appendChild(imageContainer);
        
        // Response Panel Section (Right)
        const responseCol = document.createElement('div');
        responseCol.className = 'col-md-6';
        
        const responsePanel = document.createElement('div');
        responsePanel.className = 'response-panel';
        responsePanel.id = `response-panel-${index}`;
        
        // Add initial loading state
        responsePanel.innerHTML = `
            <div class="text-center py-4">
                <div class="spinner-border text-primary" role="status">
                    <span class="visually-hidden">Loading...</span>
                </div>
                <p class="text-muted mt-2">Waiting for Gemini response...</p>
            </div>
        `;
        
        responseCol.appendChild(responsePanel);
        
        // Add both columns to the row
        rowDiv.appendChild(imageCol);
        rowDiv.appendChild(responseCol);
        
        // Add row to container
        container.appendChild(rowDiv);
    });
    
    console.log('Display columns setup completed');
}

function formatResponse(response) {
    if (!response) {
        return '<div class="alert alert-warning">No response available</div>';
    }

    // Convert response to array if it's an object
    const responses = Array.isArray(response) ? response : Object.entries(response);

    return `
        <div class="response-item">
            <small class="d-block text-muted mb-2">Gemini Analysis:</small>
            ${responses
                .map(([qNum, answer], index) => {
                    // Handle both array and object responses
                    const questionNumber = typeof qNum === 'number' ? index + 1 : qNum;
                    return `
                        <div class="response-question mb-2">
                            <span class="badge bg-primary me-1">Q${questionNumber}</span>
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
