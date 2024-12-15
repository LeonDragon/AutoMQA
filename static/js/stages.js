let currentStage = 1;
const totalStages = 4;

function updateProgress() {
    const progress = (currentStage / totalStages) * 100;
    const progressBar = document.getElementById('progress-bar');
    progressBar.style.width = `${progress}%`;
    
    // Update progress bar text based on current stage
    const stageTexts = {
        1: 'Stage 1: Answer Key Upload',
        2: 'Stage 2: Image Processing',
        3: 'Stage 3: Answer Columns',
        4: 'Stage 4: Gemini Processing'
    };
    progressBar.textContent = stageTexts[currentStage];

    // Update step indicators
    for (let i = 1; i <= totalStages; i++) {
        const step = document.getElementById(`step-${i}`);
        if (i === currentStage) {
            step.classList.add('active');
        } else {
            step.classList.remove('active');
        }
    }
}

function showStage(stageNumber) {
    // Hide all stages
    document.querySelectorAll('.stage').forEach(stage => {
        stage.classList.remove('active');
    });
    
    // Show the selected stage
    document.getElementById(`stage${stageNumber}`).classList.add('active');
    
    // Update current stage and progress
    currentStage = stageNumber;
    updateProgress();
}

function proceedToStage2() {
    showStage(2);
}

function proceedToStage3() {
    console.log('=== Stage 3 Base Transition ===');
    // Show the columns section
    const columnsSection = document.getElementById('columns-section');
    if (columnsSection) {
        columnsSection.style.display = 'block';
        // Scroll to the columns section
        columnsSection.scrollIntoView({ behavior: 'smooth' });
    }
    // Update stage display
    showStage(3);
}

function proceedToStage4() {
    showStage(4);
}

function retryProcessing() {
    // Clear previous results
    document.getElementById('processed-image').src = '';
    document.getElementById('warped-image').src = '';
    document.getElementById('columns-container').innerHTML = '';
    document.getElementById('header-image').src = '';
    
    // Go back to stage 2
    showStage(2);
}

function skipStage1() {
    showStage(2);
    // Add a warning message
    document.getElementById('answer-key-results').innerHTML = 
        '<div class="alert alert-warning">Answer key step was skipped. Scoring functionality will be limited.</div>';
}
