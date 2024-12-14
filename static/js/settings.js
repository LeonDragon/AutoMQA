// Settings handling
const settings = {
    minWidth: 20,
    minHeight: 4,
    minAspectRatio: 0.7,
    maxAspectRatio: 1.4
};

// Update range input values
document.querySelectorAll('.form-range').forEach(range => {
    const valueSpan = document.getElementById(`${range.id}-value`);
    if (valueSpan) {
        valueSpan.className = 'range-value';
        range.addEventListener('input', function() {
            valueSpan.textContent = this.value;
        });
    }
});

// Initialize settings modal
const settingsModal = new bootstrap.Modal(document.getElementById('settingsModal'));

// Handle settings apply
document.querySelector('#settingsModal .btn-primary').addEventListener('click', function() {
    // Update settings object with current values
    settings.minWidth = parseFloat(document.getElementById('min-width').value);
    settings.minHeight = parseFloat(document.getElementById('min-height').value);
    settings.minAspectRatio = parseFloat(document.getElementById('min-aspect-ratio').value);
    settings.maxAspectRatio = parseFloat(document.getElementById('max-aspect-ratio').value);
    
    settingsModal.hide();
});
