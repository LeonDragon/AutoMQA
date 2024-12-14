// Initialize tooltips
document.addEventListener('DOMContentLoaded', function() {
    // Initialize all tooltips
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });

    // Initialize the progress bar
    updateProgress();

    // Set up file input change handlers
    setupFileInputs();
});

function setupFileInputs() {
    // Add file input change handlers for both answer key and student sheet
    const fileInputs = document.querySelectorAll('input[type="file"]');
    fileInputs.forEach(input => {
        input.addEventListener('change', function() {
            const label = this.nextElementSibling;
            if (label && this.files.length > 0) {
                label.textContent = this.files[0].name;
            } else if (label) {
                label.textContent = 'Choose file';
            }
        });
    });
}
