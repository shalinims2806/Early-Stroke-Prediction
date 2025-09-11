console.log('Script.js loaded successfully');

document.addEventListener('DOMContentLoaded', function() {
    console.log('DOM fully loaded');
    
    // Just a simple test
    document.querySelectorAll('button').forEach(function(button) {
        console.log('Found button:', button.textContent);
    });
    
    // Test the form
    const form = document.getElementById('prediction-form');
    if (form) {
        console.log('Form found');
        form.addEventListener('submit', function(e) {
            console.log('Form submitted');
            e.preventDefault();
            alert('Form submit intercepted by JavaScript');
        });
    } else {
        console.error('Form not found');
    }
    
    // Test simple predict button
    const testButton = document.getElementById('test-predict-button');
    if (testButton) {
        console.log('Test button found');
        testButton.addEventListener('click', function() {
            console.log('Test button clicked');
            alert('Test button clicked');
        });
    } else {
        console.error('Test button not found');
    }
}); 