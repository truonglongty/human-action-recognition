// This file can be used for any global JavaScript functions
// that are needed across multiple pages and are not specific
// enough to be included directly in a template's script block.

// For example, a utility function (currently commented out):
// function showGlobalMessage(message, type = 'info') {
//     // This would require a dedicated container in your base.html, e.g., <div id="global-alert-container"></div>
//     const alertContainer = document.getElementById('global-alert-container');
//     if (alertContainer) {
//         const alertDiv = document.createElement('div');
//         alertDiv.className = `alert alert-${type} alert-dismissible fade show m-3`; // Added some margin
//         alertDiv.role = 'alert';
//         alertDiv.innerHTML = `
//             ${message}
//             <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
//         `;
//         // Prepend to show at the top, or append to show at the bottom
//         alertContainer.prepend(alertDiv);
//
//         // Optional: Auto-dismiss after some time
//         // setTimeout(() => {
//         //     bootstrap.Alert.getOrCreateInstance(alertDiv).close();
//         // }, 5000); // Dismiss after 5 seconds
//     }
// }

// For this project, most of the JavaScript logic has been moved
// to the respective HTML templates (index.html, webcam.html)
// within {% block scripts_extra %}
// to keep it co-located with the elements it manipulates.
// This makes page-specific logic easier to manage.

console.log("Global scripts.js loaded. Page-specific scripts are in their respective HTML templates.");

// You could add other global initializations here if needed, for example:
// document.addEventListener('DOMContentLoaded', function() {
//     // Initialize all Bootstrap tooltips if you use them
//     var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'))
//     var tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
//         return new bootstrap.Tooltip(tooltipTriggerEl)
//     })
// });
