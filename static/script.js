document.addEventListener('DOMContentLoaded', () => {
    const uploadInput = document.getElementById('document-upload');
    const uploadButton = document.getElementById('upload-button');
    const uploadStatus = document.getElementById('upload-status');
    const documentSelector = document.getElementById('document-selector');
    const documentListStatus = document.getElementById('document-list-status');
    const chatInput = document.getElementById('chat-input');
    const sendButton = document.getElementById('send-button');
    const chatMessages = document.getElementById('chat-messages');
    const chatStatus = document.getElementById('chat-status');

    // Base URL for your FastAPI backend
    const API_BASE_URL = 'http://localhost:8000/api'; // Change this for deployment!

    let documentProcessingIntervals = {}; // To store intervals for status checks

    // --- Utility Functions ---
    function displayStatus(element, message, isError = false) {
        element.textContent = message;
        element.style.color = isError ? 'red' : 'green';
        if (isError) {
            element.style.backgroundColor = '#f8d7da';
            element.style.borderColor = '#f5c6cb';
            element.style.color = '#721c24';
        } else {
            element.style.backgroundColor = '#d4edda';
            element.style.borderColor = '#c3e6cb';
            element.style.color = '#155724';
        }
    }

    function addMessageToChat(message, sender) {
        const messageElement = document.createElement('div');
        messageElement.classList.add('chat-message');
        messageElement.classList.add(sender + '-message');
        messageElement.innerHTML = message; // Use innerHTML to allow for bold/markdown from backend
        chatMessages.appendChild(messageElement);
        chatMessages.scrollTop = chatMessages.scrollHeight; // Scroll to the bottom
    }

    // --- Document Management ---

    async function listDocuments() {
        try {
            documentListStatus.textContent = 'Loading documents...';
            const response = await fetch(`${API_BASE_URL}/documents`);
            const data = await response.json();

            documentSelector.innerHTML = ''; // Clear previous options
            if (data.documents && data.documents.length > 0) {
                documentListStatus.textContent = ''; // Clear status if documents found
                data.documents.forEach(doc => {
                    const option = document.createElement('option');
                    option.value = doc.id;
                    option.textContent = `${doc.filename} (${doc.status})`;
                    if (doc.status === 'processing') {
                        option.disabled = true; // Disable if still processing
                        // Start/restart status check for this document
                        startDocumentStatusCheck(doc.id);
                    } else if (doc.status === 'ready') {
                        option.disabled = false;
                        if (documentProcessingIntervals[doc.id]) {
                            clearInterval(documentProcessingIntervals[doc.id]); // Stop checking
                            delete documentProcessingIntervals[doc.id];
                        }
                    } else { // Failed
                        option.disabled = true;
                        option.textContent += ' - Failed';
                        if (documentProcessingIntervals[doc.id]) {
                            clearInterval(documentProcessingIntervals[doc.id]); // Stop checking
                            delete documentProcessingIntervals[doc.id];
                        }
                    }
                    documentSelector.appendChild(option);
                });
            } else {
                displayStatus(documentListStatus, 'No documents uploaded yet.');
            }
        } catch (error) {
            console.error('Error listing documents:', error);
            displayStatus(documentListStatus, 'Failed to load documents.', true);
        }
    }

    async function checkDocumentStatus(documentId) {
        try {
            const response = await fetch(`${API_BASE_URL}/documents/${documentId}/status`);
            const data = await response.json();
            const option = documentSelector.querySelector(`option[value="${documentId}"]`);

            if (option) {
                option.textContent = `${data.filename} (${data.status})`;
                if (data.status === 'ready') {
                    option.disabled = false;
                    clearInterval(documentProcessingIntervals[documentId]);
                    delete documentProcessingIntervals[documentId];
                    displayStatus(uploadStatus, `Document "${data.filename}" is ready! You can now select it for chat.`, false);
                    listDocuments(); // Refresh list to update all statuses
                } else if (data.status === 'failed') {
                    option.disabled = true;
                    option.textContent += ' - Failed';
                    clearInterval(documentProcessingIntervals[documentId]);
                    delete documentProcessingIntervals[documentId];
                    displayStatus(uploadStatus, `Document "${data.filename}" failed processing.`, true);
                } else {
                    // Still processing, update text if needed, keep interval
                    option.disabled = true;
                }
            }
        } catch (error) {
            console.error(`Error checking status for ${documentId}:`, error);
            clearInterval(documentProcessingIntervals[documentId]);
            delete documentProcessingIntervals[documentId];
            displayStatus(uploadStatus, `Error checking status for a document.`, true);
        }
    }

    function startDocumentStatusCheck(documentId) {
        if (!documentProcessingIntervals[documentId]) {
            // Check status every 5 seconds
            documentProcessingIntervals[documentId] = setInterval(() => {
                checkDocumentStatus(documentId);
            }, 5000);
        }
    }

    // --- Event Listeners ---

    uploadButton.addEventListener('click', async () => {
        const file = uploadInput.files[0];
        if (!file) {
            displayStatus(uploadStatus, 'Please select a file to upload.', true);
            return;
        }

        const formData = new FormData();
        formData.append('file', file);

        displayStatus(uploadStatus, `Uploading "${file.name}"...`);
        uploadButton.disabled = true;

        try {
            const response = await fetch(`${API_BASE_URL}/documents/upload`, {
                method: 'POST',
                body: formData,
            });

            const data = await response.json();
            if (response.ok) {
                displayStatus(uploadStatus, `"${data.filename}" uploaded. Processing started... (Status: ${data.status})`);
                uploadInput.value = ''; // Clear file input
                listDocuments(); // Refresh document list to show new document and its status
                startDocumentStatusCheck(data.document_id); // Start polling for status
            } else {
                displayStatus(uploadStatus, `Upload failed: ${data.detail || 'Unknown error'}`, true);
            }
        } catch (error) {
            console.error('Error during upload:', error);
            displayStatus(uploadStatus, `Network error during upload: ${error.message}`, true);
        } finally {
            uploadButton.disabled = false;
        }
    });

    sendButton.addEventListener('click', async () => {
        const query = chatInput.value.trim();
        if (!query) {
            displayStatus(chatStatus, 'Please enter a question.', true);
            return;
        }

        // Get all selected options from the (potentially multiple-select) dropdown
        const selectedDocumentIds = Array.from(documentSelector.options)
            .filter(option => option.selected && option.value !== "") // Filter for selected and non-empty value
            .map(option => option.value); // Get the ID from the value

        if (selectedDocumentIds.length === 0) {
            displayStatus(chatStatus, 'Please select at least one document to chat with.', true);
            return;
        }

        addMessageToChat(query, 'user');
        chatInput.value = '';
        sendButton.disabled = true;
        displayStatus(chatStatus, 'Generating response...');

        try {
            const response = await fetch(`${API_BASE_URL}/chat`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    query: query,
                    document_ids: selectedDocumentIds // Send the array of IDs
                }),
            });

            const data = await response.json();
            if (response.ok) {
                addMessageToChat(data.response, 'bot');
                displayStatus(chatStatus, 'Response generated.');
            } else {
                addMessageToChat(`Error: ${data.detail || 'Unknown error'}`, 'bot');
                displayStatus(chatStatus, `Chat failed: ${data.detail || 'Unknown error'}`, true);
            }
        } catch (error) {
            console.error('Error during chat:', error);
            addMessageToChat(`Network error: ${error.message}`, 'bot');
            displayStatus(chatStatus, `Network error during chat: ${error.message}`, true);
        } finally {
            sendButton.disabled = false;
        }
    });

    // Initial load of documents when the page loads
    listDocuments();
});
