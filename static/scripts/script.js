        // Function to render markdown content
        function renderMarkdown(content) {
            marked.setOptions({
                breaks: true,
                gfm: true
            });
            return marked.parse(content);
        }
        
        // Render existing bot messages
        document.addEventListener('DOMContentLoaded', function() {
            const botResponses = document.querySelectorAll('.bot-response');
            botResponses.forEach(function(element) {
                const content = element.getAttribute('data-content');
                element.innerHTML = renderMarkdown(content);
            });
            scrollToBottom();
        });
        
        // Handle form submission
        document.getElementById('chatForm').addEventListener('submit', function(e) {
            e.preventDefault();
            
            const messageInput = document.getElementById('messageInput');
            const message = messageInput.value.trim();
            
            if (!message) return;
            
            addMessage('user', message);
            messageInput.value = '';
            
            document.getElementById('loadingIndicator').style.display = 'block';
            scrollToBottom();
            
            fetch('{{ url_for("chat") }}', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/x-www-form-urlencoded',
                },
                body: 'message=' + encodeURIComponent(message)
            })
            .then(response => response.json())
            .then(data => {
                document.getElementById('loadingIndicator').style.display = 'none';
                
                if (data.reply) {
                    addMessage('bot', data.reply);
                } else if (data.error) {
                    addMessage('bot', '⚠️ ' + data.error);
                }
                scrollToBottom();
            })
            .catch(error => {
                document.getElementById('loadingIndicator').style.display = 'none';
                console.error('Error:', error);
                addMessage('bot', '⚠️ Sorry, something went wrong. Please try again.');
                scrollToBottom();
            });
        });
        
        function addMessage(sender, text) {
            const chatContainer = document.getElementById('chatContainer');
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${sender}-message`;
            
            if (sender === 'user') {
                messageDiv.innerHTML = `<strong>You:</strong> ${escapeHtml(text)}`;
            } else {
                messageDiv.innerHTML = `<strong>AI:</strong> <div class="bot-response">${renderMarkdown(text)}</div>`;
            }
            
            chatContainer.appendChild(messageDiv);
        }
        
        function scrollToBottom() {
            const chatContainer = document.getElementById('chatContainer');
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }
        
        function escapeHtml(unsafe) {
            return unsafe
                .replace(/&/g, "&amp;")
                .replace(/</g, "&lt;")
                .replace(/>/g, "&gt;")
                .replace(/"/g, "&quot;")
                .replace(/'/g, "&#039;");
        }
        
        // View Documents Function
        function viewDocuments() {
            const modal = new bootstrap.Modal(document.getElementById('documentsModal'));
            modal.show();
            loadDocumentsList();
        }
        
        // Separate function to load documents list without creating new modal
        function loadDocumentsList() {
            const content = document.getElementById('documentsContent');
            content.innerHTML = '<div class="text-center"><div class="spinner-border" role="status"><span class="visually-hidden">Loading...</span></div></div>';
            
            fetch('{{ url_for("documents") }}')
                .then(response => response.json())
                .then(data => {
                    if (data.documents && data.documents.length > 0) {
                        let html = '';
                        data.documents.forEach(doc => {
                            html += `
                                <div class="document-item">
                                    <div class="d-flex justify-content-between align-items-start">
                                        <div class="flex-grow-1">
                                            <h6 class="mb-1">📄 ${doc.filename}</h6>
                                            <small class="text-muted">${doc.chunk_count} chunks</small>
                                            <div class="document-preview">${doc.content_preview}</div>
                                        </div>
                                        <button class="btn btn-outline-danger btn-sm ms-2" 
                                                onclick="deleteDocument(${doc.id}, '${doc.filename}')">
                                            🗑️ Delete
                                        </button>
                                    </div>
                                </div>
                            `;
                        });
                        content.innerHTML = html;
                    } else {
                        content.innerHTML = '<div class="text-center text-muted">No documents uploaded yet.</div>';
                    }
                })
                .catch(error => {
                    console.error('Error:', error);
                    content.innerHTML = '<div class="text-center text-danger">Error loading documents.</div>';
                });
        }
        
        // Delete Document Function
        function deleteDocument(docId, filename) {
            if (confirm(`Are you sure you want to delete "${filename}"?`)) {
                fetch(`/delete_document/${docId}`, {
                    method: 'POST'
                })
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        loadDocumentsList(); // Refresh only the content, not the modal
                    } else {
                        alert('Error deleting document: ' + (data.error || 'Unknown error'));
                    }
                })
                .catch(error => {
                    console.error('Error:', error);
                    alert('Error deleting document');
                });
            }
        }
        
        // Clear History Function
        function clearHistory() {
            const modal = new bootstrap.Modal(document.getElementById('clearHistoryModal'));
            modal.show();
        }
        
        function clearDocuments() {
        const modal = new bootstrap.Modal(document.getElementById('clearDocumentsModal'));
        modal.show();
        }
        
        // Auto-focus on message input
        document.getElementById('messageInput').focus();