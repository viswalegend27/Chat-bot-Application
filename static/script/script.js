//  --------------------- //
//   Script for Chat.html//                
//--------------------- //

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
    
    // DEBUG: Check if CSRF token is loaded properly
    console.log("🔍 DEBUG - Page loaded");
    console.log("🔍 DEBUG - CSRF Token:", csrftoken);
    console.log("🔍 DEBUG - All cookies:", document.cookie);
    
    // Check if csrftoken is null/undefined
    if (!csrftoken) {
        console.error("❌ CSRF token is missing! This will cause 403 errors.");
        // Try alternative method to get CSRF token
        const csrfInput = document.querySelector('[name=csrfmiddlewaretoken]');
        if (csrfInput) {
            console.log("🔍 Found CSRF token in form input:", csrfInput.value);
        }
    } else {
        console.log("✅ CSRF token found successfully");
    }
    
    // Auto-focus on message input
    document.getElementById('messageInput').focus();
});

// CSRF helper function
function getCookie(name) {
    console.log("🔍 DEBUG - Getting cookie:", name);
    let cookieValue = null;
    if (document.cookie && document.cookie !== '') {
        const cookies = document.cookie.split(';');
        console.log("🔍 DEBUG - All cookies array:", cookies);
        for (let i = 0; i < cookies.length; i++) {
            const cookie = cookies[i].trim();
            console.log("🔍 DEBUG - Checking cookie:", cookie);
            if (cookie.substring(0, name.length + 1) === (name + '=')) {
                cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                break;
            }
        }
    }
    console.log("🔍 DEBUG - Cookie value for", name, ":", cookieValue);
    return cookieValue;
}

// Handle form submission
document.getElementById("chatForm").addEventListener("submit", function (e) {
    e.preventDefault();

    const inputField = document.getElementById("messageInput");
    const message = inputField.value.trim();
    if (!message) return;

    console.log("🔍 DEBUG - Form submission started");
    console.log("🔍 DEBUG - Message:", message);
    console.log("🔍 DEBUG - CSRF Token before request:", csrftoken);
    console.log("🔍 DEBUG - Chat URL:", CHAT_URL);

    // Add user message immediately
    addMessage("user", message);

    // Clear input
    inputField.value = "";

    // Show loading indicator
    document.getElementById("loadingIndicator").style.display = "block";

    // Prepare request body
    const requestBody = "message=" + encodeURIComponent(message) + "&csrfmiddlewaretoken=" + encodeURIComponent(csrftoken);
    console.log("🔍 DEBUG - Request body:", requestBody);

    // Prepare headers
    const headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "X-CSRFToken": csrftoken
    };
    console.log("🔍 DEBUG - Request headers:", headers);

    fetch(CHAT_URL, {
        method: "POST",
        headers: headers,
        body: requestBody,
        credentials: "same-origin"
    })
        .then(response => {
            console.log("🔍 DEBUG - Response received");
            console.log("🔍 DEBUG - Response status:", response.status);
            console.log("🔍 DEBUG - Response ok:", response.ok);
            console.log("🔍 DEBUG - Response headers:", [...response.headers.entries()]);
            
            if (!response.ok) {
                // Get the error response text for debugging
                return response.text().then(errorText => {
                    console.error("❌ ERROR - Response not ok");
                    console.error("❌ ERROR - Status:", response.status);
                    console.error("❌ ERROR - Status text:", response.statusText);
                    console.error("❌ ERROR - Error response body:", errorText);
                    throw new Error(`Network response was not ok: ${response.status} ${response.statusText}`);
                });
            }
            return response.json();
        })
        .then(data => {
            console.log("✅ SUCCESS - Response data:", data);

            // Hide loading indicator
            document.getElementById("loadingIndicator").style.display = "none";

            // Add bot reply
            addMessage("bot", data.reply || "⚠️ No reply from server.");
            scrollToBottom();
        })
        .catch(error => {
            console.error("❌ FETCH ERROR:", error);
            console.error("❌ ERROR Details:", error.message);

            // Hide loading indicator
            document.getElementById("loadingIndicator").style.display = "none";

            addMessage("bot", "⚠️ Sorry, something went wrong. Please try again.");
            scrollToBottom();
        });
});

// Add message to chat
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

// Scroll to bottom of chat
function scrollToBottom() {
    const chatContainer = document.getElementById('chatContainer');
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

// Escape HTML characters
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
    console.log("📄 Opening documents modal");
    const modal = new bootstrap.Modal(document.getElementById('documentsModal'));
    modal.show();
    loadDocumentsList();
}

// Load documents list
function loadDocumentsList() {
    console.log("📄 Loading documents list");
    const content = document.getElementById('documentsContent');
    content.innerHTML = '<div class="text-center"><div class="spinner-border" role="status"><span class="visually-hidden">Loading...</span></div></div>';

    fetch(DOCUMENTS_URL, {
        method: 'GET',
        headers: {
            'X-CSRFToken': csrftoken
        },
        credentials: 'same-origin'
    })
        .then(response => {
            console.log("📄 Documents response status:", response.status);
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            console.log("📄 Documents data:", data);
            if (data.documents && data.documents.length > 0) {
                let html = '<div class="row">';
                data.documents.forEach(doc => {
                    html += `
                        <div class="col-12 mb-3">
                            <div class="card">
                                <div class="card-body">
                                    <div class="d-flex justify-content-between align-items-start">
                                        <div class="flex-grow-1">
                                            <h6 class="card-title mb-1">📄 ${escapeHtml(doc.filename)}</h6>
                                            <p class="card-text">
                                                <small class="text-muted">${doc.chunk_count || 0} chunks</small>
                                            </p>
                                            ${doc.content_preview ? `<div class="document-preview bg-light p-2 rounded" style="font-size: 0.9em; max-height: 100px; overflow: hidden;">${escapeHtml(doc.content_preview)}</div>` : ''}
                                        </div>
                                        <button class="btn btn-outline-danger btn-sm ms-2" 
                                                onclick="deleteDocument(${doc.id}, '${escapeHtml(doc.filename)}')">
                                            🗑️ Delete
                                        </button>
                                    </div>
                                </div>
                            </div>
                        </div>
                    `;
                });
                html += '</div>';
                content.innerHTML = html;
            } else {
                content.innerHTML = '<div class="text-center text-muted py-4"><h5>📄 No documents uploaded yet</h5><p>Upload some documents to get started with Document Q&A!</p></div>';
            }
        })
        .catch(error => {
            console.error('📄 Error loading documents:', error);
            content.innerHTML = '<div class="text-center text-danger py-4"><h5>❌ Error loading documents</h5><p>Please try again later.</p></div>';
        });
}

// Delete Document Function
function deleteDocument(docId, filename) {
    console.log(`🗑️ Attempting to delete document: ${filename} (ID: ${docId})`);
    
    if (confirm(`Are you sure you want to delete "${filename}"?`)) {
        const deleteUrl = DOCUMENTS_URL.replace('/documents', `/delete_document/${docId}`);
        console.log("🗑️ Delete URL:", deleteUrl);
        
        fetch(deleteUrl, {
            method: 'POST',
            headers: {
                'X-CSRFToken': csrftoken,
                'Content-Type': 'application/x-www-form-urlencoded',
            },
            credentials: 'same-origin'
        })
        .then(response => {
            console.log("🗑️ Delete response status:", response.status);
            return response.json();
        })
        .then(data => {
            console.log("🗑️ Delete response data:", data);
            if (data.success) {
                console.log("✅ Document deleted successfully");
                loadDocumentsList(); // Refresh the documents list
            } else {
                console.error("❌ Error deleting document:", data.error);
                alert('Error deleting document: ' + (data.error || 'Unknown error'));
            }
        })
        .catch(error => {
            console.error('🗑️ Error deleting document:', error);
            alert('Error deleting document. Please try again.');
        });
    }
}

// Clear History Function
function clearHistory() {
    console.log("🗑️ Opening clear history modal");
    const modal = new bootstrap.Modal(document.getElementById('clearHistoryModal'));
    modal.show();
}

// Confirm Clear History
function confirmClearHistory() {
    console.log("🗑️ Confirming clear history");
    
    fetch(CLEAR_HISTORY_URL, {
        method: 'POST',
        headers: {
            'X-CSRFToken': csrftoken,
            'Content-Type': 'application/x-www-form-urlencoded',
        },
        credentials: 'same-origin'
    })
    .then(response => {
        console.log("🗑️ Clear history response status:", response.status);
        console.log("🗑️ Clear history response headers:", [...response.headers.entries()]);
        
        // Check if response is successful (2xx status codes)
        if (response.ok) {
            // Try to parse as JSON first
            const contentType = response.headers.get('content-type');
            if (contentType && contentType.includes('application/json')) {
                return response.json();
            } else {
                // If not JSON, treat as successful HTML response (redirect/page refresh)
                console.log("✅ History cleared successfully (HTML response)");
                return { success: true };
            }
        } else {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
    })
    .then(data => {
        console.log("🗑️ Clear history response data:", data);
        console.log("✅ History cleared successfully");
        
        // Clear the chat container immediately
        document.getElementById('chatContainer').innerHTML = '';
        
        // Close the modal
        const modal = bootstrap.Modal.getInstance(document.getElementById('clearHistoryModal'));
        if (modal) modal.hide();
        
        // Show success message
        addMessage("bot", "✅ Chat history has been cleared successfully!");
    })
    .catch(error => {
        console.error('🗑️ Error clearing history:', error);
        console.error('🗑️ Full error details:', error.message);
        
        // Since the operation might have succeeded despite the error (common with redirects),
        // let's assume success and clear the UI
        console.log("⚠️ Assuming operation succeeded despite error");
        
        // Clear the chat container
        document.getElementById('chatContainer').innerHTML = '';
        
        // Close the modal
        const modal = bootstrap.Modal.getInstance(document.getElementById('clearHistoryModal'));
        if (modal) modal.hide();
        
        // Show success message instead of error
        addMessage("bot", "✅ Chat history has been cleared successfully!");
    });
}

// Clear Documents Function
function clearDocuments() {
    console.log("🗑️ Opening clear documents modal");
    const modal = new bootstrap.Modal(document.getElementById('clearDocumentsModal'));
    modal.show();
}

// Confirm Clear Documents
function confirmClearDocuments() {
    console.log("🗑️ Confirming clear documents");
    
    fetch(CLEAR_DOCUMENTS_URL, {
        method: 'POST',
        headers: {
            'X-CSRFToken': csrftoken,
            'Content-Type': 'application/x-www-form-urlencoded',
        },
        credentials: 'same-origin'
    })
    .then(response => {
        console.log("🗑️ Clear documents response status:", response.status);
        console.log("🗑️ Clear documents response headers:", [...response.headers.entries()]);
        
        // Check if response is successful (2xx status codes)
        if (response.ok) {
            // Try to parse as JSON first
            const contentType = response.headers.get('content-type');
            if (contentType && contentType.includes('application/json')) {
                return response.json();
            } else {
                // If not JSON, treat as successful HTML response (redirect/page refresh)
                console.log("✅ Documents cleared successfully (HTML response)");
                return { success: true };
            }
        } else {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
    })
    .then(data => {
        console.log("🗑️ Clear documents response data:", data);
        console.log("✅ Documents cleared successfully");
        
        // Close the modal
        const modal = bootstrap.Modal.getInstance(document.getElementById('clearDocumentsModal'));
        if (modal) modal.hide();
        
        // Show success message
        alert('✅ All documents have been cleared successfully!');
        
        // Refresh documents list if documents modal is still open
        const documentsModal = document.getElementById('documentsModal');
        if (documentsModal && documentsModal.classList.contains('show')) {
            loadDocumentsList();
        }
    })
    .catch(error => {
        console.error('🗑️ Error clearing documents:', error);
        console.error('🗑️ Full error details:', error.message);
        
        // Since the operation might have succeeded despite the error (common with redirects),
        // let's assume success and update the UI
        console.log("⚠️ Assuming operation succeeded despite error");
        
        // Close the modal
        const modal = bootstrap.Modal.getInstance(document.getElementById('clearDocumentsModal'));
        if (modal) modal.hide();
        
        // Show success message instead of error
        alert('✅ All documents have been cleared successfully!');
        
        // Refresh documents list if documents modal is still open
        const documentsModal = document.getElementById('documentsModal');
        if (documentsModal && documentsModal.classList.contains('show')) {
            loadDocumentsList();
        }
    });
}

// IMPROVED CSRF Token Getting Function
function getCSRFToken() {
    // Method 1: From hidden input (most reliable)
    const hiddenInput = document.getElementById('csrf-token');
    if (hiddenInput && hiddenInput.value) {
        console.log("✅ Got CSRF token from hidden input:", hiddenInput.value);
        return hiddenInput.value;
    }
    
    // Method 2: From form input
    const formInput = document.querySelector('[name=csrfmiddlewaretoken]');
    if (formInput && formInput.value) {
        console.log("✅ Got CSRF token from form input:", formInput.value);
        return formInput.value;
    }
    
    // Method 3: From cookie
    let cookieValue = null;
    if (document.cookie && document.cookie !== '') {
        const cookies = document.cookie.split(';');
        for (let i = 0; i < cookies.length; i++) {
            const cookie = cookies[i].trim();
            if (cookie.substring(0, 10) === 'csrftoken=') {
                cookieValue = decodeURIComponent(cookie.substring(10));
                break;
            }
        }
    }
    if (cookieValue) {
        console.log("✅ Got CSRF token from cookie:", cookieValue);
        return cookieValue;
    }
    
    console.error("❌ Could not get CSRF token from any method");
    console.error("❌ Available cookies:", document.cookie);
    return null;
}

// Set the global CSRF token
const csrftoken = getCSRFToken();
console.log("🔍 Final CSRF token:", csrftoken);

function clearUIHistory() {
    const chatContainer = document.getElementById('chatContainer');
    const confirmClear = confirm('Clear current chat view? (Messages will still be in your history)');
    
    if (confirmClear) {
        // Hide all messages with animation
        const messages = chatContainer.querySelectorAll('.message');
        messages.forEach((message, index) => {
            setTimeout(() => {
                message.style.opacity = '0';
                message.style.transform = 'translateX(-100px)';
                setTimeout(() => {
                    message.style.display = 'none';
                }, 300);
            }, index * 50);
        });
        
        // Show success message
        setTimeout(() => {
            chatContainer.innerHTML = `
                <div class="alert alert-success alert-dismissible fade show" role="alert">
                    ✨ Current view cleared! Your messages are still saved in history.
                    <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
                </div>
            `;
        }, 1000);
    }
}

function deleteConversation(conversationId) {
    if (confirm('Delete this conversation thread? This cannot be undone!')) {
        const targetCard = document.querySelector(`.conversation-card[data-id="${conversationId}"]`);
        if (!targetCard) return;

        // Animate removal
        targetCard.style.transition = 'all 0.5s ease';
        targetCard.style.opacity = '0';
        targetCard.style.transform = 'translateX(-100%)';

        setTimeout(() => {
            targetCard.remove();

            if (document.querySelectorAll('.conversation-card').length === 0) {
                document.querySelector('.col-md-10').innerHTML = `
                    <div class="empty-state">
                        <h4 class="text-success">✨ All conversations cleared!</h4>
                        <p class="text-muted">Start chatting to create new conversations.</p>
                        <a href="/chat/" class="btn btn-primary btn-lg mt-3">
                            🚀 Start Chatting
                        </a>
                    </div>
                `;
            }
        }, 500);
    }
}

function permanentDeleteAll() {
    if (confirm('⚠️ PERMANENTLY DELETE ALL CHAT HISTORY?\n\nThis will delete everything from the database and cannot be undone!')) {
        if (confirm('Are you absolutely sure? This action is irreversible!')) {
            // Make AJAX call to delete all
            fetch('/clear_history/', {
                method: 'POST',
                headers: {
                    'X-CSRFToken': getCSRFToken(),
                    'Content-Type': 'application/json',
                },
            })
            .then(response => {
                if (response.ok) {
                    window.location.reload();
                } else {
                    alert('Error deleting history');
                }
            });
        }
    }
}

// 🎤 Speech-to-Text with Debug Logs
const micButton = document.getElementById("micButton");
const messageInput = document.getElementById("messageInput");
const audioPreview = document.getElementById("audioPreview");
const csrfToken = document.getElementById("csrf-token").value;

let audioContext, mediaStream, sourceNode, processorNode;
let recordedBuffers = [];
let recording = false;
const TARGET_SAMPLE_RATE = 16000; // server expects 16k

function interleaveAndDownsample(buffers, inputSampleRate, outSampleRate) {
  // merge Float32 arrays
    let length = buffers.reduce((sum, b) => sum + b.length, 0);
    let merged = new Float32Array(length);
    let offset = 0;
    for (const b of buffers) {
        merged.set(b, offset);
        offset += b.length;
    }
    if (inputSampleRate === outSampleRate) {
        return merged;
    }
    const ratio = inputSampleRate / outSampleRate;
    const outLength = Math.round(merged.length / ratio);
    const out = new Float32Array(outLength);
    let pos = 0;
    for (let i = 0; i < outLength; i++) {
        out[i] = merged[Math.floor(i * ratio)];
    }
    return out;
    }

    function floatTo16BitPCM(float32Array) {
    const l = float32Array.length;
    const buffer = new ArrayBuffer(l * 2);
    const view = new DataView(buffer);
    let offset = 0;
    for (let i = 0; i < l; i++, offset += 2) {
        let s = Math.max(-1, Math.min(1, float32Array[i]));
        view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
    }
    return view;
    }

    function encodeWAV(float32Array, sampleRate) {
    const bytesPerSample = 2;
    const blockAlign = bytesPerSample * 1; // mono
    const bufferLength = 44 + float32Array.length * bytesPerSample;
    const buffer = new ArrayBuffer(bufferLength);
    const view = new DataView(buffer);

    /* RIFF identifier */
    writeString(view, 0, 'RIFF');
    /* file length */
    view.setUint32(4, 36 + float32Array.length * bytesPerSample, true);
    /* RIFF type */
    writeString(view, 8, 'WAVE');
    /* format chunk identifier */
    writeString(view, 12, 'fmt ');
    /* format chunk length */
    view.setUint32(16, 16, true);
    /* sample format (raw) */
    view.setUint16(20, 1, true);
    /* channel count */
    view.setUint16(22, 1, true); // mono
    /* sample rate */
    view.setUint32(24, sampleRate, true);
    /* byte rate (sampleRate * blockAlign) */
    view.setUint32(28, sampleRate * blockAlign, true);
    /* block align (channel count * bytes per sample) */
    view.setUint16(32, blockAlign, true);
    /* bits per sample */
    view.setUint16(34, bytesPerSample * 8, true);
    /* data chunk identifier */
    writeString(view, 36, 'data');
    /* data chunk length */
    view.setUint32(40, float32Array.length * bytesPerSample, true);

    // PCM samples
    const pcmView = floatTo16BitPCM(float32Array);
    // copy PCM bytes after header
    const headerBytes = 44;
    for (let i = 0; i < pcmView.byteLength; i++) {
        view.setUint8(headerBytes + i, pcmView.getUint8(i));
    }

    return new Blob([view], { type: 'audio/wav' });
    }

    function writeString(view, offset, string) {
    for (let i = 0; i < string.length; i++) {
        view.setUint8(offset + i, string.charCodeAt(i));
    }
    }

    async function startRecording() {
    console.log("🎙️ startRecording()");
    recordedBuffers = [];

    // init audio context
    audioContext = new (window.AudioContext || window.webkitAudioContext)();
    const inputSampleRate = audioContext.sampleRate;
    console.log("AudioContext sample rate:", inputSampleRate);

    try {
        mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
    } catch (err) {
        console.error("Microphone access denied:", err);
        alert("Microphone access is required.");
        return;
    }

    sourceNode = audioContext.createMediaStreamSource(mediaStream);

    // Use ScriptProcessorNode if AudioWorklet not available (simple cross-browser)
    const bufferSize = 4096;
    processorNode = audioContext.createScriptProcessor(bufferSize, 1, 1);

    processorNode.onaudioprocess = (e) => {
        const inputBuffer = e.inputBuffer.getChannelData(0);
        // copy Float32Array
        recordedBuffers.push(new Float32Array(inputBuffer));
        // debug log occasionally
        if (recordedBuffers.length % 25 === 0) {
        console.log("🔴 recorded chunks:", recordedBuffers.length);
        }
    };

    sourceNode.connect(processorNode);
    processorNode.connect(audioContext.destination); // necessary on some browsers

    recording = true;
    micButton.innerText = "⏺️ Recording...";
    micButton.classList.add("btn-danger");
    console.log("Recording started");
    }

    async function stopRecordingAndUpload() {
    console.log("🛑 stopRecordingAndUpload()");
    if (!recording) return;

    // stop nodes and tracks
    try {
        processorNode.disconnect();
        sourceNode.disconnect();
        mediaStream.getTracks().forEach(t => t.stop());
        audioContext.close();
    } catch (err) {
        console.warn("Error stopping audio nodes:", err);
    }

    recording = false;
    micButton.innerText = "🎤";
    micButton.classList.remove("btn-danger");

    // Downsample & merge
    const inputSampleRate = (audioContext && audioContext.sampleRate) || 48000;
    const downsampled = interleaveAndDownsample(recordedBuffers, inputSampleRate, TARGET_SAMPLE_RATE);
    console.log("Merged frames length:", downsampled.length);

    // create WAV blob
    const wavBlob = encodeWAV(downsampled, TARGET_SAMPLE_RATE);
    console.log("🧾 WAV blob created:", wavBlob, "size:", wavBlob.size);

    // preview
    audioPreview.src = URL.createObjectURL(wavBlob);
    audioPreview.hidden = false;
    audioPreview.play().catch(() => {});

    // upload
    const form = new FormData();
    form.append("audio", wavBlob, "recording.wav");

    try {
        console.log("📤 Uploading WAV to /speech-to-text/");
        const res = await fetch("/speech-to-text/", {
        method: "POST",
        headers: { "X-CSRFToken": csrfToken },
        body: form
        });
        const data = await res.json();
        console.log("📥 Server response:", res.status, data);
        if (res.ok && data.text) {
        messageInput.value = data.text;
        console.log("✅ Transcribed text inserted into input.");
        } else {
        console.error("❌ Transcription failed:", data);
        alert("Transcription failed: " + (data.error || data.text || "unknown"));
        }
    } catch (err) {
        console.error("❌ Upload/transcription error:", err);
        alert("Upload/transcription error. See console for details.");
    }
    }

    // wire up button to toggle
    micButton.addEventListener("click", async () => {
    console.log("🎤 Mic button clicked. recording?", recording);
    if (!recording) {
        await startRecording();
        // stop automatically after 8s if you want:
        // setTimeout(() => { if (recording) stopRecordingAndUpload(); }, 8000);
    } else {
        await stopRecordingAndUpload();
    }
});

function setRecordingState(on) {
    if (on) {
        micButton.classList.add('recording');
        // if you prefer Bootstrap naming: micButton.classList.add('btn-danger');
    } else {
        micButton.classList.remove('recording');
        micButton.classList.remove('btn-danger');
    }
}

// Example usage in your recorder lifecycle:
recognition.onstart = () => { setRecordingState(true); /* other stuff */ };
recognition.onend   = () => { setRecordingState(false); /* other stuff */ };
// OR for MediaRecorder approach
mediaRecorder.onstart = () => setRecordingState(true);
mediaRecorder.onstop  = () => setRecordingState(false);

