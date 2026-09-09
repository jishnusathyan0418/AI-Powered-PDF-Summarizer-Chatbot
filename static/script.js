document.addEventListener('DOMContentLoaded', () => {
  const messages = document.getElementById('message-list');
  const input = document.getElementById('message-input');
  const send = document.getElementById('send-button');
  const reset = document.getElementById('reset-button');
  const loading = document.querySelector('.loading-animation');
  let ready = false;
  let busy = false;
  let lightMode = true;

  function setBusy(value) {
    busy = value;
    loading.style.display = value ? 'inline-block' : 'none';
    send.disabled = value || !ready;
    input.disabled = value || !ready;
    reset.disabled = value;
    const upload = document.getElementById('upload-button');
    if (upload) upload.disabled = value || ready;
    document.getElementById('upload-status').textContent = value && !ready ? 'Reading your document...' : ready ? 'Document ready to explore' : 'Waiting for a document';
  }

  function appendMessage(text, user = false) {
    const line = document.createElement('div');
    line.className = 'message-line' + (user ? ' my-text' : '');
    const box = document.createElement('div');
    box.className = 'message-box' + (user ? ' my-text' : '') + (!lightMode ? ' dark' : '') + (!user ? ' assistant-message' : '');
    if (user) {
      box.textContent = text;
    } else {
      box.innerHTML = formatAssistantText(text);
    }
    line.appendChild(box);
    messages.appendChild(line);
    const chat = document.getElementById('chat-window');
    chat.scrollTop = chat.scrollHeight;
    return box;
  }

  function formatAssistantText(text) {
    const escaped = String(text)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#039;');
    return escaped
      .replace(/^#{1,6}\s+/gm, '')
      .replace(/^\s*[-*]\s+/gm, '• ')
      .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
      .replace(/__(.+?)__/g, '<strong>$1</strong>')
      .replace(/`{1,3}/g, '')
      .replace(/\n/g, '<br>');
  }

  async function callApi(path, options) {
    const response = await fetch(path, options);
    let data;
    try {
      data = await response.json();
    } catch {
      throw new Error('The server returned an unreadable response. Please try again.');
    }
    if (!response.ok) throw new Error(data.botResponse || 'The request failed. Please try again.');
    return data.botResponse;
  }

  function showWelcome() {
    appendMessage('Upload a PDF to get started. Once it is ready, ask me for a summary or any detail you want to find.');
    const fileInput = document.getElementById('file-upload');
    const upload = document.getElementById('upload-button');
    upload.onclick = () => fileInput.click();
    if (fileInput.dataset.bound === 'true') {
      setBusy(false);
      return;
    }
    fileInput.dataset.bound = 'true';
    fileInput.addEventListener('change', async () => {
      const file = fileInput.files[0];
      if (!file || busy) return;
      setBusy(true);
      const body = new FormData();
      body.append('file', file);
      try {
        appendMessage(await callApi('/process-document', { method: 'POST', body }));
        ready = true;
      } catch (error) {
        appendMessage(error.message);
        fileInput.value = '';
      } finally {
        setBusy(false);
        if (ready) input.focus();
      }
    });
    setBusy(false);
  }

  async function sendMessage() {
    const message = input.value.trim();
    if (!message || busy || !ready) return;
    appendMessage(message, true);
    input.value = '';
    setBusy(true);
    try {
      appendMessage(await callApi('/process-message', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ userMessage: message }),
      }));
    } catch (error) {
      appendMessage(error.message);
    } finally {
      setBusy(false);
      input.focus();
    }
  }

  send.addEventListener('click', sendMessage);
  input.addEventListener('keydown', (event) => {
    if (event.key === 'Enter') {
      event.preventDefault();
      sendMessage();
    }
  });
  reset.addEventListener('click', async () => {
    if (busy) return;
    setBusy(true);
    try {
      await callApi('/reset', { method: 'POST' });
      ready = false;
      messages.replaceChildren();
      input.value = '';
      showWelcome();
    } catch (error) {
      appendMessage(error.message);
    } finally {
      setBusy(false);
    }
  });
  document.getElementById('light-dark-mode-switch').addEventListener('change', () => {
    lightMode = !lightMode;
    document.body.classList.toggle('dark-mode', !lightMode);
    document.querySelectorAll('.message-box').forEach(box => box.classList.toggle('dark', !lightMode));
  });
  showWelcome();
});
