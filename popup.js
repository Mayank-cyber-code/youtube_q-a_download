const askBtn = document.getElementById('ask-btn');
const videoUrlInput = document.getElementById('video-url');
const questionInput = document.getElementById('question');
const answerDiv = document.getElementById('answer');
const errorDiv = document.getElementById('error');

// Use your deployed Render web service URL here (example Render URL)
const BACKEND_API_URL = "https://youtube-q-a-download.onrender.com/api/ask";

askBtn.addEventListener('click', async () => {
  const video_url = videoUrlInput.value.trim();
  const question = questionInput.value.trim();

  answerDiv.textContent = '';
  errorDiv.textContent = '';

  if (!video_url || !question) {
    errorDiv.textContent = 'Please enter both YouTube video URL and a question.';
    return;
  }

  askBtn.disabled = true;
  askBtn.textContent = 'Asking...';

  try {
    const response = await fetch(BACKEND_API_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ video_url, question }),
    });

    if (!response.ok) {
      const errorData = await response.json();
      errorDiv.textContent = errorData.detail || 'Error: Failed to get answer.';
      return;
    }

    const data = await response.json();

    if (data.answer) {
      answerDiv.textContent = data.answer;
    } else {
      answerDiv.textContent = 'No answer received from the server.';
    }
  } catch (error) {
    errorDiv.textContent = 'Network error or server unreachable.';
  } finally {
    askBtn.disabled = false;
    askBtn.textContent = 'Ask';
  }
});

// Optional: Submit on Enter key press (Shift+Enter for newline)
questionInput.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    askBtn.click();
  }
});
