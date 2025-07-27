const askBtn = document.getElementById('ask-btn');
const videoUrlInput = document.getElementById('video-url');
const questionInput = document.getElementById('question');
const answerDiv = document.getElementById('answer');
const errorDiv = document.getElementById('error');

const BACKEND_API_URL = "https://youtube-q-a-download.onrender.com/api/ask"; // Update with your backend URL

// Utility: Check if URL matches YouTube video pattern and extract video URL
function isYouTubeVideoUrl(url) {
  // Basic check - adjust if needed
  return /^https?:\/\/(www\.)?youtube\.com\/watch\?v=[\w-]{11}/.test(url) ||
         /^https?:\/\/youtu\.be\/[\w-]{11}/.test(url) ||
         /^https?:\/\/(www\.)?youtube\.com\/shorts\/[\w-]{11}/.test(url);
}

// Get the currently active tab in the last focused chrome window
function getActiveYouTubeTabUrl() {
  return new Promise((resolve) => {
    chrome.windows.getLastFocused({ populate: true }, (window) => {
      if (!window || !window.tabs) {
        resolve(null);
        return;
      }
      // Find active tab which is also a YouTube video page
      const activeTab = window.tabs.find(tab => tab.active);
      if (activeTab && isYouTubeVideoUrl(activeTab.url)) {
        resolve(activeTab.url);
        return;
      }
      // No active tab is YouTube video url
      resolve(null);
    });
  });
}

// Function to update UI based on whether we're on YouTube video tab
async function initializePopup() {
  const youtubeUrl = await getActiveYouTubeTabUrl();
  if (youtubeUrl) {
    // Prefill input with url and enable button
    videoUrlInput.value = youtubeUrl;
    errorDiv.textContent = '';
    askBtn.disabled = false;
  } else {
    videoUrlInput.value = '';
    errorDiv.textContent = 'Please switch to a YouTube video tab to ask questions.';
    askBtn.disabled = true;
  }
}

// On popup open, try to get the active YouTube video URL
initializePopup();

askBtn.addEventListener('click', async () => {
  const video_url = videoUrlInput.value.trim();
  const question = questionInput.value.trim();

  answerDiv.textContent = '';
  errorDiv.textContent = '';

  if (!video_url || !question) {
    errorDiv.textContent = 'Please enter both YouTube video URL and your question.';
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
