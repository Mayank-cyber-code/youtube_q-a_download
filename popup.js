// Generate or retrieve persistent session ID for conversational memory
function getSessionId() {
  let sessionId = localStorage.getItem('yt_qna_session_id');
  if (!sessionId) {
    if ('randomUUID' in crypto) {
      sessionId = crypto.randomUUID();
    } else {
      // Fallback for older browsers
      sessionId = Math.random().toString(36).substring(2, 15) +
                  Math.random().toString(36).substring(2, 15);
    }
    localStorage.setItem('yt_qna_session_id', sessionId);
  }
  return sessionId;
}

const sessionId = getSessionId();

// Change this to your deployed backend URL
const BACKEND_BASE = "https://youtube-q-a-download.onrender.com";
const API_BASE = `${BACKEND_BASE}/api`;

const askBtn = document.getElementById('ask-btn');
const edaBtn = document.getElementById('eda-btn');
const benchmarkBtn = document.getElementById('benchmark-btn');

const videoInput = document.getElementById('video-url');
const questionInput = document.getElementById('question');

const errorDiv = document.getElementById('error');
const answerDiv = document.getElementById('answer');

const edaResultsDiv = document.getElementById('eda-results');
const wordcloudImg = document.getElementById('wordcloud');

const benchmarkLog = document.getElementById('benchmark-log');

// Validate YouTube video URLs
function isYouTubeUrl(url) {
  return /^https?:\/\/(www\.)?youtube\.com\/watch\?v=[\w-]{11}/.test(url) ||
         /^https?:\/\/youtu\.be\/[\w-]{11}/.test(url) ||
         /^https?:\/\/(www\.)?youtube\.com\/shorts\/[\w-]{11}/.test(url);
}

// Get active YouTube tab URL (only works in Chrome extension context)
function getActiveYouTubeUrl() {
  return new Promise((resolve) => {
    if (!chrome || !chrome.tabs) return resolve(null);
    chrome.windows.getLastFocused({populate: true}, (win) => {
      if (!win || !win.tabs) return resolve(null);
      const activeTab = win.tabs.find(tab => tab.active);
      if (activeTab && isYouTubeUrl(activeTab.url)) {
        resolve(activeTab.url);
      } else {
        resolve(null);
      }
    });
  });
}

// Update "Ask" button state based on inputs
function updateAskButtonState() {
  const enabled = videoInput.value.trim() !== '' && questionInput.value.trim() !== '';
  askBtn.disabled = !enabled;
}

// Initialize popup with autofill if possible
async function initializePopup() {
  const ytUrl = await getActiveYouTubeUrl();
  if (ytUrl) {
    videoInput.value = ytUrl;
    errorDiv.textContent = '';
  } else {
    videoInput.value = '';
    errorDiv.textContent = 'Please switch to a YouTube tab to autofill URL.';
  }
  updateAskButtonState();
}

initializePopup();

videoInput.addEventListener('input', () => {
  errorDiv.textContent = '';
  updateAskButtonState();
});

questionInput.addEventListener('input', () => {
  errorDiv.textContent = '';
  updateAskButtonState();
});

// Ask button click handler
askBtn.addEventListener('click', async () => {
  errorDiv.textContent = '';
  answerDiv.textContent = '';

  const video_url = videoInput.value.trim();
  const question = questionInput.value.trim();

  if (!video_url || !question) {
    errorDiv.textContent = 'Please enter both YouTube video URL and your question.';
    return;
  }

  askBtn.disabled = true;
  askBtn.textContent = 'Asking…';

  try {
    const response = await fetch(`${API_BASE}/ask`, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({video_url, question, session_id: sessionId}),
    });

    if (!response.ok) {
      const errData = await response.json().catch(() => ({}));
      errorDiv.textContent = errData.detail || 'Failed to get answer.';
      return;
    }

    const data = await response.json();

    if (data.answer) {
      answerDiv.textContent = data.answer;
      if (data.latency_sec !== undefined) {
        answerDiv.textContent += `\n\n(Response time: ${data.latency_sec} seconds)`;
      }
    } else {
      answerDiv.textContent = 'No answer received from server.';
    }

  } catch (err) {
    console.error(err);
    errorDiv.textContent = 'Network error or server unreachable.';
  } finally {
    askBtn.disabled = false;
    askBtn.textContent = 'Ask';
  }
});

// EDA button click handler
edaBtn.addEventListener('click', async () => {
  errorDiv.textContent = '';
  edaResultsDiv.textContent = '';
  wordcloudImg.style.display = 'none';
  wordcloudImg.src = '';

  const video_url = videoInput.value.trim();

  if (!video_url) {
    errorDiv.textContent = 'Please enter a YouTube video URL first.';
    return;
  }

  edaBtn.disabled = true;
  edaBtn.textContent = 'Analyzing…';

  try {
    const response = await fetch(`${API_BASE}/eda`, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({video_url}),
    });

    if (!response.ok) {
      const errData = await response.json().catch(() => ({}));
      errorDiv.textContent = errData.detail || 'Failed to get EDA data.';
      return;
    }

    const data = await response.json();

    const eda = data.eda || {};
    const commonWords = eda.common_words ? eda.common_words.map(w => w[0]).join(', ') : '';
    const tfidfKeywords = eda.tfidf_keywords ? eda.tfidf_keywords.join(', ') : '';

    edaResultsDiv.innerHTML = `
      <strong>Word Count:</strong> ${eda.word_count || 0}<br/>
      <strong>Unique Words:</strong> ${eda.unique_words || 0}<br/>
      <strong>Avg. Sentence Length:</strong> ${eda.avg_sentence_len || 0}<br/>
      <strong>Sentiment (Polarity):</strong> ${eda.sentiment?.polarity || 0}<br/>
      <strong>Sentiment (Subjectivity):</strong> ${eda.sentiment?.subjectivity || 0}<br/>
      <strong>Top 20 Common Words:</strong> ${commonWords}<br/>
      <strong>Top 10 TF-IDF Keywords:</strong> ${tfidfKeywords}<br/>
      <strong>Summary:</strong> ${data.summary || 'Unavailable'}
    `;

    if (data.wordcloud_base64) {
      wordcloudImg.src = "data:image/png;base64," + data.wordcloud_base64;
      wordcloudImg.style.display = 'block';
    }
  } catch (err) {
    console.error(err);
    errorDiv.textContent = 'Network error or server unreachable.';
  } finally {
    edaBtn.disabled = false;
    edaBtn.textContent = 'Analyze Transcript';
  }
});

// Benchmark log button click handler
benchmarkBtn.addEventListener('click', async () => {
  errorDiv.textContent = '';
  benchmarkLog.textContent = '';

  benchmarkBtn.disabled = true;
  benchmarkBtn.textContent = 'Loading…';

  try {
    const response = await fetch(`${API_BASE}/benchmark_log`);

    if (!response.ok) {
      errorDiv.textContent = 'Failed to load benchmark log.';
      return;
    }

    const data = await response.json();

    benchmarkLog.textContent = JSON.stringify(data.answers_log, null, 2);

  } catch (err) {
    console.error(err);
    errorDiv.textContent = 'Network error or server unreachable.';
  } finally {
    benchmarkBtn.disabled = false;
    benchmarkBtn.textContent = 'Load Benchmark Log';
  }
});

// Submit question on Enter key without Shift
questionInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    askBtn.click();
  }
});
