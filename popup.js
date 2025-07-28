const askBtn = document.getElementById('ask-btn');
const edaBtn = document.getElementById('eda-btn');
const benchmarkBtn = document.getElementById('benchmark-btn');
const videoUrlInput = document.getElementById('video-url');
const questionInput = document.getElementById('question');
const answerDiv = document.getElementById('answer');
const errorDiv = document.getElementById('error');
const edaResultsDiv = document.getElementById('eda-results');
const wordcloudImg = document.getElementById('wordcloud');
const benchmarkLogPre = document.getElementById('benchmark-log');

// Replace with your actual backend URL (update here!)
const BACKEND_BASE_URL = "https://youtube-q-a-download.onrender.com/api";

const BACKEND_API_URL = `${BACKEND_BASE_URL}/ask`;
const EDA_API_URL = `${BACKEND_BASE_URL}/eda`;
const BENCHMARK_API_URL = `${BACKEND_BASE_URL}/benchmark_log`;

// Utility: Check if URL matches YouTube video pattern
function isYouTubeVideoUrl(url) {
  return /^https?:\/\/(www\.)?youtube\.com\/watch\?v=[\w-]{11}/.test(url) ||
         /^https?:\/\/youtu\.be\/[\w-]{11}/.test(url) ||
         /^https?:\/\/(www\.)?youtube\.com\/shorts\/[\w-]{11}/.test(url);
}

// Get the current active YouTube video tab URL
function getActiveYouTubeTabUrl() {
  return new Promise((resolve) => {
    chrome.windows.getLastFocused({ populate: true }, (window) => {
      if (!window || !window.tabs) {
        resolve(null);
        return;
      }
      const activeTab = window.tabs.find(tab => tab.active);
      if (activeTab && isYouTubeVideoUrl(activeTab.url)) {
        resolve(activeTab.url);
      } else {
        resolve(null);
      }
    });
  });
}

// Enable Ask button only if video URL and question inputs are valid
function updateAskButtonState() {
  askBtn.disabled = !(videoUrlInput.value.trim() && questionInput.value.trim());
}

// Initialize popup: prefill video URL if possible and control Ask button
async function initializePopup() {
  const youtubeUrl = await getActiveYouTubeTabUrl();
  if (youtubeUrl) {
    videoUrlInput.value = youtubeUrl;
    errorDiv.textContent = '';
  } else {
    videoUrlInput.value = '';
    errorDiv.textContent = 'Please switch to a YouTube video tab to ask questions.';
  }
  updateAskButtonState();
}

initializePopup();

// Update ask button state when inputs change
videoUrlInput.addEventListener('input', () => {
  errorDiv.textContent = '';
  updateAskButtonState();
});
questionInput.addEventListener('input', () => {
  errorDiv.textContent = '';
  updateAskButtonState();
});

// Ask Q&A API call
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

// EDA API call
edaBtn.addEventListener('click', async () => {
  const video_url = videoUrlInput.value.trim();

  edaResultsDiv.textContent = '';
  wordcloudImg.style.display = 'none';
  errorDiv.textContent = '';

  if (!video_url) {
    errorDiv.textContent = 'Please enter the YouTube video URL for EDA.';
    return;
  }

  edaBtn.disabled = true;
  edaBtn.textContent = 'Analyzing...';

  try {
    const response = await fetch(EDA_API_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ video_url }),
    });

    if (!response.ok) {
      const err = await response.json();
      errorDiv.textContent = err.detail || 'Error: Failed to get EDA.';
      return;
    }

    const data = await response.json();

    const eda = data.eda || {};
    const commonWords = eda.common_words ? eda.common_words.map(w => w[0]).join(', ') : '';
    const tfidfKeywords = eda.tfidf_keywords ? eda.tfidf_keywords.join(', ') : '';

    let edaHtml = `
      <strong>Word Count:</strong> ${eda.word_count || 0} <br/>
      <strong>Unique Words:</strong> ${eda.unique_words || 0} <br/>
      <strong>Avg. Sentence Length:</strong> ${eda.avg_sentence_len || 0} <br/>
      <strong>Sentiment (Polarity):</strong> ${eda.sentiment?.polarity || 0} <br/>
      <strong>Sentiment (Subjectivity):</strong> ${eda.sentiment?.subjectivity || 0} <br/>
      <strong>Top Common Words:</strong> ${commonWords} <br/>
      <strong>TF-IDF Keywords:</strong> ${tfidfKeywords} <br/>
      <strong>Summary:</strong> ${data.summary || 'Unavailable'}
    `;

    edaResultsDiv.innerHTML = edaHtml;

    if (data.wordcloud_base64) {
      wordcloudImg.src = "data:image/png;base64," + data.wordcloud_base64;
      wordcloudImg.style.display = '';
    }
  } catch (error) {
    errorDiv.textContent = 'Network error or server unreachable.';
  } finally {
    edaBtn.disabled = false;
    edaBtn.textContent = 'Analyze Transcript';
  }
});

// Benchmark Log API call
benchmarkBtn.addEventListener('click', async () => {
  benchmarkLogPre.textContent = '';
  errorDiv.textContent = '';
  benchmarkBtn.disabled = true;
  benchmarkBtn.textContent = 'Loading...';

  try {
    const response = await fetch(BENCHMARK_API_URL);

    if (!response.ok) {
      errorDiv.textContent = 'Error loading benchmark log.';
      return;
    }

    const data = await response.json();
    benchmarkLogPre.textContent = JSON.stringify(data.answers_log, null, 2);
  } catch (error) {
    errorDiv.textContent = 'Network error or server unreachable.';
  } finally {
    benchmarkBtn.disabled = false;
    benchmarkBtn.textContent = 'Load Benchmark Log';
  }
});

// Optional: Submit Q&A on Enter (Shift+Enter for newline)
questionInput.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    askBtn.click();
  }
});
