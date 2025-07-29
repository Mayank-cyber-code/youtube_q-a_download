// Generate or retrieve persistent session ID for conversation memory
function getSessionId() {
  let sessionId = localStorage.getItem('yt_qna_session_id');
  if (!sessionId) {
    if ('randomUUID' in crypto) {
      sessionId = crypto.randomUUID();
    } else {
      // Fallback random ID for browsers missing crypto.randomUUID
      sessionId = Math.random().toString(36).substring(2, 15) + 
                  Math.random().toString(36).substring(2, 15);
    }
    localStorage.setItem('yt_qna_session_id', sessionId);
  }
  return sessionId;
}

const sessionId = getSessionId();

const BACKEND_BASE = "https://your-backend-url";  // CHANGE this before deployment
const API_BASE = BACKEND_BASE + "/api";

const askBtn = document.getElementById('ask-btn');
const edaBtn = document.getElementById('eda-btn');
const benchmarkBtn = document.getElementById('benchmark-btn');

const videoInput = document.getElementById('video-url');
const questionInput = document.getElementById('question');

const errorDiv = document.getElementById('error');
const answerDiv = document.getElementById('answer');

const edaResultsDiv = document.getElementById('eda-results');
const wordcloudImg = document.getElementById('wordcloud');

const benchmarkLogPre = document.getElementById('benchmark-log');

// Utility for validating YouTube URLs
function isYouTubeUrl(url) {
  return /^https?:\/\/(www\.)?youtube\.com\/watch\?v=[\w-]{11}/.test(url) ||
         /^https?:\/\/youtu\.be\/[\w-]{11}/.test(url) ||
         /^https?:\/\/(www\.)?youtube\.com\/shorts\/[\w-]{11}/.test(url);
}

// Get active YouTube tab URL to auto-fill input
function getActiveYouTubeUrl() {
  return new Promise((resolve) => {
    if (!chrome.tabs) return resolve(null);
    chrome.windows.getLastFocused({ populate: true }, (window) => {
      if (!window || !window.tabs) return resolve(null);
      const activeTab = window.tabs.find(t => t.active);
      if (activeTab && isYouTubeUrl(activeTab.url)) resolve(activeTab.url);
      else resolve(null);
    });
  });
}

// Enable/disable the Ask button based on inputs
function updateAskBtnState() {
  askBtn.disabled = !(videoInput.value.trim() && questionInput.value.trim());
}

// Initialize popup: fill URL if YouTube tab is active
async function initPopup() {
  const url = await getActiveYouTubeUrl();
  if (url) {
    videoInput.value = url;
    errorDiv.textContent = "";
  } else {
    videoInput.value = "";
    errorDiv.textContent = "Please switch to a YouTube tab to auto-fill the URL.";
  }
  updateAskBtnState();
}
initPopup();

videoInput.addEventListener('input', () => {
  errorDiv.textContent = "";
  updateAskBtnState();
});
questionInput.addEventListener('input', () => {
  errorDiv.textContent = "";
  updateAskBtnState();
});

// Ask Q&A handler
askBtn.addEventListener('click', async () => {
  errorDiv.textContent = "";
  answerDiv.textContent = "";

  const video_url = videoInput.value.trim();
  const question = questionInput.value.trim();

  if (!video_url || !question) {
    errorDiv.textContent = "Please enter the video URL and your question.";
    return;
  }

  askBtn.disabled = true;
  askBtn.textContent = "Asking…";

  try {
    const response = await fetch(`${API_BASE}/ask`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ video_url, question, session_id: sessionId }),
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      errorDiv.textContent = error.detail || "Failed to get an answer.";
      return;
    }

    const data = await response.json();

    if (data.answer) {
      answerDiv.textContent = data.answer;
    } else {
      answerDiv.textContent = "No answer received.";
    }

    if (data.latency_sec !== undefined) {
      answerDiv.textContent += `\n\n(Response time: ${data.latency_sec} seconds)`;
    }
  } catch {
    errorDiv.textContent = "Network or server error.";
  } finally {
    askBtn.disabled = false;
    askBtn.textContent = "Ask";
  }
});

// EDA handler
edaBtn.addEventListener("click", async () => {
  errorDiv.textContent = "";
  edaResultsDiv.textContent = "";
  wordcloudImg.style.display = "none";

  const video_url = videoInput.value.trim();
  if (!video_url) {
    errorDiv.textContent = "Please enter a video URL first.";
    return;
  }

  edaBtn.disabled = true;
  edaBtn.textContent = "Analyzing…";

  try {
    const response = await fetch(`${API_BASE}/eda`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ video_url }),
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      errorDiv.textContent = error.detail || "Failed to get EDA.";
      return;
    }

    const data = await response.json();

    const eda = data.eda || {};
    const commonWords = eda.common_words ? eda.common_words.map(w => w[0]).join(", ") : "";
    const tfidfKeywords = eda.tfidf_keywords ? eda.tfidf_keywords.join(", ") : "";

    edaResultsDiv.innerHTML = `
      <strong>Word Count:</strong> ${eda.word_count || 0}<br/>
      <strong>Unique Words:</strong> ${eda.unique_words || 0}<br/>
      <strong>Avg. Sentence Length:</strong> ${eda.avg_sentence_len || 0}<br/>
      <strong>Sentiment (Polarity):</strong> ${eda.sentiment?.polarity || 0}<br/>
      <strong>Sentiment (Subjectivity):</strong> ${eda.sentiment?.subjectivity || 0}<br/>
      <strong>Top 20 Common Words:</strong> ${commonWords}<br/>
      <strong>Top 10 TF-IDF Keywords:</strong> ${tfidfKeywords}<br/>
      <strong>Summary:</strong> ${data.summary || "Unavailable"}
    `;

    if (data.wordcloud_base64) {
      wordcloudImg.src = "data:image/png;base64," + data.wordcloud_base64;
      wordcloudImg.style.display = "block";
    }

  } catch {
    errorDiv.textContent = "Network or server error.";
  } finally {
    edaBtn.disabled = false;
    edaBtn.textContent = "Analyze Transcript";
  }
});

// Benchmark log handler
benchmarkBtn.addEventListener("click", async () => {
  errorDiv.textContent = "";
  benchmarkLogPre.textContent = "";

  benchmarkBtn.disabled = true;
  benchmarkBtn.textContent = "Loading…";

  try {
    const response = await fetch(`${API_BASE}/benchmark_log`);

    if (!response.ok) {
      errorDiv.textContent = "Failed to load benchmark log.";
      return;
    }

    const data = await response.json();
    benchmarkLogPre.textContent = JSON.stringify(data.answers_log, null, 2);

  } catch {
    errorDiv.textContent = "Network or server error.";
  } finally {
    benchmarkBtn.disabled = false;
    benchmarkBtn.textContent = "Load Benchmark Log";
  }
});

// Submit question on Enter (without shift)
questionInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    askBtn.click();
  }
});
