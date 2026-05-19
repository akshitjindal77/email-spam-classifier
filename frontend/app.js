function openDashboard() {
  const loading   = document.getElementById('loading');
  const dashboard = document.getElementById('dashboard');
  loading.style.opacity = '0';
  setTimeout(() => {
    loading.style.display   = 'none';
    dashboard.style.display = 'flex';
  }, 800);
}

async function fetchPredict(text, model) {
  const res = await fetch('/predict', {
    method:  'POST',
    headers: { 'Content-Type': 'application/json' },
    body:    JSON.stringify({ text, model }),
  });
  return res.json();
}

async function analyse() {
  const text = document.getElementById('email-input').value.trim();
  if (!text) { alert('Please enter some email text first.'); return; }

  const model = document.getElementById('model-select').value;
  const btn   = document.getElementById('analyse-btn');
  btn.disabled  = true;
  btn.innerHTML = '<div class="spinner"></div>';

  try {
    if (model === 'both') {
      const [nbData, lrData] = await Promise.all([
        fetchPredict(text, 'naive_bayes'),
        fetchPredict(text, 'logreg'),
      ]);
      renderBothResults(nbData, lrData);
    } else {
      const data = await fetchPredict(text, model);
      renderSingleResult(data);
    }
  } catch {
    alert('Error contacting the API. Make sure the server is running.');
  } finally {
    btn.disabled  = false;
    btn.innerHTML = 'Analyse Email';
  }
}

function renderSingleResult(data) {
  document.querySelector('.card').classList.remove('wide');

  const result = document.getElementById('result');
  result.className     = data.label;
  result.style.display = 'block';
  result.innerHTML     = buildResultHTML(data);

  setTimeout(() => animateBar(data.model, data.confidence, data.label), 50);
  renderTokenChart(data.top_tokens, `token-chart-${data.model}`);
}

function renderBothResults(nbData, lrData) {
  document.querySelector('.card').classList.add('wide');

  const result = document.getElementById('result');
  result.className     = 'both';
  result.style.display = 'block';
  result.innerHTML = `
    <div class="results-grid">
      <div class="result-panel ${nbData.label}">${buildResultHTML(nbData)}</div>
      <div class="result-panel ${lrData.label}">${buildResultHTML(lrData)}</div>
    </div>`;

  setTimeout(() => {
    animateBar(nbData.model, nbData.confidence, nbData.label);
    animateBar(lrData.model, lrData.confidence, lrData.label);
  }, 50);

  renderTokenChart(nbData.top_tokens, `token-chart-${nbData.model}`);
  renderTokenChart(lrData.top_tokens, `token-chart-${lrData.model}`);
}

function buildResultHTML(data) {
  const labelText = data.label === 'spam' ? '🚨 Spam' : '✅ Ham (Legitimate)';
  const pct       = (data.confidence * 100).toFixed(1);
  const modelName = data.model === 'naive_bayes' ? 'Multinomial Naive Bayes' : 'Logistic Regression';
  return `
    <div class="result-label">${labelText}</div>
    <div class="confidence-meter">
      <div class="confidence-bar-bg">
        <div id="bar-${data.model}" class="confidence-bar-fill"></div>
      </div>
      <span class="result-confidence">Confidence: ${pct}%</span>
    </div>
    <div id="token-chart-${data.model}"></div>
    <div class="result-model">Model: ${modelName}</div>`;
}

function animateBar(modelId, confidence, label) {
  const bar = document.getElementById(`bar-${modelId}`);
  if (!bar) return;
  bar.style.background = label === 'spam' ? '#dc2626' : '#16a34a';
  setTimeout(() => { bar.style.width = (confidence * 100).toFixed(1) + '%'; }, 50);
}

function renderTokenChart(tokens, containerId) {
  const container = document.getElementById(containerId);
  if (!container || !tokens || tokens.length === 0) return;

  const maxAbs = Math.max(...tokens.map(t => Math.abs(t.score)));
  container.style.display = 'block';
  container.innerHTML = '<div class="chart-title">Top token contributions</div>';

  tokens.forEach(({ token, score }) => {
    const isSpam = score > 0;
    const pct    = ((Math.abs(score) / maxAbs) * 100).toFixed(1);
    const color  = isSpam ? '#dc2626' : '#16a34a';
    const label  = isSpam ? 'spam signal' : 'ham signal';
    container.innerHTML += `
      <div class="chart-row">
        <span class="chart-token" title="${label}">${token}</span>
        <div class="chart-bar-wrap">
          <div class="chart-bar" style="width:${pct}%;background:${color};"></div>
        </div>
        <span class="chart-score" style="color:${color};">${score > 0 ? '+' : ''}${score.toFixed(3)}</span>
      </div>`;
  });
}
