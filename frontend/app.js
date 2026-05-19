function openDashboard() {
  const loading   = document.getElementById('loading');
  const dashboard = document.getElementById('dashboard');
  loading.style.opacity = '0';
  setTimeout(() => {
    loading.style.display   = 'none';
    dashboard.style.display = 'flex';
  }, 800);
}

async function analyse() {
  const text = document.getElementById('email-input').value.trim();
  if (!text) { alert('Please enter some email text first.'); return; }

  const btn = document.getElementById('analyse-btn');
  btn.disabled  = true;
  btn.innerHTML = '<div class="spinner"></div>';

  try {
    const model = document.getElementById('model-select').value;
    const res   = await fetch('/predict', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ text, model }),
    });
    const data = await res.json();
    renderResult(data);
  } catch {
    alert('Error contacting the API. Make sure the server is running.');
  } finally {
    btn.disabled  = false;
    btn.innerHTML = 'Analyse Email';
  }
}

function renderResult(data) {
  const result = document.getElementById('result');
  result.className     = 'result ' + data.label;
  result.style.display = 'block';

  // Label
  document.getElementById('result-label').textContent =
    data.label === 'spam' ? '🚨 Spam' : '✅ Ham (Legitimate)';

  // Confidence meter
  const pct = (data.confidence * 100).toFixed(1);
  document.getElementById('result-confidence-text').textContent = `Confidence: ${pct}%`;
  const bar = document.getElementById('confidence-bar-fill');
  bar.style.width = '0%';
  bar.style.background = data.label === 'spam' ? '#dc2626' : '#16a34a';
  setTimeout(() => { bar.style.width = pct + '%'; }, 50);

  // Model
  document.getElementById('result-model').textContent =
    'Model: ' + (data.model === 'naive_bayes' ? 'Multinomial Naive Bayes' : 'Logistic Regression');

  // Token chart
  renderTokenChart(data.top_tokens);
}

function renderTokenChart(tokens) {
  const container = document.getElementById('token-chart');
  if (!tokens || tokens.length === 0) { container.style.display = 'none'; return; }

  const maxAbs = Math.max(...tokens.map(t => Math.abs(t.score)));

  container.style.display = 'block';
  container.innerHTML = '<div class="chart-title">Top token contributions</div>';

  tokens.forEach(({ token, score }) => {
    const isSpam  = score > 0;
    const pct     = ((Math.abs(score) / maxAbs) * 100).toFixed(1);
    const color   = isSpam ? '#dc2626' : '#16a34a';
    const label   = isSpam ? 'spam signal' : 'ham signal';

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
