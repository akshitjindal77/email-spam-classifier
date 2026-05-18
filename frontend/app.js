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

    const result = document.getElementById('result');
    result.className = 'result ' + data.label;
    document.getElementById('result-label').textContent =
      data.label === 'spam' ? '🚨 Spam' : '✅ Ham (Legitimate)';
    document.getElementById('result-confidence').textContent =
      'Confidence: ' + (data.confidence * 100).toFixed(1) + '%';
    document.getElementById('result-model').textContent =
      'Model: ' + (data.model === 'naive_bayes' ? 'Multinomial Naive Bayes' : 'Logistic Regression');
    result.style.display = 'block';
  } catch {
    alert('Error contacting the API. Make sure the server is running.');
  } finally {
    btn.disabled  = false;
    btn.innerHTML = 'Analyse Email';
  }
}
