/* ============================================================
   Health Setu — Client Script
   ============================================================ */

/* ---- Mobile nav ---- */
(function () {
  var toggle = document.getElementById('navToggle');
  var links  = document.getElementById('navLinks');
  if (!toggle || !links) return;
  toggle.addEventListener('click', function () { links.classList.toggle('open'); });
  links.querySelectorAll('a').forEach(function (a) {
    a.addEventListener('click', function () { links.classList.remove('open'); });
  });
})();

/* ============================================================
   SYMPTOM GUIDANCE
   ============================================================ */
(function () {
  var form        = document.getElementById('symptomForm');
  if (!form) return;

  var errorBox    = document.getElementById('formError');
  var resultPanel = document.getElementById('resultPanel');
  var resultBody  = document.getElementById('resultBody');
  var submitBtn   = document.getElementById('submitBtn');

  form.addEventListener('submit', function (e) {
    e.preventDefault();
    clearError();

    var checked = Array.from(form.querySelectorAll('input[type="checkbox"]:checked'))
                       .map(function (cb) { return cb.value; });

    if (!checked.length) {
      showError('Please select at least one symptom before submitting.');
      return;
    }

    setLoading(true);
    resultPanel.style.display = 'block';
    resultBody.innerHTML = loadingHTML();

    fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ symptoms: checked }),
    })
      .then(function (r) {
        return r.json().then(function (d) { return { ok: r.ok, data: d }; });
      })
      .then(function (res) {
        if (!res.ok) throw new Error(res.data.error || 'Server error');
        renderResult(res.data);
      })
      .catch(function (err) {
        showError(err.message);
        resultPanel.style.display = 'none';
      })
      .finally(function () { setLoading(false); });
  });

  function setLoading(on) {
    if (!submitBtn) return;
    submitBtn.disabled = on;
    submitBtn.textContent = on ? 'Analysing…' : 'Get Guidance';
  }

  function showError(msg) {
    errorBox.textContent = msg;
    errorBox.style.display = 'flex';
  }

  function clearError() {
    errorBox.textContent = '';
    errorBox.style.display = 'none';
  }

  function loadingHTML() {
    return '<div style="padding:2rem;text-align:center;color:#64748b">Analysing your symptoms\u2026</div>';
  }

  var URGENCY_LABELS = {
    low:       'Low urgency',
    moderate:  'See a doctor if unsure',
    high:      'Seek attention soon',
    emergency: 'EMERGENCY \u2014 seek immediate care',
  };

  function renderResult(data) {
    var a  = data.advice || {};
    var urg = (a.urgency || 'moderate').toLowerCase();
    var urgLabel = URGENCY_LABELS[urg] || urg;

    var stepsHtml = (a.next_steps || []).map(function (s, i) {
      return '<li><span class="step-dot">' + (i + 1) + '</span><span>' + esc(s) + '</span></li>';
    }).join('');

    resultBody.innerHTML =
      '<div class="result-condition-row">' +
        '<div class="result-icon">' + (a.icon || '\u2139\uFE0F') + '</div>' +
        '<div>' +
          '<div class="result-condition">' + esc(data.condition) + '</div>' +
          '<div class="result-conf">Model confidence: ' + data.confidence + '%</div>' +
        '</div>' +
      '</div>' +
      '<div class="urgency-pill urgency-' + urg + '">' + esc(urgLabel) + '</div>' +
      '<p class="result-info">' + esc(a.general_info || '') + '</p>' +
      '<div class="result-steps-title">Suggested next steps</div>' +
      '<ul class="result-steps">' + stepsHtml + '</ul>' +
      '<div class="result-disclaimer">' + esc(data.disclaimer || '') + '</div>';

    resultPanel.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }

  function esc(s) {
    return String(s)
      .replace(/&/g,'&amp;').replace(/</g,'&lt;')
      .replace(/>/g,'&gt;').replace(/"/g,'&quot;');
  }
})();

/* ============================================================
   HOSPITAL FINDER
   ============================================================ */
(function () {
  var searchBtn = document.getElementById('searchBtn');
  if (!searchBtn) return;

  var clearBtn  = document.getElementById('clearBtn');
  var cityEl    = document.getElementById('cityFilter');
  var nameEl    = document.getElementById('nameSearch');
  var loading   = document.getElementById('hospitalLoading');
  var errBox    = document.getElementById('hospitalError');
  var grid      = document.getElementById('hospitalResults');
  var noRes     = document.getElementById('noResults');

  // Load all on page init
  doSearch();

  searchBtn.addEventListener('click', doSearch);

  clearBtn.addEventListener('click', function () {
    cityEl.value = '';
    nameEl.value = '';
    doSearch();
  });

  nameEl.addEventListener('keydown', function (e) {
    if (e.key === 'Enter') doSearch();
  });

  function doSearch() {
    var city  = cityEl.value;
    var query = nameEl.value.trim();

    loading.style.display = 'block';
    errBox.style.display  = 'none';
    grid.innerHTML        = '';
    noRes.style.display   = 'none';

    var p = new URLSearchParams();
    if (city)  p.append('city',  city);
    if (query) p.append('query', query);

    fetch('/api/hospitals?' + p.toString())
      .then(function (r) {
        if (!r.ok) throw new Error('Failed to load hospital data (' + r.status + ')');
        return r.json();
      })
      .then(function (list) {
        loading.style.display = 'none';
        if (!list.length) { noRes.style.display = 'block'; return; }
        grid.innerHTML = list.map(buildCard).join('');
      })
      .catch(function (err) {
        loading.style.display = 'none';
        errBox.textContent = err.message;
        errBox.style.display = 'flex';
      });
  }

  function buildCard(h) {
    var isGov = (h.type || '').toLowerCase().includes('government');
    var typeClass = isGov ? 'badge-gov' : 'badge-pvt';

    return (
      '<div class="hospital-card">' +
        '<div class="hospital-card-top">' +
          '<div class="hospital-name">' + esc(h.name) + '</div>' +
          '<span class="badge ' + typeClass + '">' + esc(h.type || 'Healthcare') + '</span>' +
        '</div>' +
        '<div class="hospital-detail"><span class="di">📍</span><span>' + esc(h.address) + '</span></div>' +
        '<div class="hospital-detail"><span class="di">📞</span><span>' + esc(h.phone) + '</span></div>' +
        '<div class="hospital-detail"><span class="di">🏙️</span><span>' + esc(h.city) + '</span></div>' +
        (h.emergency === 'Yes' ? '<div class="badge-emergency">🚨 Emergency Services</div>' : '') +
        '<div class="hospital-footer"><span class="badge badge-demo">' + esc(h.note || 'Demo Data') + '</span></div>' +
      '</div>'
    );
  }

  function esc(s) {
    return String(s || '')
      .replace(/&/g,'&amp;').replace(/</g,'&lt;')
      .replace(/>/g,'&gt;').replace(/"/g,'&quot;');
  }
})();
