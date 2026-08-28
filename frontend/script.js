const API = window.location.origin;

const $ = id => document.getElementById(id);
const uploadZone = $('upload-zone');
const fileInput = $('file-input');

// upload handlers
uploadZone.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', e => { if (e.target.files.length) handleFile(e.target.files[0]); });

uploadZone.addEventListener('dragover', e => { e.preventDefault(); uploadZone.classList.add('dragover'); });
uploadZone.addEventListener('dragleave', () => uploadZone.classList.remove('dragover'));
uploadZone.addEventListener('drop', e => {
    e.preventDefault();
    uploadZone.classList.remove('dragover');
    if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
});

async function handleFile(file) {
    if (!file.type.startsWith('image/')) {
        showError('Please upload a valid image file.');
        return;
    }
    if (file.size > 10 * 1024 * 1024) {
        showError('File too large (max 10MB).');
        return;
    }

    showLoading();

    // preview
    const reader = new FileReader();
    reader.onload = e => { $('preview-image').src = e.target.result; };
    reader.readAsDataURL(file);

    const form = new FormData();
    form.append('file', file);

    try {
        const res = await fetch(`${API}/api/analyze`, { method: 'POST', body: form });
        if (!res.ok) {
            const err = await res.json().catch(() => ({}));
            throw new Error(err.detail || `HTTP ${res.status}`);
        }
        showResults(await res.json());
        loadHistory();
    } catch (err) {
        showError(err.message || 'Analysis failed.');
    }
}

// ui states
function showLoading() {
    $('upload-section').classList.add('hidden');
    $('loading-section').classList.remove('hidden');
    $('error-section').classList.add('hidden');
    $('results-section').classList.add('hidden');
}

function showError(msg) {
    $('upload-section').classList.add('hidden');
    $('loading-section').classList.add('hidden');
    $('error-section').classList.remove('hidden');
    $('results-section').classList.add('hidden');
    $('error-message').textContent = msg;
}

function showResults(result) {
    $('loading-section').classList.add('hidden');
    $('error-section').classList.add('hidden');
    $('results-section').classList.remove('hidden');
    renderResults(result);
}

function resetUI() {
    $('upload-section').classList.remove('hidden');
    $('loading-section').classList.add('hidden');
    $('error-section').classList.add('hidden');
    $('results-section').classList.add('hidden');
    fileInput.value = '';
}

function renderResults(r) {
    const score = r.quality_score;
    const colors = { ACCEPTABLE: '#22c55e', DEGRADED: '#f59e0b', DEFECTIVE: '#ef4444' };
    const color = colors[r.quality_label] || '#6366f1';

    // gauge
    const offset = 314 - (score / 100) * 314;
    const gauge = $('gauge-fill');
    gauge.style.strokeDashoffset = offset;
    gauge.style.stroke = color;

    $('score-value').textContent = Math.round(score);
    $('score-value').style.color = color;

    const label = $('quality-label');
    label.textContent = r.quality_label;
    label.className = `quality-label ${r.quality_label}`;

    $('confidence-badge').textContent = `Confidence: ${(r.confidence * 100).toFixed(0)}%`;

    if (r.model_signals) {
        $('model-info').textContent = `${r.model_signals.model_type} | CNN: ${r.model_signals.cnn_quality_score}`;
    }

    // issues
    const list = $('issues-list');
    if (r.issues && r.issues.length) {
        list.innerHTML = r.issues.map(iss => `
            <div class="issue-item severity-${iss.severity}">
                <div class="issue-header">
                    <span class="issue-type">${iss.type.replace(/_/g, ' ')}</span>
                    <span class="issue-severity ${iss.severity}">${iss.severity}</span>
                </div>
                <div class="issue-confidence">Confidence: ${(iss.confidence * 100).toFixed(1)}%</div>
                ${iss.evidence ? `<div class="issue-evidence">${iss.evidence.map(e =>
                    `${e.feature}=${e.value} (${e.direction})`).join(', ')}</div>` : ''}
            </div>
        `).join('');
    } else {
        list.innerHTML = '<p class="no-issues">✓ No issues detected</p>';
    }

    // heatmap
    if (r.heatmap) $('heatmap-image').src = r.heatmap;

    // stats
    const statNames = {
        width: 'Width', height: 'Height', mean_brightness: 'Brightness',
        sharpness_laplacian_var: 'Sharpness', noise_sigma_estimate: 'Noise σ',
        contrast_rms: 'Contrast', mean_saturation: 'Saturation',
        colorfulness: 'Colorfulness', entropy: 'Entropy', dynamic_range: 'Dynamic Range',
    };
    if (r.image_stats) {
        $('stats-grid').innerHTML = Object.entries(r.image_stats).map(([k, v]) => `
            <div class="stat-item">
                <div class="stat-label">${statNames[k] || k}</div>
                <div class="stat-value">${typeof v === 'number' ? v.toFixed(1) : v}</div>
            </div>
        `).join('');
    }
}

// history
async function loadHistory() {
    try {
        const res = await fetch(`${API}/api/analyses?limit=20`);
        if (!res.ok) return;
        const data = await res.json();
        const tbody = $('history-body');

        if (!data.analyses || !data.analyses.length) {
            tbody.innerHTML = '<tr><td colspan="6" class="empty-row">No analyses yet</td></tr>';
            return;
        }

        tbody.innerHTML = data.analyses.map(a => `
            <tr>
                <td>${a.id}</td>
                <td title="${a.filename}">${a.filename && a.filename.length > 25 ? a.filename.slice(0, 25) + '...' : (a.filename || '')}</td>
                <td><strong>${Math.round(a.quality_score)}</strong></td>
                <td><span class="label-badge ${a.quality_label}">${a.quality_label}</span></td>
                <td>${a.issues ? a.issues.length : 0}</td>
                <td>${a.created_at ? new Date(a.created_at).toLocaleString() : '-'}</td>
            </tr>
        `).join('');
    } catch (e) {
        console.error('History load failed:', e);
    }
}

document.addEventListener('DOMContentLoaded', loadHistory);
