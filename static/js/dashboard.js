// Salon Intelligence Hub - Frontend Dashboard Controller

let currentVideoName = "";
let currentZones = {};
let activeZoneKey = "station_1";
let isDrawing = false;
let startX = 0, startY = 0;

let bgImage = new Image();
let isLivePreview = false;
let livePreviewTimer = null;

// ----------------------------------------------------------------------
// Dashboard Polling & UI Rendering
// ----------------------------------------------------------------------

function updateDashboard() {
    fetch('/api/data')
        .then(res => res.json())
        .then(data => {
            // Update KPIs
            document.getElementById('entries-count').innerText = data.total_entries;
            document.getElementById('barber-count').innerText = data.active_barbers_count;
            document.getElementById('load-factor').innerText = data.load_factor || "0%";
            document.getElementById('turnaround-rate').innerText = data.turnaround_rate || "0.0/hr";

            // Sync video dropdown if options changed
            syncVideoDropdown(data.available_videos, data.current_video);

            // Render styling stations
            renderStations(data.stations || data.chairs || {});

            // Render waiting lounge
            renderWaitingLounge(data.waiting);

            // Render stylist ledger
            renderStylists(data.barbers || {});
        })
        .catch(err => console.error("Error fetching dashboard data:", err));
}

function syncVideoDropdown(availableVideos, activeVideo) {
    if (!availableVideos || !activeVideo) return;
    const selectEl = document.getElementById('video-select');

    if (currentVideoName !== activeVideo || selectEl.options.length !== availableVideos.length) {
        currentVideoName = activeVideo;
        selectEl.innerHTML = '';
        availableVideos.forEach(vid => {
            const opt = document.createElement('option');
            opt.value = vid;
            opt.innerText = vid;
            if (vid === activeVideo) opt.selected = true;
            selectEl.appendChild(opt);
        });
    }
}

function renderStations(stations) {
    const container = document.getElementById('stations-container');
    container.innerHTML = '';

    Object.keys(stations).forEach(zid => {
        const station = stations[zid];
        const isOccupied = station.status === 'Occupied';
        const dotClass = isOccupied ? 'status-occupied' : 'status-vacant';
        const durationStr = isOccupied ? `${station.current_duration}s` : '--';
        const remarksStr = station.remarks ? station.remarks : (isOccupied ? 'In Active Service' : 'Ready for Client');

        const row = `
            <div class="station-row">
                <div class="station-info">
                    <div class="station-dot ${dotClass}"></div>
                    <div style="overflow: hidden;">
                        <div class="station-title">${station.name}</div>
                        <div class="station-desc" title="${remarksStr}">${remarksStr}</div>
                    </div>
                </div>
                <div class="station-stats">
                    <div class="stat-col">
                        <span class="stat-lbl">Active Time</span>
                        <span class="stat-val ${isOccupied ? 'active-time' : ''}">${durationStr}</span>
                    </div>
                    <div class="stat-col">
                        <span class="stat-lbl">Total Occupied</span>
                        <span class="stat-val">${station.total_occupancy_time || 0}s</span>
                    </div>
                    <div class="stat-col">
                        <span class="stat-lbl">Avg Duration</span>
                        <span class="stat-val">${station.average_duration}s</span>
                    </div>
                    <div class="stat-col">
                        <span class="stat-lbl">Clients</span>
                        <span class="stat-val">${station.total_services}</span>
                    </div>
                </div>
            </div>
        `;
        container.insertAdjacentHTML('beforeend', row);
    });
}

function renderWaitingLounge(waitingData) {
    document.getElementById('waiting-count').innerText = waitingData.count;
    document.getElementById('waiting-avg').innerText = `${waitingData.avg_wait_time}s`;

    const container = document.getElementById('waiting-container');
    container.innerHTML = '';

    if (!waitingData.queue || waitingData.queue.length === 0) {
        container.innerHTML = '<p style="font-size: 11px; color: var(--text-muted); text-align: center; padding: 10px;">No clients currently waiting</p>';
        return;
    }

    waitingData.queue.forEach(item => {
        const card = `
            <div class="barber-card" style="border-color: rgba(191, 163, 124, 0.15); margin-bottom: 8px; padding: 12px 16px;">
                <div class="barber-meta">
                    <div class="barber-pic" style="background: #efe4da; border-color: var(--accent-gold); width: 32px; height: 32px; font-size: 11px;">C</div>
                    <div class="barber-info">
                        <h4 style="font-size: 13px;">Client #${item.client_id}</h4>
                        <p style="font-size: 10px;">Waiting in lounge</p>
                    </div>
                </div>
                <div style="display: flex; align-items: center; gap: 15px;">
                    <span class="barber-badge idle" style="background: rgba(191, 163, 124, 0.1); color: var(--accent-gold); border: 1px solid rgba(191, 163, 124, 0.15); font-size: 9px; padding: 3px 8px;">Waiting</span>
                    <span class="barber-clock" style="font-size: 13px;">${item.duration}s</span>
                </div>
            </div>
        `;
        container.insertAdjacentHTML('beforeend', card);
    });
}

function renderStylists(barbers) {
    const container = document.getElementById('barbers-container');
    container.innerHTML = '';

    Object.keys(barbers).forEach(bid => {
        const barber = barbers[bid];
        const initial = barber.name.charAt(0);
        const isServicing = barber.status === 'Servicing';
        const statusClass = isServicing ? 'servicing' : 'idle';
        const statusText = isServicing ? 'Servicing Client' : 'Idle / Between Clients';
        const avatarHtml = barber.avatar ? 
            `<img src="${barber.avatar}" style="width: 42px; height: 42px; border-radius: 50%; object-fit: cover; border: 2px solid var(--accent-gold);" alt="${barber.name}">` :
            `<div class="barber-pic">${initial}</div>`;

        const card = `
            <div class="barber-card" style="padding: 14px 18px; margin-bottom: 12px; border-radius: 12px; background: #ffffff; border: 1px solid var(--border-salon); display: flex; justify-content: space-between; align-items: center; box-shadow: 0 2px 8px rgba(0,0,0,0.02);">
                <div class="barber-meta" style="display: flex; align-items: center; gap: 14px;">
                    ${avatarHtml}
                    <div class="barber-info">
                        <h4 style="font-size: 15px; font-weight: 700; color: var(--text-dark); margin: 0 0 2px 0;">${barber.name}</h4>
                        <p style="font-size: 11px; font-weight: 600; color: var(--accent-gold); margin: 0 0 3px 0;">📍 ${barber.station_name || 'Styling Station'} • <span style="color: #666;">${barber.specialization || 'Senior Stylist'}</span></p>
                        <p style="font-size: 11px; color: var(--text-muted); margin: 0;">⚡ Status: <strong style="color: ${isServicing ? 'var(--luxury-green)' : '#999'};">${statusText}</strong> ${isServicing ? `(${barber.current_duration}s active)` : ''}</p>
                    </div>
                </div>
                <div style="display: flex; flex-direction: column; align-items: flex-end; gap: 6px;">
                    <span class="barber-badge ${statusClass}" style="font-size: 10px; padding: 4px 10px; text-transform: uppercase;">${barber.status}</span>
                    <div style="font-size: 11px; color: var(--text-muted); text-align: right;">
                        <div>Total Time: <strong style="color: var(--text-dark);">${barber.total_service_time}s</strong></div>
                        <div>Served: <strong style="color: var(--text-dark);">${barber.total_clients || 0} clients</strong></div>
                    </div>
                </div>
            </div>
        `;
        container.insertAdjacentHTML('beforeend', card);
    });
}

// ----------------------------------------------------------------------
// Video Switching & Upload Handlers
// ----------------------------------------------------------------------

function onVideoSelectChanged() {
    const selectedVideo = document.getElementById('video-select').value;
    fetch('/api/select_video', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({video: selectedVideo})
    })
    .then(res => res.json())
    .then(() => {
        document.getElementById('video-stream').src = `/video_feed?t=${Date.now()}`;
    });
}

function onUploadVideo(event) {
    const file = event.target.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append('file', file);

    fetch('/api/upload_video', {
        method: 'POST',
        body: formData
    })
    .then(res => res.json())
    .then(() => {
        alert("Footage uploaded successfully! Starting analytics...");
        document.getElementById('video-stream').src = `/video_feed?t=${Date.now()}`;
    });
}

// ----------------------------------------------------------------------
// Calibration Studio & Canvas Logic
// ----------------------------------------------------------------------

function openCalibrationModal() {
    document.getElementById('calib-modal').style.display = 'flex';
    refreshCalibrationSnapshot();

    fetch(`/api/get_zones?video=${currentVideoName}`)
        .then(r => r.json())
        .then(data => {
            currentZones = data.zones || {};
            renderZoneEditorList();
            drawCalibrationCanvas();
        });
}

function closeCalibrationModal() {
    document.getElementById('calib-modal').style.display = 'none';
    if (isLivePreview) toggleLivePreview();
}

function refreshCalibrationSnapshot() {
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.src = `/api/snapshot?t=${Date.now()}`;
    img.onload = () => {
        bgImage = img;
        drawCalibrationCanvas();
    };
}

function toggleLivePreview() {
    isLivePreview = !isLivePreview;
    const btn = document.getElementById('btn-live-preview');

    if (isLivePreview) {
        btn.style.background = 'var(--luxury-rose)';
        btn.style.color = '#fff';
        btn.innerText = '⏸ Freeze Frame';
        livePreviewTimer = setInterval(refreshCalibrationSnapshot, 200);
    } else {
        btn.style.background = '#faf8f5';
        btn.style.color = 'var(--text-dark)';
        btn.innerText = '▶ Live Stream Preview';
        clearInterval(livePreviewTimer);
    }
}

function renderZoneEditorList() {
    const container = document.getElementById('zone-editor-list');
    container.innerHTML = '';

    const keys = Object.keys(currentZones);
    if (keys.length === 0) {
        container.innerHTML = '<p style="font-size: 12px; color: var(--text-muted); text-align: center;">No zones configured. Click "+ Add New Zone" to create one.</p>';
        return;
    }

    if (!currentZones[activeZoneKey]) {
        activeZoneKey = keys[0];
    }

    keys.forEach(k => {
        const z = currentZones[k];
        const isActive = (k === activeZoneKey);

        const card = document.createElement('div');
        card.className = `zone-item-card ${isActive ? 'active' : ''}`;
        card.onclick = (e) => {
            if (e.target.tagName !== 'INPUT' && e.target.tagName !== 'SELECT' && e.target.tagName !== 'BUTTON') {
                activeZoneKey = k;
                renderZoneEditorList();
                drawCalibrationCanvas();
            }
        };

        card.innerHTML = `
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="font-size: 11px; font-weight: 700; color: var(--text-muted); text-transform: uppercase;">Key: ${k}</span>
                <div style="display: flex; gap: 6px; align-items: center;">
                    <select class="zone-select" onchange="onZoneTypeChange('${k}', this.value)">
                        <option value="service" ${z.type === 'service' ? 'selected' : ''}>Styling Station</option>
                        <option value="waiting" ${z.type === 'waiting' ? 'selected' : ''}>Waiting Lounge</option>
                    </select>
                    <button style="background: none; border: none; cursor: pointer; font-size: 14px;" title="Delete Zone" onclick="deleteZone('${k}')">🗑</button>
                </div>
            </div>
            <div>
                <label style="font-size: 10px; font-weight: 600; color: var(--text-muted);">Zone Name / Label:</label>
                <input type="text" class="zone-input" value="${z.name || ''}" placeholder="e.g. VIP Haircut Chair" oninput="onZoneNameChange('${k}', this.value)">
            </div>
            <div>
                <label style="font-size: 10px; font-weight: 600; color: var(--text-muted);">Remarks / Stylist Note:</label>
                <input type="text" class="zone-input" value="${z.remarks || ''}" placeholder="e.g. Alex - Beard & Styling" oninput="onZoneRemarksChange('${k}', this.value)">
            </div>
        `;
        container.appendChild(card);
    });
}

function onZoneNameChange(key, val) {
    if (currentZones[key]) {
        currentZones[key].name = val;
        drawCalibrationCanvas();
    }
}

function onZoneRemarksChange(key, val) {
    if (currentZones[key]) {
        currentZones[key].remarks = val;
    }
}

function onZoneTypeChange(key, val) {
    if (currentZones[key]) {
        currentZones[key].type = val;
        drawCalibrationCanvas();
    }
}

function addNewZone() {
    let index = 1;
    while (currentZones[`station_${index}`] || currentZones[`zone_${index}`]) {
        index++;
    }
    const newKey = `station_${index}`;
    
    // Position comfortably staggered across 640x360 canvas
    const xOffset = 50 + ((index * 45) % 400);
    const yOffset = 60 + ((index * 35) % 180);
    
    currentZones[newKey] = {
        name: `Station ${index.toString().padStart(2, '0')} (Custom Zone)`,
        type: "service",
        remarks: "Assigned Stylist",
        x1: xOffset,
        y1: yOffset,
        x2: Math.min(620, xOffset + 130),
        y2: Math.min(340, yOffset + 130)
    };
    
    activeZoneKey = newKey;
    renderZoneEditorList();
    drawCalibrationCanvas();
    
    // Auto-scroll list to the newly added zone
    const container = document.getElementById('zone-editor-list');
    if (container) {
        setTimeout(() => {
            container.scrollTop = container.scrollHeight;
        }, 50);
    }
}

function deleteZone(key) {
    if (confirm(`Are you sure you want to delete "${currentZones[key]?.name || key}"?`)) {
        delete currentZones[key];
        const remaining = Object.keys(currentZones);
        activeZoneKey = remaining.length > 0 ? remaining[0] : "";
        renderZoneEditorList();
        drawCalibrationCanvas();
    }
}

const canvas = document.getElementById('calib-canvas');
const ctx = canvas.getContext('2d');

function drawCalibrationCanvas() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (bgImage && bgImage.complete && bgImage.naturalWidth > 0) {
        ctx.drawImage(bgImage, 0, 0, canvas.width, canvas.height);
    } else {
        ctx.fillStyle = '#1e1b18';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
    }

    Object.keys(currentZones).forEach(k => {
        const z = currentZones[k];
        if (z && z.x1 !== undefined) {
            const isService = (z.type === 'service');
            const isActive = (k === activeZoneKey);

            ctx.fillStyle = isService ? 'rgba(214, 140, 159, 0.42)' : 'rgba(110, 140, 220, 0.42)';
            ctx.strokeStyle = isActive ? '#ffffff' : (isService ? '#d68c9f' : '#6e8cdc');
            ctx.lineWidth = isActive ? 3 : 1.5;

            const zw = z.x2 - z.x1;
            const zh = z.y2 - z.y1;
            ctx.fillRect(z.x1, z.y1, zw, zh);
            ctx.strokeRect(z.x1, z.y1, zw, zh);

            const labelText = z.name || k;
            ctx.font = 'bold 11px Plus Jakarta Sans';
            const textWidth = ctx.measureText(labelText).width;

            ctx.fillStyle = isActive ? 'rgba(45, 42, 38, 0.85)' : 'rgba(0, 0, 0, 0.65)';
            ctx.fillRect(z.x1, z.y1 - 22, textWidth + 16, 22);

            ctx.fillStyle = '#ffffff';
            ctx.fillText(labelText, z.x1 + 8, z.y1 - 7);
        }
    });
}

canvas.addEventListener('mousedown', (e) => {
    if (!activeZoneKey || !currentZones[activeZoneKey]) return;
    const rect = canvas.getBoundingClientRect();
    startX = (e.clientX - rect.left) * (canvas.width / rect.width);
    startY = (e.clientY - rect.top) * (canvas.height / rect.height);
    isDrawing = true;
});

canvas.addEventListener('mousemove', (e) => {
    if (!isDrawing) return;
    const rect = canvas.getBoundingClientRect();
    const currX = (e.clientX - rect.left) * (canvas.width / rect.width);
    const currY = (e.clientY - rect.top) * (canvas.height / rect.height);

    drawCalibrationCanvas();

    ctx.strokeStyle = '#fff';
    ctx.setLineDash([4, 4]);
    ctx.lineWidth = 2;
    ctx.strokeRect(
        Math.min(startX, currX),
        Math.min(startY, currY),
        Math.abs(currX - startX),
        Math.abs(currY - startY)
    );
    ctx.setLineDash([]);
});

canvas.addEventListener('mouseup', (e) => {
    if (!isDrawing || !activeZoneKey || !currentZones[activeZoneKey]) return;
    isDrawing = false;
    const rect = canvas.getBoundingClientRect();
    const endX = (e.clientX - rect.left) * (canvas.width / rect.width);
    const endY = (e.clientY - rect.top) * (canvas.height / rect.height);

    const x1 = Math.round(Math.min(startX, endX));
    const y1 = Math.round(Math.min(startY, endY));
    const x2 = Math.round(Math.max(startX, endX));
    const y2 = Math.round(Math.max(startY, endY));

    if (x2 - x1 > 20 && y2 - y1 > 20) {
        currentZones[activeZoneKey].x1 = x1;
        currentZones[activeZoneKey].y1 = y1;
        currentZones[activeZoneKey].x2 = x2;
        currentZones[activeZoneKey].y2 = y2;
        drawCalibrationCanvas();
    }
});

function saveCalibrationZones() {
    fetch('/api/save_zones', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({video: currentVideoName, zones: currentZones})
    })
    .then(r => r.json())
    .then(() => {
        alert("Zone configurations & remarks applied successfully! Live AI stats are now tracking your updated zones.");
        closeCalibrationModal();
    });
}

// Start polling
setInterval(updateDashboard, 500);
updateDashboard();

