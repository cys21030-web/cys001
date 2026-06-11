/* Shared JavaScript Utilities */

/**
 * Format bytes for display
 */
function formatBytes(bytes, decimals = 2) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const dm = decimals < 0 ? 0 : decimals;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
}

/**
 * Show notification
 */
function showNotification(message, type = 'info', duration = 3000) {
    // Create notification element
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.textContent = message;
    notification.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        background: ${getNotificationColor(type)};
        color: white;
        padding: 15px 20px;
        border-radius: 4px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
        z-index: 1000;
        animation: slideIn 0.3s ease;
    `;
    
    document.body.appendChild(notification);
    
    setTimeout(() => {
        notification.style.animation = 'slideOut 0.3s ease';
        setTimeout(() => notification.remove(), 300);
    }, duration);
}

function getNotificationColor(type) {
    const colors = {
        'success': '#28a745',
        'error': '#dc3545',
        'warning': '#ffc107',
        'info': '#0066cc'
    };
    return colors[type] || colors['info'];
}

/**
 * Make API call with error handling
 */
async function apiCall(endpoint, options = {}) {
    try {
        const defaultOptions = {
            headers: {
                'Content-Type': 'application/json'
            }
        };
        
        const response = await fetch(endpoint, { ...defaultOptions, ...options });
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }
        
        return await response.json();
    } catch (error) {
        console.error('API Error:', error);
        throw error;
    }
}

/**
 * Draw 8x8 heat map
 */
function drawHeatMap(data, containerId) {
    const container = document.getElementById(containerId);
    if (!container) return;
    
    container.innerHTML = '';
    
    // Normalize data to 0-1
    const min = Math.min(...data);
    const max = Math.max(...data);
    const normalized = data.map(v => (v - min) / (max - min));
    
    // Create cells
    for (let i = 0; i < 64; i++) {
        const cell = document.createElement('div');
        cell.className = 'heat-cell';
        
        // Heatmap color (blue to red)
        const value = normalized[i];
        let r, g, b;
        
        if (value < 0.5) {
            // Blue to cyan
            r = 0;
            g = Math.floor(255 * (value / 0.5));
            b = 255;
        } else {
            // Cyan to red
            r = Math.floor(255 * ((value - 0.5) / 0.5));
            g = 255 - Math.floor(255 * ((value - 0.5) / 0.5));
            b = 255 - Math.floor(255 * ((value - 0.5) / 0.5));
        }
        
        cell.style.backgroundColor = `rgb(${r}, ${g}, ${b})`;
        cell.textContent = Math.round(data[i]);
        container.appendChild(cell);
    }
}

/**
 * Draw 3D point cloud on canvas
 */
function drawPointCloud(points, canvasId) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    const w = canvas.width;
    const h = canvas.height;
    
    // Clear canvas
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, w, h);
    
    // Draw grid
    ctx.strokeStyle = '#ddd';
    ctx.beginPath();
    ctx.moveTo(0, h / 2);
    ctx.lineTo(w, h / 2);
    ctx.moveTo(w / 2, 0);
    ctx.lineTo(w / 2, h);
    ctx.stroke();
    
    // Find bounds
    const xs = points.map(p => p.x);
    const ys = points.map(p => p.y);
    const xMin = Math.min(...xs);
    const xMax = Math.max(...xs);
    const yMin = Math.min(...ys);
    const yMax = Math.max(...ys);
    
    const xRange = xMax - xMin || 1;
    const yRange = yMax - yMin || 1;
    
    // Draw points
    ctx.fillStyle = '#0066cc';
    ctx.globalAlpha = 0.7;
    
    points.forEach(p => {
        const px = ((p.x - xMin) / xRange) * (w - 20) + 10;
        const py = ((p.y - yMin) / yRange) * (h - 20) + 10;
        
        ctx.fillRect(px - 2, py - 2, 4, 4);
    });
    
    ctx.globalAlpha = 1;
}

/**
 * Add animation styles
 */
const style = document.createElement('style');
style.textContent = `
    @keyframes slideIn {
        from {
            transform: translateX(400px);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
    
    @keyframes slideOut {
        from {
            transform: translateX(0);
            opacity: 1;
        }
        to {
            transform: translateX(400px);
            opacity: 0;
        }
    }
`;
document.head.appendChild(style);
