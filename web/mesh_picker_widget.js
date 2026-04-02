/**
 * SegviGen Mesh Picker Widget
 *
 * Adds an "Open 3D Picker" button to the SegviGenMeshPicker node.
 * When clicked it opens picker.html in a popup window; the user clicks
 * on the mesh surface and confirms, the voxel coordinates are sent back
 * via postMessage and stored in the node's picked_points_json widget.
 */

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// Reference to the currently open picker popup.
let pickerWin = null;

// ── Open the picker popup ─────────────────────────────────────────────────────
function openPicker(node) {
    const filename   = node._sgMeshFile   || '';
    const meshType   = node._sgMeshType   || 'output';
    const resolution = node._sgResolution || 64;

    if (!filename) {
        alert(
            "SegviGen Picker: no mesh available yet.\n\n" +
            "Queue the workflow once first so the node registers the mesh file, " +
            "then click the button again."
        );
        return;
    }

    const qs = new URLSearchParams({
        mesh:       filename,
        meshType:   meshType,
        resolution: String(resolution),
        nodeId:     String(node.id),
        subfolder:  '',
    }).toString();

    const url = `/extensions/ComfyUI-SegviGen/picker.html?${qs}`;

    if (pickerWin && !pickerWin.closed) {
        pickerWin.location.href = url;
        pickerWin.focus();
    } else {
        pickerWin = window.open(
            url,
            'segvigen_picker',
            'width=920,height=660,resizable=yes,menubar=no,toolbar=no,location=no'
        );
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/** Force a canvas redraw, compatible with different ComfyUI versions. */
function dirtyCanvas() {
    try { app.canvas?.setDirty(true, true); } catch (_) {}
    try { app.graph?.setDirtyCanvas(true, true); } catch (_) {}
}

/** Update the count label widget using a stored direct reference. */
function refreshCountLabel(node, count) {
    // Prefer the direct reference stored at node creation time.
    const label = node._sgCountWidget
        ?? node.widgets?.find(w => w.name?.includes('Points stored'));
    if (label) {
        label.name = `● Points stored: ${count}`;
        dirtyCanvas();
    }
}

/** Write points to both the LiteGraph widget value AND the backing DOM element. */
function writePoints(node, points) {
    const jsonStr = JSON.stringify(points);
    const widget = node.widgets?.find(w => w.name === 'picked_points_json');
    if (!widget) {
        console.warn('[SegviGen] picked_points_json widget not found on node', node.id);
        return;
    }
    widget.value = jsonStr;
    // Keep the DOM input in sync so ComfyUI picks up the new value on queue.
    if (widget.inputEl) {
        widget.inputEl.value = jsonStr;
    }
    refreshCountLabel(node, points.length);
    dirtyCanvas();
    console.log(`[SegviGen] ${points.length} point(s) written to node ${node.id}`);
}

// ── Receive confirmed points from the popup ────────────────────────────────────

function handlePickerMessage(data) {
    // data is the WS message detail: {nodeId, points} — no type field here,
    // the event type is already validated by the api.addEventListener call.
    const { nodeId, points } = data;
    console.log('[SegviGen] picker message received, nodeId=', nodeId, 'points=', points);

    if (!Array.isArray(points)) {
        console.warn('[SegviGen] picker: points is not an array');
        return;
    }

    const node = app.graph?.getNodeById(parseInt(nodeId, 10));
    if (!node) {
        console.warn('[SegviGen] picker: node not found for id', nodeId);
        return;
    }

    writePoints(node, points);
}

// Listen on ComfyUI's own WebSocket — works in all environments including
// Electron/Desktop ComfyUI where window.opener and BroadcastChannel are blocked.
api.addEventListener('segvigen.points_confirmed', ({ detail }) => {
    handlePickerMessage(detail);
});

// ── Register ComfyUI extension ────────────────────────────────────────────────
app.registerExtension({
    name: 'SegviGen.MeshPicker',

    nodeCreated(node) {
        if (node.comfyClass !== 'SegviGenMeshPicker') return;

        // Persistent state on the node instance.
        node._sgMeshFile   = '';
        node._sgMeshType   = 'output';
        node._sgResolution = 512;  // canonical 512-space

        // ── Intercept onExecuted to capture mesh info from Python output ──────
        const _origOnExecuted = node.onExecuted;
        node.onExecuted = function (output) {
            if (_origOnExecuted) _origOnExecuted.call(this, output);

            const file = output?.mesh_filename?.[0];
            if (file) this._sgMeshFile = file;

            const mtype = output?.mesh_type?.[0];
            if (mtype) this._sgMeshType = mtype;

            const res = output?.voxel_resolution?.[0];
            if (res) this._sgResolution = res;

            // Sync count from server-side parse (authoritative after a run).
            const count = output?.picked_count?.[0] ?? 0;
            refreshCountLabel(this, count);
        };

        // ── Style the raw JSON widget ─────────────────────────────────────────
        const jsonWidget = node.widgets?.find(w => w.name === 'picked_points_json');
        if (jsonWidget?.inputEl) {
            jsonWidget.inputEl.style.opacity    = '0.5';
            jsonWidget.inputEl.style.fontSize   = '10px';
            jsonWidget.inputEl.style.fontFamily = 'monospace';
            // Refresh count when JSON is edited manually.
            jsonWidget.inputEl.addEventListener('input', () => {
                try {
                    const pts = JSON.parse(jsonWidget.inputEl.value || '[]');
                    refreshCountLabel(node, Array.isArray(pts) ? pts.length : 0);
                } catch (_) {}
            });
        }

        // ── Derive initial count from current widget value ─────────────────────
        let initialCount = 0;
        if (jsonWidget) {
            try {
                const v = JSON.parse(jsonWidget.value || '[]');
                initialCount = Array.isArray(v) ? v.length : 0;
            } catch (_) {}
        }

        // ── Add count label widget (stored by direct reference) ───────────────
        const countBtn = node.addWidget(
            'button',
            `● Points stored: ${initialCount}`,
            null,
            () => {}   // no-op click
        );
        node._sgCountWidget = countBtn;   // direct reference avoids name-search

        // ── Add the main action button ────────────────────────────────────────
        node.addWidget(
            'button',
            '🎯 Open 3D Picker',
            null,
            () => openPicker(node)
        );
    },
});
