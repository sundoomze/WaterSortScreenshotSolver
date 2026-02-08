// vision.worker.js (classic worker)
// OpenCV.js parsing logic runs here to avoid blocking the UI thread.
//
// Loads OpenCV.js from docs.opencv.org via importScripts.
//
// Messages:
//  - {type:'analyze', width, height, rgba:ArrayBuffer, capacity, dbscanEps, cvVer }
// Replies:
//  - {type:'result', result, debug }
//  - {type:'error', error }

'use strict';

let cvPromise = null;
let cvVersionLoaded = null;
let currentRequestId = null;
let isBusy = false;

function log(message) {
  self.postMessage({ type: 'log', requestId: currentRequestId, message });
}


function isThenable(x) {
  return !!x && (typeof x === 'object' || typeof x === 'function') && typeof x.then === 'function';
}

/**
 * Some OpenCV.js builds expose a "thenable" object that later gains cv.Mat etc,
 * but it *still* has a .then property. If we resolve a Promise with that value,
 * the Promise resolution procedure will treat it as a thenable and call .then
 * again, which can deadlock forever.
 *
 * To prevent that, once we detect a "ready" cv object (cv.Mat exists), we wrap
 * it in a Proxy that hides the `then` property. This turns it into a plain
 * value for Promise resolution and for `await`.
 */
function makeCvNonThenable(cvObj) {
  if (!cvObj) return cvObj;
  if (!cvObj.Mat) return cvObj;           // only wrap once it looks ready
  if (!isThenable(cvObj)) return cvObj;   // already safe

  return new Proxy(cvObj, {
    get(target, prop, receiver) {
      if (prop === 'then') return undefined;
      return Reflect.get(target, prop, receiver);
    },
    has(target, prop) {
      if (prop === 'then') return false;
      return Reflect.has(target, prop);
    },
  });
}

function loadOpenCv(cvVer) {
  // NOTE:
  // In a Web Worker, Emscripten-derived builds typically set `scriptDirectory`
  // to the WORKER script URL (self.location.href), not to the imported opencv.js
  // URL. If we use `scriptDirectory + path`, OpenCV will try to fetch
  // `opencv_js.wasm` from our GitHub Pages site and hang/fail.
  //
  // So we *force* the wasm path to docs.opencv.org by ignoring `scriptDirectory`.

  if (cvPromise) return cvPromise;

  cvVersionLoaded = cvVer;

  const base = (cvVer === '4.x') ? 'https://docs.opencv.org/4.x/' : `https://docs.opencv.org/${cvVer}/`;
  const scriptUrl = base + 'opencv.js';

  cvPromise = new Promise((resolve, reject) => {
    let settled = false;
    const finish = (ok, val) => {
      if (settled) return;
      settled = true;
      try { clearTimeout(timeout); } catch (_) {}
      if (ok) {
        const safe = makeCvNonThenable(val);
        try { self.cv = safe; } catch (_) {}
        resolve(safe);
      } else {
        reject(val);
      }
    };

    const timeout = setTimeout(() => {
      finish(false, new Error('OpenCV load timeout. Check DevTools → Network for opencv_js.wasm 404/CORS issues.'));
    }, 25000);

    try {
      // Helper that resolves cv for BOTH OpenCV.js styles:
      //   1) legacy: `cv` is an object and you wait for Module.onRuntimeInitialized
      //   2) newer:  `cv` is a Promise/thenable that resolves to the cv object
      const resolveCvIfReady = () => {
        const cvAny = self.cv;
        if (!cvAny) return false;

        // Already-ready object style
        if (cvAny.Mat) {
          finish(true, cvAny);
          return true;
        }

        // Promise / thenable style
        if (typeof cvAny.then === 'function') {
          // IMPORTANT: cvAny may be a non-compliant thenable whose .then()
          // return value is not a Promise (so `.catch` can explode). Always
          // normalize via Promise.resolve.
          Promise.resolve(cvAny)
            .then((cvObj) => {
              if (cvObj && cvObj.Mat) finish(true, cvObj);
              else finish(false, new Error('OpenCV promise resolved, but cv.Mat was missing.'));
            })
            .catch((err) => finish(false, err));
          return true;
        }

        return false;
      };

      self.Module = {
        locateFile: (path /*, scriptDirectory */) => base + path,
        onRuntimeInitialized: () => {
          // Some OpenCV.js builds *never* call Module.onRuntimeInitialized because
          // they expose `cv` as a Promise/thenable instead. We still keep this for
          // compatibility, but always try to resolve cv in a version-agnostic way.
          if (!resolveCvIfReady()) {
            finish(false, new Error('OpenCV runtime initialized but `cv` was not found.'));
          }
        },
        onAbort: (what) => finish(false, new Error('OpenCV aborted: ' + what)),
      };

      importScripts(scriptUrl);

      // Some builds make cv usable immediately after importScripts, OR
      // expose cv as a Promise/thenable right away. Handle both.
      resolveCvIfReady();

      // Legacy fallback: some older builds use cv.onRuntimeInitialized.
      // If cv exists but isn't ready/promise-y, install callback.
      if (self.cv && typeof self.cv === 'object' && typeof self.cv.then !== 'function' && !self.cv.Mat) {
        self.cv.onRuntimeInitialized = () => {
          if (!resolveCvIfReady()) finish(true, self.cv);
        };
      }
    } catch (err) {
      finish(false, err);
    }
  });

  // Allow retries if loading fails
  cvPromise.catch(() => {
    cvPromise = null;
  });

  return cvPromise;
}

function nowMs() {
  // performance.now() is available in modern workers; fall back to Date.now().
  return (self.performance && typeof self.performance.now === 'function') ? self.performance.now() : Date.now();
}

function clampInt(v, lo, hi) {
  return Math.max(lo, Math.min(hi, v | 0));
}

function maybeDownscaleForDetection(cv, bgr, maxDim) {
  // OpenCV.js in WASM is *much* slower than native OpenCV on large frames.
  // Bottles are big, so we can detect them on a downscaled frame and then
  // scale boxes back up for sampling on the full-res image.
  const h = bgr.rows;
  const w = bgr.cols;
  const m = Math.max(w, h);
  if (m <= maxDim) {
    return { mat: bgr, scale: 1.0, owns: false };
  }

  const scale = maxDim / m;
  const newW = Math.max(1, Math.round(w * scale));
  const newH = Math.max(1, Math.round(h * scale));

  const dst = new cv.Mat();
  cv.resize(bgr, dst, new cv.Size(newW, newH), 0, 0, cv.INTER_AREA);
  return { mat: dst, scale, owns: true };
}

function scaleBoxesToOriginal(boxesSmall, invScale, origW, origH) {
  // Scale a list of {x,y,w,h} boxes back to the original image coords.
  // Expand slightly to compensate for rounding.
  const out = [];
  for (const b of boxesSmall) {
    let x = Math.round(b.x * invScale);
    let y = Math.round(b.y * invScale);
    let w = Math.round(b.w * invScale);
    let h = Math.round(b.h * invScale);

    const pad = Math.max(2, Math.round(Math.min(w, h) * 0.03));
    x -= pad;
    y -= pad;
    w += pad * 2;
    h += pad * 2;

    x = clampInt(x, 0, origW - 1);
    y = clampInt(y, 0, origH - 1);
    w = clampInt(w, 1, origW - x);
    h = clampInt(h, 1, origH - y);

    out.push({ x, y, w, h });
  }
  return out;
}

function bgrToHex(b, g, r) {
  const rr = Math.max(0, Math.min(255, Math.round(r)));
  const gg = Math.max(0, Math.min(255, Math.round(g)));
  const bb = Math.max(0, Math.min(255, Math.round(b)));
  return '#' + rr.toString(16).padStart(2, '0').toUpperCase()
           + gg.toString(16).padStart(2, '0').toUpperCase()
           + bb.toString(16).padStart(2, '0').toUpperCase();
}

function iou(a, b) {
  const ax1 = a.x, ay1 = a.y, ax2 = a.x + a.w, ay2 = a.y + a.h;
  const bx1 = b.x, by1 = b.y, bx2 = b.x + b.w, by2 = b.y + b.h;
  const ix1 = Math.max(ax1, bx1), iy1 = Math.max(ay1, by1);
  const ix2 = Math.min(ax2, bx2), iy2 = Math.min(ay2, by2);
  if (ix2 <= ix1 || iy2 <= iy1) return 0.0;
  const inter = (ix2 - ix1) * (iy2 - iy1);
  const union = a.w * a.h + b.w * b.h - inter;
  return union <= 0 ? 0.0 : inter / union;
}

function clusterRowsByY(boxes, eps) {
  // Very small helper to cluster bottle boxes into rows by y-center proximity.
  const rows = [];
  const sorted = boxes.slice().sort((a, b) => (a.y + a.h / 2) - (b.y + b.h / 2));
  for (const b of sorted) {
    const cy = b.y + b.h / 2;
    let best = null;
    for (const row of rows) {
      if (Math.abs(cy - row.cy) <= eps) { best = row; break; }
    }
    if (!best) {
      rows.push({ cy, boxes: [b] });
    } else {
      best.boxes.push(b);
      best.cy = best.boxes.reduce((s, x) => s + (x.y + x.h / 2), 0) / best.boxes.length;
    }
  }
  // sort each row left->right, and rows top->bottom
  rows.sort((a, b) => a.cy - b.cy);
  for (const row of rows) row.boxes.sort((a, b) => a.x - b.x);
  return rows;
}

function _rowStepBytes(mat) {
  // OpenCV.js exposes step in a few shapes. Prefer step[0] when present.
  const s = mat.step;
  if (typeof s === 'number') return s;
  if (s && typeof s[0] === 'number') return s[0];
  if (s && s.length) return s[0];
  // fallback
  return mat.cols * mat.channels();
}

function medianOfChannelMat(mat, channelIndex) {
  // ROI mats in OpenCV.js are often not contiguous. Using ucharPtr(y,x) ensures we
  // read the correct pixel values regardless of row stride.
  const rows = mat.rows;
  const cols = mat.cols;
  const vals = new Array(rows * cols);
  let k = 0;
  for (let y = 0; y < rows; y++) {
    for (let x = 0; x < cols; x++) {
      const px = mat.ucharPtr(y, x);
      vals[k++] = px[channelIndex] ?? 0;
    }
  }
  vals.sort((a, b) => a - b);
  return vals[(vals.length / 2) | 0] ?? 0;
}

function medianBgr(mat) {
  const b = medianOfChannelMat(mat, 0);
  const g = medianOfChannelMat(mat, 1);
  const r = medianOfChannelMat(mat, 2);
  return [b, g, r];
}

function medianHsv(mat) {
  const h = medianOfChannelMat(mat, 0);
  const s = medianOfChannelMat(mat, 1);
  const v = medianOfChannelMat(mat, 2);
  return [h, s, v];
}

function isFilledSlot(hsv) {
  const h = hsv[0], s = hsv[1], v = hsv[2];

  // Near-white / near-grey pixels are usually glass highlights or the rock pile,
  // not actual liquid. This prevents the rock bottle from becoming a fake
  // "very light yellow" color.
  //
  // (We keep this conservative to avoid dropping legitimate pastel colors.)
  if (v >= 175 && s <= 55) return false;

  if (v >= 130 && s >= 40) return true;
  if (s >= 150 && v >= 80) return true;
  return false;
}

function detectBottles(cv, bgr, hsvLower, hsvUpper, opts) {
  const minArea = opts?.minArea ?? 40000;
  const aspectMin = opts?.aspectMin ?? 2.0;
  const aspectMax = opts?.aspectMax ?? 4.5;
  const nmsIou = opts?.nmsIou ?? 0.30;

  const hsv = new cv.Mat();
  cv.cvtColor(bgr, hsv, cv.COLOR_BGR2HSV);

  const low = new cv.Mat(hsv.rows, hsv.cols, hsv.type(), [hsvLower[0], hsvLower[1], hsvLower[2], 0]);
  const high = new cv.Mat(hsv.rows, hsv.cols, hsv.type(), [hsvUpper[0], hsvUpper[1], hsvUpper[2], 255]);
  const mask = new cv.Mat();
  cv.inRange(hsv, low, high, mask);

  const k3 = cv.Mat.ones(3, 3, cv.CV_8U);
  const k5 = cv.Mat.ones(5, 5, cv.CV_8U);
  cv.morphologyEx(mask, mask, cv.MORPH_OPEN, k3);
  cv.dilate(mask, mask, k5);

  const contours = new cv.MatVector();
  const hierarchy = new cv.Mat();
  cv.findContours(mask, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);

  const candidates = [];
  for (let i = 0; i < contours.size(); i++) {
    const cnt = contours.get(i);
    const rect = cv.boundingRect(cnt);
    const x = rect.x, y = rect.y, w = rect.width, h = rect.height;
    const area = w * h;
    const aspect = h / (w + 1e-6);

    if (area < minArea) { cnt.delete(); continue; }
    if (aspect < aspectMin || aspect > aspectMax) { cnt.delete(); continue; }

    candidates.push({ x, y, w, h, area });
    cnt.delete();
  }

  // NMS
  candidates.sort((a, b) => b.area - a.area);
  const selected = [];
  for (const c of candidates) {
    let ok = true;
    for (const s of selected) {
      if (iou(c, s) >= nmsIou) { ok = false; break; }
    }
    if (ok) selected.push(c);
  }

  // cluster rows and sort
  const medH = (() => {
    if (selected.length === 0) return 100;
    const hs = selected.map((b) => b.h).sort((a, b) => a - b);
    return hs[Math.floor(hs.length / 2)];
  })();

  const rowEps = Math.max(40.0, medH * 0.60);
  const rows = clusterRowsByY(selected, rowEps);
  const ordered = [];
  for (const row of rows) ordered.push(...row.boxes);

  // cleanup
  hsv.delete(); low.delete(); high.delete(); mask.delete(); k3.delete(); k5.delete(); contours.delete(); hierarchy.delete();

  return { boxes: ordered, rows: rows.map((r) => r.boxes.map((b) => ordered.indexOf(b))) };
}

function clamp(v, lo, hi) {
  return Math.max(lo, Math.min(hi, v));
}

function medianNumber(arr) {
  if (!arr || arr.length === 0) return 0;
  const a = arr.slice().sort((x, y) => x - y);
  const mid = Math.floor(a.length / 2);
  return (a.length % 2) ? a[mid] : (a[mid - 1] + a[mid]) / 2;
}

function kmeans1D(values, k, iters = 10) {
  // Returns `k` centers (sorted). Assumes values.length >= k.
  const n = values.length;
  const sorted = values.slice().sort((a, b) => a - b);
  const centers = [];
  for (let i = 0; i < k; i++) {
    const q = (i + 0.5) / k;
    centers.push(sorted[Math.floor(q * (n - 1))]);
  }

  for (let t = 0; t < iters; t++) {
    const groups = Array.from({ length: k }, () => []);
    for (const v of values) {
      let best = 0;
      let bestDist = Infinity;
      for (let i = 0; i < k; i++) {
        const d = Math.abs(v - centers[i]);
        if (d < bestDist) { bestDist = d; best = i; }
      }
      groups[best].push(v);
    }

    for (let i = 0; i < k; i++) {
      if (groups[i].length) {
        centers[i] = groups[i].reduce((s, v) => s + v, 0) / groups[i].length;
      }
    }
  }

  centers.sort((a, b) => a - b);
  return centers;
}

function nmsBoxes(candidates, nmsIou = 0.30) {
  const boxes = (candidates || []).map((b) => ({ ...b, area: b.area ?? (b.w * b.h) }));
  boxes.sort((a, b) => (b.area ?? 0) - (a.area ?? 0));

  const selected = [];
  for (const c of boxes) {
    let ok = true;
    for (const s of selected) {
      if (iou(c, s) >= nmsIou) { ok = false; break; }
    }
    if (ok) selected.push(c);
  }
  return selected;
}

function inferBottleFromMouthOutline(cv, roiBgr, offX, offY, fullW, fullH, wMed, hMed) {
  // Fallback detector: look for the blue outline around the bottle mouth.
  // This helps when the bottle body outline is too faint to contour-detect
  // (often the case for empty bottles), but the mouth/neck outline is still visible.
  try {
    const hsv = new cv.Mat();
    cv.cvtColor(roiBgr, hsv, cv.COLOR_BGR2HSV);

    const low = new cv.Mat(hsv.rows, hsv.cols, hsv.type(), [85, 8, 55, 0]);
    const high = new cv.Mat(hsv.rows, hsv.cols, hsv.type(), [140, 255, 255, 255]);
    const mask = new cv.Mat();
    cv.inRange(hsv, low, high, mask);

    const k3 = cv.Mat.ones(3, 3, cv.CV_8U);
    cv.morphologyEx(mask, mask, cv.MORPH_OPEN, k3);
    cv.dilate(mask, mask, k3);

    const topH = Math.max(1, Math.round(mask.rows * 0.38));
    const topMask = mask.roi(new cv.Rect(0, 0, mask.cols, topH));

    const contours = new cv.MatVector();
    const hierarchy = new cv.Mat();
    cv.findContours(topMask, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);

    let bestRect = null;
    let bestArea = 0;

    for (let i = 0; i < contours.size(); i++) {
      const cnt = contours.get(i);
      const r = cv.boundingRect(cnt);
      const rw = r.width, rh = r.height;
      const area = rw * rh;
      const ar = rw / (rh + 1e-6);

      // mouth/neck outline is a short, wide-ish shape near the top
      if (area < 60) { cnt.delete(); continue; }
      if (ar < 1.4) { cnt.delete(); continue; }

      // Width should be a fraction of the bottle width.
      if (rw < wMed * 0.30 || rw > wMed * 0.95) { cnt.delete(); continue; }
      // Height should be relatively small.
      if (rh < hMed * 0.02 || rh > hMed * 0.22) { cnt.delete(); continue; }

      if (area > bestArea) {
        bestArea = area;
        bestRect = r;
      }
      cnt.delete();
    }

    // cleanup
    contours.delete();
    hierarchy.delete();
    topMask.delete();
    hsv.delete();
    low.delete();
    high.delete();
    mask.delete();
    k3.delete();

    if (!bestRect) return null;

    const mouthCx = bestRect.x + bestRect.width / 2;
    const mouthCy = bestRect.y + bestRect.height / 2;

    const absCx = offX + mouthCx;
    const absCy = offY + mouthCy;

    // Mouth center is near the top of the bottle; place a median-sized bottle box under it.
    const bx = clampInt(Math.round(absCx - wMed / 2), 0, fullW - 1);
    const by = clampInt(Math.round(absCy - hMed * 0.10), 0, fullH - 1);
    const bw = clampInt(Math.round(wMed), 1, fullW - bx);
    const bh = clampInt(Math.round(hMed), 1, fullH - by);
    return { x: bx, y: by, w: bw, h: bh, area: bw * bh };
  } catch (_err) {
    return null;
  }
}

function detectBottleNear(cv, bgr, cx, cy, wMed, hMed) {
  const W = bgr.cols, H = bgr.rows;
  const winW = wMed * 1.9;
  const winH = hMed * 1.9;

  const x0 = clampInt(Math.round(cx - winW / 2), 0, W - 1);
  const y0 = clampInt(Math.round(cy - winH / 2), 0, H - 1);
  const w0 = clampInt(Math.round(winW), 1, W - x0);
  const h0 = clampInt(Math.round(winH), 1, H - y0);

  const roi = bgr.roi(new cv.Rect(x0, y0, w0, h0));
  // Dark + bright outline pass (the fixed layout lets us be aggressive here).
  const lower = [85, 5, 35];
  const upper = [140, 255, 255];

  const minArea = Math.max(2000, (wMed * hMed) * 0.20);

  const { boxes } = detectBottles(cv, roi, lower, upper, {
    minArea,
    aspectMin: 2.0,
    aspectMax: 6.5,
    nmsIou: 0.30,
  });

  // If we failed to detect a full bottle contour, try the bottle-mouth fallback
  // before giving up. This improves reliability on empty bottles.
  if (!boxes.length) {
    const mouthBox = inferBottleFromMouthOutline(cv, roi, x0, y0, W, H, wMed, hMed);
    roi.delete();
    return mouthBox;
  }

  roi.delete();

  let best = null;
  let bestScore = Infinity;
  for (const b of boxes) {
    const bx = b.x + x0;
    const by = b.y + y0;
    const bcX = bx + b.w / 2;
    const bcY = by + b.h / 2;
    const dist = Math.hypot(bcX - cx, bcY - cy);

    const sizePenalty = Math.abs((b.w * b.h) - (wMed * hMed)) / (wMed * hMed + 1e-6);
    const score = dist + sizePenalty * (wMed * 0.35);

    if (score < bestScore) {
      bestScore = score;
      best = { x: bx, y: by, w: b.w, h: b.h, area: b.w * b.h };
    }
  }

  return best;
}

function enforceFixedBottleGrid(cv, bgr, candidates) {
  // This game layout never changes: 3 bottles on the top row + 3 on the bottom row.
  const W = bgr.cols, H = bgr.rows;
  const EXPECTED = 6;

  let boxes = (candidates || []).slice();

  // Filter out obvious junk (curtains / bottom UI) by position.
  const filtered = boxes.filter((b) => {
    const cx = b.x + b.w / 2;
    const cy = b.y + b.h / 2;
    return (cx >= W * 0.20 && cx <= W * 0.80 && cy >= H * 0.14 && cy <= H * 0.85);
  });
  if (filtered.length) boxes = filtered;

  // If we got a bunch of candidates (e.g., stray UI outlines), aggressively
  // keep only those that look bottle-sized. This prevents false positives
  // (curtains, banners, etc) from skewing the grid inference.
  if (boxes.length > EXPECTED) {
    const areas = boxes.map((b) => b.w * b.h);
    const medA = medianNumber(areas);
    const loA = medA * 0.45;
    const hiA = medA * 2.25;
    const filteredByArea = boxes.filter((b) => {
      const a = b.w * b.h;
      return a >= loA && a <= hiA;
    });
    if (filteredByArea.length >= 3) boxes = filteredByArea;

    boxes.sort((a, b) => (b.w * b.h) - (a.w * a.h));
    if (boxes.length > EXPECTED * 2) boxes = boxes.slice(0, EXPECTED * 2);
  }

  const wMed = boxes.length ? medianNumber(boxes.map((b) => b.w)) : (W * 0.11);
  const hMed = boxes.length ? medianNumber(boxes.map((b) => b.h)) : (H * 0.32);

  // Estimate spacing
  const xs = boxes.map((b) => b.x + b.w / 2);
  const ys = boxes.map((b) => b.y + b.h / 2);

  let dx = wMed * 1.60;
  if (xs.length >= 2) {
    const sx = xs.slice().sort((a, b) => a - b);
    const diffs = [];
    for (let i = 1; i < sx.length; i++) {
      const d = sx[i] - sx[i - 1];
      if (d >= wMed * 0.6) diffs.push(d);
    }
    if (diffs.length) dx = medianNumber(diffs);
  }

  let dy = Math.max(hMed * 1.25, H * 0.25);
  if (ys.length >= 2) {
    const centers = kmeans1D(ys, 2, 8);
    const sep = Math.abs(centers[1] - centers[0]);
    if (sep >= hMed * 0.8) dy = sep;
  }

  // X centers (3 cols)
  let xCenters;
  if (xs.length >= 3) {
    xCenters = kmeans1D(xs, 3, 10);
  } else if (xs.length === 2) {
    const sx = xs.slice().sort((a, b) => a - b);
    const x1 = sx[0], x2 = sx[1];
    const diff = x2 - x1;

    if (diff > dx * 1.6) {
      const step = diff / 2;
      xCenters = [x1, x1 + step, x1 + 2 * step];
      dx = step;
    } else {
      // adjacent
      if (x1 < W / 2) xCenters = [x1, x2, x2 + diff];
      else xCenters = [x1 - diff, x1, x2];
      dx = diff;
    }
  } else if (xs.length === 1) {
    const x = xs[0];
    const col = (x < W * 0.45) ? 0 : (x > W * 0.55) ? 2 : 1;
    if (col === 0) xCenters = [x, x + dx, x + 2 * dx];
    else if (col === 1) xCenters = [x - dx, x, x + dx];
    else xCenters = [x - 2 * dx, x - dx, x];
  } else {
    xCenters = [W * 0.38, W * 0.50, W * 0.62];
    dx = (xCenters[1] - xCenters[0]);
  }
  xCenters = xCenters.map((v) => clamp(v, wMed * 0.6, W - wMed * 0.6)).sort((a, b) => a - b);

  // Y centers (2 rows)
  let yCenters;
  if (ys.length >= 2) {
    const centers = kmeans1D(ys, 2, 10);
    const sep = Math.abs(centers[1] - centers[0]);
    if (sep < hMed * 0.8) {
      // Probably only one row detected; infer the other.
      const y0 = centers[0];
      if (y0 < H / 2) yCenters = [y0, y0 + dy];
      else yCenters = [y0 - dy, y0];
    } else {
      yCenters = centers;
    }
  } else if (ys.length === 1) {
    const y = ys[0];
    const row = (y < H / 2) ? 0 : 1;
    if (row === 0) yCenters = [y, y + dy];
    else yCenters = [y - dy, y];
  } else {
    yCenters = [H * 0.33, H * 0.65];
    dy = yCenters[1] - yCenters[0];
  }
  yCenters = yCenters.map((v) => clamp(v, hMed * 0.6, H - hMed * 0.6)).sort((a, b) => a - b);

  // Grid predictions (top row L->R, then bottom row L->R)
  const preds = [];
  for (let r = 0; r < 2; r++) {
    for (let c = 0; c < 3; c++) preds.push({ r, c, cx: xCenters[c], cy: yCenters[r] });
  }

  // Assign candidate boxes to predictions (greedy by distance)
  const assign = new Array(EXPECTED).fill(null);
  const used = new Set();
  const pairs = [];

  for (let bi = 0; bi < boxes.length; bi++) {
    const b = boxes[bi];
    const bcX = b.x + b.w / 2;
    const bcY = b.y + b.h / 2;

    for (let pi = 0; pi < preds.length; pi++) {
      const p = preds[pi];
      const dist = Math.hypot(bcX - p.cx, bcY - p.cy);

      // Only consider reasonably close assignments
      if (dist <= Math.max(wMed, hMed) * 1.25) {
        pairs.push({ dist, bi, pi });
      }
    }
  }

  pairs.sort((a, b) => a.dist - b.dist);
  for (const pr of pairs) {
    if (assign[pr.pi]) continue;
    if (used.has(pr.bi)) continue;
    assign[pr.pi] = boxes[pr.bi];
    used.add(pr.bi);
  }

  // For missing predictions, run a local search around the predicted center.
  for (let pi = 0; pi < assign.length; pi++) {
    if (assign[pi]) continue;
    const p = preds[pi];
    const found = detectBottleNear(cv, bgr, p.cx, p.cy, wMed, hMed);
    if (found) assign[pi] = found;
  }

  // Still missing? Fall back to a synthetic box with median size.
  for (let pi = 0; pi < assign.length; pi++) {
    if (assign[pi]) continue;
    const p = preds[pi];
    assign[pi] = {
      x: clampInt(Math.round(p.cx - wMed / 2), 0, W - 1),
      y: clampInt(Math.round(p.cy - hMed / 2), 0, H - 1),
      w: clampInt(Math.round(wMed), 1, W),
      h: clampInt(Math.round(hMed), 1, H),
      area: wMed * hMed,
    };
  }

  // Ensure stable ordering / integer coords
  return assign.map((b) => ({
    x: clampInt(Math.round(b.x), 0, W - 1),
    y: clampInt(Math.round(b.y), 0, H - 1),
    w: clampInt(Math.round(b.w), 1, W),
    h: clampInt(Math.round(b.h), 1, H),
  }));
}

function detectBottlesRobust(cv, bgr) {
  // Multi-pass outline thresholding + fixed-grid normalization.
  const W = bgr.cols, H = bgr.rows;

  // Coarse ROI (exclude the side curtains + bottom UI). If the crop is too aggressive for a
  // particular screenshot, the fixed-grid step below will still recover missing bottles.
  const rx = clampInt(Math.round(W * 0.14), 0, W - 1);
  const ry = clampInt(Math.round(H * 0.12), 0, H - 1);
  const rw = clampInt(Math.round(W * 0.72), 1, W - rx);
  const rh = clampInt(Math.round(H * 0.73), 1, H - ry);

  const roiMat = bgr.roi(new cv.Rect(rx, ry, rw, rh));

  const det = maybeDownscaleForDetection(cv, roiMat, 900);
  const detScale = det.scale;
  const detInv = 1.0 / detScale;

  const area = det.mat.cols * det.mat.rows;
  const minAreaScaled = Math.max(8000, area * 0.008);

  const passes = [
    { lower: [85, 18, 80], upper: [140, 255, 255] }, // bright outline
    { lower: [85, 5, 40], upper: [140, 255, 255] },  // dark outline (empty bottles)
  ];

  const all = [];
  for (const p of passes) {
    const res = detectBottles(cv, det.mat, p.lower, p.upper, {
      minArea: minAreaScaled,
      aspectMin: 2.0,
      aspectMax: 6.5,
      nmsIou: 0.35,
    });
    all.push(...res.boxes);
  }

  if (det.owns) det.mat.delete();
  roiMat.delete();

  // Merge duplicates across passes
  const combinedSmall = nmsBoxes(all, 0.30);

  // Scale back to full-res ROI coords and then offset to full image coords.
  const scaled = (detScale === 1.0)
    ? combinedSmall
    : scaleBoxesToOriginal(combinedSmall, detInv, rw, rh);

  const boxes = scaled.map((b) => ({
    x: b.x + rx,
    y: b.y + ry,
    w: b.w,
    h: b.h,
    area: b.w * b.h,
  }));

  // Enforce the fixed 3x2 grid (prevents curtain false positives and fills missing empty bottles)
  return enforceFixedBottleGrid(cv, bgr, boxes);
}

function detectRockBottles(cv, bgr, boxes) {
  // Rock bottles are indicated by a pile of light-colored rocks near the *base* of the bottle.
  // Depending on the screenshot, that pile can be slightly *inside* the bottle outline and/or
  // slightly *below* it. So we scan the bottom portion of the bottle and a bit below.

  const hsv = new cv.Mat();
  cv.cvtColor(bgr, hsv, cv.COLOR_BGR2HSV);

  const H = hsv.rows;
  const W = hsv.cols;

  const flags = [];
  const rockBoxes = []; // per-bottle rock bbox in absolute image coords, or null

  for (let bi = 0; bi < boxes.length; bi++) {
    const b = boxes[bi];

    // Region: bottom ~30% of bottle + up to 30% below the bottle.
    const yStart = clampInt(Math.floor(b.y + b.h * 0.70), 0, H - 1);
    const yEnd = clampInt(Math.floor(b.y + b.h + b.h * 0.30), 0, H);

    if (yStart >= yEnd) {
      flags.push(false);
      rockBoxes.push(null);
      continue;
    }

    const rect = new cv.Rect(b.x, yStart, b.w, yEnd - yStart);
    const region = hsv.roi(rect);
    const channels = new cv.MatVector();
    cv.split(region, channels);
    const s = channels.get(1);
    const v = channels.get(2);

    // rock mask: low saturation + high value (rocks are light/grey-ish).
    // Be a bit generous here; we later restrict by position (bottom of bottle)
    // and by connected-component size.
    const sMask = new cv.Mat();
    const vMask = new cv.Mat();
    cv.threshold(s, sMask, 110, 255, cv.THRESH_BINARY_INV); // s < 110
    cv.threshold(v, vMask, 150, 255, cv.THRESH_BINARY);     // v > 150
    const rockMask = new cv.Mat();
    cv.bitwise_and(sMask, vMask, rockMask);

    const denom = rockMask.rows * rockMask.cols;
    const ratio = denom > 0 ? (cv.countNonZero(rockMask) / denom) : 0;
    const isRock = ratio >= 0.06;
    flags.push(isRock);

    let rockBox = null;
    if (isRock) {
      // Find the largest connected rock component to get a stable bounding box.
      const contours = new cv.MatVector();
      const hierarchy = new cv.Mat();
      cv.findContours(rockMask, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);

      let bestRect = null;
      let bestArea = 0;

      for (let i = 0; i < contours.size(); i++) {
        const cnt = contours.get(i);
        const area = cv.contourArea(cnt);
        if (area > bestArea) {
          bestArea = area;
          bestRect = cv.boundingRect(cnt);
        }
        cnt.delete();
      }

      if (bestRect) {
        let rx = b.x + bestRect.x;
        let ry = yStart + bestRect.y;
        let rw = bestRect.width;
        let rh = bestRect.height;

        // Pad slightly so the debug box feels like it wraps the whole rock pile.
        const pad = Math.max(2, Math.round(Math.min(rw, rh) * 0.10));
        rx = clampInt(rx - pad, 0, W - 1);
        ry = clampInt(ry - pad, 0, H - 1);
        rw = clampInt(rw + pad * 2, 1, W - rx);
        rh = clampInt(rh + pad * 2, 1, H - ry);

        rockBox = { x: rx, y: ry, w: rw, h: rh, idx: bi };
      }

      contours.delete();
      hierarchy.delete();
    }

    rockBoxes.push(rockBox);

    // cleanup
    region.delete();
    s.delete(); v.delete();
    channels.delete();
    sMask.delete(); vMask.delete(); rockMask.delete();
  }

  hsv.delete();
  return { flags, rockBoxes };
}

function sampleBottleSlots(cv, bgr, boxes, capacity, rockBoxes = null, debugSampling = false) {
  const hsv = new cv.Mat();
  cv.cvtColor(bgr, hsv, cv.COLOR_BGR2HSV);

  const bottles = [];
  const samplesBgr = [];
  const samplePoints = [];

  let dbgCount = 0;
  const dbgLimit = 16;

  for (let bi = 0; bi < boxes.length; bi++) {
    const b = boxes[bi];

    const mx = Math.round(b.w * 0.28);
    const top = Math.round(b.h * 0.18);
    const bot = Math.round(b.h * 0.04);

    const ix = b.x + mx;
    const iw = Math.max(1, b.w - 2 * mx);
    const iy = b.y + top;
    const ih = Math.max(1, b.h - top - bot);
    const slotH = ih / capacity;

    const slots = [];

    for (let si = 0; si < capacity; si++) {
      const cy = Math.round(iy + (si + 0.5) * slotH);
      const patchHalf = 6;
      const y1 = Math.max(iy, cy - patchHalf);
      const y2 = Math.min(iy + ih, cy + patchHalf);
      const rect = new cv.Rect(ix, y1, iw, Math.max(1, y2 - y1));

      const patchBgr = bgr.roi(rect);
      const patchHsv = hsv.roi(rect);

      // If this patch overlaps the detected rock pile for this bottle, force it to be empty.
      // This prevents the rock texture (light/desaturated) from getting clustered as a liquid color.
      let forceEmpty = false;
      const rb = rockBoxes && rockBoxes[bi];
      if (rb) {
        const px = ix;
        const py = y1;
        const pw = iw;
        const ph = Math.max(1, y2 - y1);

        const ix0 = Math.max(px, rb.x);
        const iy0 = Math.max(py, rb.y);
        const ix1 = Math.min(px + pw, rb.x + rb.w);
        const iy1 = Math.min(py + ph, rb.y + rb.h);

        const interW = Math.max(0, ix1 - ix0);
        const interH = Math.max(0, iy1 - iy0);
        const inter = interW * interH;
        if (inter > 0) {
          const ratio = inter / (pw * ph + 1e-6);
          if (ratio >= 0.10) forceEmpty = true;
        }
      }

      const medHsv = medianHsv(patchHsv);
      let filled = isFilledSlot(medHsv);
      if (forceEmpty) filled = false;

      let sampleIdx = null;
      let medBgr = null;

      if (filled) {
        medBgr = medianBgr(patchBgr);
        sampleIdx = samplesBgr.length;
        samplesBgr.push(medBgr);
      }

      if (debugSampling && dbgCount < dbgLimit) {
        const ch = patchBgr.channels();
        const stepBytes = _rowStepBytes(patchBgr);
        const expectedLen = patchBgr.rows * patchBgr.cols * ch;
        const dataLen = patchBgr.data.length;
        // cv.mean handles ROI stride correctly; use it as a reference sanity-check.
        const mean = cv.mean(patchBgr); // [b,g,r,a]
        const meanBgr = [mean[0], mean[1], mean[2]].map((x) => Math.round(x));
        const medBgrTxt = medBgr ? `[${medBgr.map((x) => Math.round(x)).join(',')}]` : 'null';
        log(
          `sample b${bi + 1} s${si + 1} filled=${filled} rect=(${ix},${y1},${iw},${Math.max(1, y2 - y1)}) ` +
          `rows=${patchBgr.rows} cols=${patchBgr.cols} ch=${ch} step=${stepBytes} dataLen=${dataLen} expected=${expectedLen} ` +
          `medianBGR=${medBgrTxt} meanBGR=[${meanBgr.join(',')}] medianHSV=[${medHsv.map((x) => Math.round(x)).join(',')}]`
        );
        dbgCount++;
      }

      const cx = Math.round(ix + iw / 2);

      slots.push({
        filled,
        sample_idx: sampleIdx,
        hsv: medHsv,
        bgr: medBgr,
      });

      samplePoints.push({
        bottleIndex: bi,
        slotIndex: si,
        x: cx,
        y: cy,
        colorId: null,
        colorHex: null,
      });

      patchBgr.delete();
      patchHsv.delete();
    }

    bottles.push({
      bbox: [b.x, b.y, b.w, b.h],
      slots,
      rock: false,
    });
  }

  hsv.delete();
  return { bottles, samplesBgr, samplePoints };
}

function clusterColorsDbscan(cv, samplesBgr, eps) {
  const n = samplesBgr.length;
  if (n === 0) return { labels: [], clusterInfo: {} };

  // Build BGR mat Nx1, CV_8UC3
  const bgrMat = new cv.Mat(n, 1, cv.CV_8UC3);
  for (let i = 0; i < n; i++) {
    const [b, g, r] = samplesBgr[i];
    bgrMat.data[i * 3 + 0] = b;
    bgrMat.data[i * 3 + 1] = g;
    bgrMat.data[i * 3 + 2] = r;
  }

  const labMat = new cv.Mat();
  cv.cvtColor(bgrMat, labMat, cv.COLOR_BGR2Lab);

  const labs = [];
  for (let i = 0; i < n; i++) {
    labs.push([
      labMat.data[i * 3 + 0],
      labMat.data[i * 3 + 1],
      labMat.data[i * 3 + 2],
    ]);
  }

  bgrMat.delete();
  labMat.delete();

  // DBSCAN (minPts=1) => connected components under eps distance
  const labels = new Array(n).fill(-1);
  let cid = 0;

  function dist(a, b) {
    const dx = a[0] - b[0];
    const dy = a[1] - b[1];
    const dz = a[2] - b[2];
    return Math.sqrt(dx * dx + dy * dy + dz * dz);
  }

  for (let i = 0; i < n; i++) {
    if (labels[i] !== -1) continue;
    // BFS flood fill
    const queue = [i];
    labels[i] = cid;
    while (queue.length) {
      const p = queue.pop();
      for (let j = 0; j < n; j++) {
        if (labels[j] !== -1) continue;
        if (dist(labs[p], labs[j]) <= eps) {
          labels[j] = cid;
          queue.push(j);
        }
      }
    }
    cid++;
  }

  const clusterInfo = {};
  for (let k = 0; k < cid; k++) {
    const idxs = [];
    for (let i = 0; i < n; i++) if (labels[i] === k) idxs.push(i);

    const count = idxs.length;
    let sb = 0, sg = 0, sr = 0;
    let sL = 0, sA = 0, sB = 0;
    for (const i of idxs) {
      const [b, g, r] = samplesBgr[i];
      sb += b; sg += g; sr += r;
      const [L, A, B] = labs[i];
      sL += L; sA += A; sB += B;
    }
    const cb = sb / count, cg = sg / count, cr = sr / count;
    const cL = sL / count, cA = sA / count, cB = sB / count;
    clusterInfo[k] = {
      id: k,
      count,
      center_bgr: [cb, cg, cr],
      center_lab: [cL, cA, cB],
      center_hex: bgrToHex(cb, cg, cr),
    };
  }

  return { labels, clusterInfo };
}


function labDist3(a, b) {
  const dx = a[0] - b[0];
  const dy = a[1] - b[1];
  const dz = a[2] - b[2];
  return Math.sqrt(dx * dx + dy * dy + dz * dz);
}

function mergeClustersToExpectedCount(labelsIn, clusterInfoIn, capacity, dbscanEps) {
  // If total filled slots is divisible by capacity, then each color must appear exactly `capacity` times.
  // That lets us infer the expected number of colors. If DBSCAN over-splits (e.g. "Yellow 1" + "Yellow 2"),
  // we greedily merge the closest clusters until the expected count is reached.
  const labels = labelsIn.slice();
  const clusterInfo = clusterInfoIn; // mutate in place (caller doesn't reuse the raw object)

  const nSamples = labels.length;
  const cap = (typeof capacity === 'number' && capacity > 0) ? capacity : null;
  if (!cap || nSamples === 0) return { labels, clusterInfo, merges: [], expectedColors: null };
  if (nSamples % cap !== 0) return { labels, clusterInfo, merges: [], expectedColors: null };

  const expectedColors = nSamples / cap;
  const merges = [];

  const maxDist = Math.max(8.0, (typeof dbscanEps === 'number' ? dbscanEps : 16.0) * 1.2);

  function mergeInto(keepId, dropId) {
    // update labels
    for (let i = 0; i < labels.length; i++) {
      if (labels[i] === dropId) labels[i] = keepId;
    }

    const a = clusterInfo[keepId];
    const b = clusterInfo[dropId];
    if (!a || !b) return;

    const tot = a.count + b.count;
    const wA = a.count / tot;
    const wB = b.count / tot;

    const cb = a.center_bgr[0] * wA + b.center_bgr[0] * wB;
    const cg = a.center_bgr[1] * wA + b.center_bgr[1] * wB;
    const cr = a.center_bgr[2] * wA + b.center_bgr[2] * wB;

    const cl = a.center_lab[0] * wA + b.center_lab[0] * wB;
    const ca = a.center_lab[1] * wA + b.center_lab[1] * wB;
    const cb2 = a.center_lab[2] * wA + b.center_lab[2] * wB;

    a.count = tot;
    a.center_bgr = [cb, cg, cr];
    a.center_lab = [cl, ca, cb2];
    a.center_hex = bgrToHex(cb, cg, cr);

    delete clusterInfo[dropId];
  }

  while (Object.keys(clusterInfo).length > expectedColors) {
    const ids = Object.keys(clusterInfo).map((k) => parseInt(k, 10)).sort((a, b) => a - b);

    let best = null;
    for (let i = 0; i < ids.length; i++) {
      for (let j = i + 1; j < ids.length; j++) {
        const idA = ids[i];
        const idB = ids[j];
        const a = clusterInfo[idA];
        const b = clusterInfo[idB];
        if (!a || !b) continue;

        // Two different real colors will each have ~capacity occurrences, so their sum would exceed capacity.
        if (a.count + b.count > cap) continue;

        const d = labDist3(a.center_lab, b.center_lab);
        if (d > maxDist) continue;

        const keep = Math.min(idA, idB);
        const drop = Math.max(idA, idB);

        if (!best || d < best.d) best = { keep, drop, d };
      }
    }

    if (!best) break;

    mergeInto(best.keep, best.drop);
    merges.push(best);
  }

  return { labels, clusterInfo, merges, expectedColors };
}

function detectRedBadges(cv, bgr) {
  const hsv = new cv.Mat();
  cv.cvtColor(bgr, hsv, cv.COLOR_BGR2HSV);

  const lower1 = new cv.Mat(hsv.rows, hsv.cols, hsv.type(), [0, 120, 120, 0]);
  const upper1 = new cv.Mat(hsv.rows, hsv.cols, hsv.type(), [10, 255, 255, 255]);
  const lower2 = new cv.Mat(hsv.rows, hsv.cols, hsv.type(), [170, 120, 120, 0]);
  const upper2 = new cv.Mat(hsv.rows, hsv.cols, hsv.type(), [179, 255, 255, 255]);

  const m1 = new cv.Mat();
  const m2 = new cv.Mat();
  cv.inRange(hsv, lower1, upper1, m1);
  cv.inRange(hsv, lower2, upper2, m2);
  const mask = new cv.Mat();
  cv.bitwise_or(m1, m2, mask);

  const k3 = cv.Mat.ones(3, 3, cv.CV_8U);
  cv.morphologyEx(mask, mask, cv.MORPH_OPEN, k3);
  cv.dilate(mask, mask, k3);

  const contours = new cv.MatVector();
  const hierarchy = new cv.Mat();
  cv.findContours(mask, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);

  const H = hsv.rows;
  const candidates = [];
  for (let i = 0; i < contours.size(); i++) {
    const cnt = contours.get(i);
    const rect = cv.boundingRect(cnt);
    const x = rect.x, y = rect.y, w = rect.width, h = rect.height;
    const area = w * h;

    if (area < 3500 || area > 25000) { cnt.delete(); continue; }
    if (y < H * 0.65) { cnt.delete(); continue; }

    const peri = cv.arcLength(cnt, true);
    const cArea = cv.contourArea(cnt);
    const circ = (peri <= 0) ? 0 : (4.0 * Math.PI * cArea / (peri * peri));
    const ar = w / (h + 1e-6);

    if (circ < 0.45) { cnt.delete(); continue; }
    if (ar < 0.7 || ar > 1.3) { cnt.delete(); continue; }

    candidates.push({ x, y, w, h, area });
    cnt.delete();
  }

  candidates.sort((a, b) => b.area - a.area);
  const boxes = candidates.slice(0, 3).map((c) => [c.x, c.y, c.w, c.h]);
  boxes.sort((a, b) => a[0] - b[0]);

  // cleanup
  hsv.delete();
  lower1.delete(); upper1.delete(); lower2.delete(); upper2.delete();
  m1.delete(); m2.delete(); mask.delete();
  k3.delete(); contours.delete(); hierarchy.delete();

  return boxes;
}

function buildResult(cv, bgr, capacity, dbscanEps, debugSampling = false) {
  const tStart = nowMs();

  
// 1) bottles (robust fixed 3x2 grid)
log('step: detect bottles');
const tDetect0 = nowMs();
const boxes = detectBottlesRobust(cv, bgr);
const tDetect1 = nowMs();

log(`step: bottles found = ${boxes.length}`);

if (!boxes || boxes.length === 0) {
  throw new Error('No bottles detected (outline HSV thresholds likely need tuning).');
}

// 2) rock
  log('step: detect rocks');
  const tRock0 = nowMs();
  const { flags: rockFlags, rockBoxes } = detectRockBottles(cv, bgr, boxes);
  const tRock1 = nowMs();

  // 3) slots
  log('step: sample slots');
  const tSlots0 = nowMs();
  const { bottles, samplesBgr, samplePoints } = sampleBottleSlots(cv, bgr, boxes, capacity, rockBoxes, debugSampling);
  const tSlots1 = nowMs();

  // 4) colors
  log('step: cluster colors');
  const tColors0 = nowMs();
  const { labels: rawLabels, clusterInfo: clusterInfo0 } = clusterColorsDbscan(cv, samplesBgr, dbscanEps);
  const clusters0 = Object.keys(clusterInfo0).length;

  // NOTE: capacity is the per-bottle slot count (typically 4). We use it to infer
  // the expected number of colors and to safely merge over-split clusters.
  const merged = mergeClustersToExpectedCount(rawLabels, clusterInfo0, capacity, dbscanEps);
  const labels = merged.labels;
  const clusterInfo = merged.clusterInfo;

  const tColors1 = nowMs();
  const clusters1 = Object.keys(clusterInfo).length;
  if (merged.merges.length) {
    log(`step: merged ${merged.merges.length} cluster pair(s) (${clusters0}→${clusters1}, expected=${merged.expectedColors})`);
  }

  // attach rock and color ids
  for (let bi = 0; bi < bottles.length; bi++) {
    bottles[bi].rock = !!rockFlags[bi];
    bottles[bi].rock_bbox = rockBoxes?.[bi] ? [rockBoxes[bi].x, rockBoxes[bi].y, rockBoxes[bi].w, rockBoxes[bi].h] : null;
    for (const slot of bottles[bi].slots) {
      if (slot.filled && slot.sample_idx !== null) {
        const cid = labels[slot.sample_idx];
        slot.color_id = cid;
        slot.color_hex = clusterInfo[cid].center_hex;
      } else {
        slot.color_id = null;
        slot.color_hex = null;
      }
    }
  }

  // color counts
  const colorsById = {};
  const colors = [];
  const colorUnitCounts = {};
  const ids = Object.keys(clusterInfo).map((k) => parseInt(k, 10)).sort((a, b) => a - b);
  for (const cid of ids) {
    colorsById[String(cid)] = clusterInfo[cid].center_hex;
    colors.push(clusterInfo[cid]);
    colorUnitCounts[String(cid)] = clusterInfo[cid].count;
  }

  // bottle contents (top->bottom ids + hex)
  const bottleContentsIds = bottles.map((b) => b.slots.map((s) => (s.color_id ?? null)));
  const bottleContentsHex = bottles.map((b) => b.slots.map((s) => (s.color_hex ?? null)));

  // powerup badge boxes (OCR in main thread)
  log('step: detect powerup badges');
  const tBadges0 = nowMs();
  const badgeBoxes = detectRedBadges(cv, bgr);
  const tBadges1 = nowMs();

  
// layout rows: fixed 3x2 ordering (fallback to y-clustering if something goes off-rail)
const layoutRows = (boxes.length === 6)
  ? [[0, 1, 2], [3, 4, 5]]
  : (() => {
      const medH = (() => {
        const hs = boxes.map((b) => b.h).sort((a, b) => a - b);
        return hs[Math.floor(hs.length / 2)] ?? 100;
      })();
      const rowEps = Math.max(40.0, medH * 0.60);
      const rowStructs = clusterRowsByY(boxes, rowEps);
      return rowStructs.map((row) => row.boxes.map((b) => boxes.indexOf(b)));
    })();

// Lightweight timing logs (visible in DevTools console)
  const tEnd = nowMs();
  log(`timing: detect=${(tDetect1 - tDetect0).toFixed(1)}ms rock=${(tRock1 - tRock0).toFixed(1)}ms slots=${(tSlots1 - tSlots0).toFixed(1)}ms colors=${(tColors1 - tColors0).toFixed(1)}ms badges=${(tBadges1 - tBadges0).toFixed(1)}ms total=${(tEnd - tStart).toFixed(1)}ms`);

  // fill samplePoints color hex (for overlay)
  for (const p of samplePoints) {
    const bi = p.bottleIndex;
    const si = p.slotIndex;
    const slot = bottles[bi]?.slots?.[si];
    if (slot && slot.color_id !== null) {
      p.colorId = slot.color_id;
      p.colorHex = slot.color_hex;
    }
  }

  return {
    result: {
      num_bottles: bottles.length,
      capacity,
      powerups: {
        retries: null,
        shuffles: null,
        add_bottles: null,
        badge_boxes: badgeBoxes,
        tesseract_available: true,
      },
      rock_bottles: rockFlags.map((f, i) => f ? i : null).filter((x) => x !== null),
      num_colors: ids.length,
      colors_by_id: colorsById,
      colors,
      color_unit_counts: colorUnitCounts,
      bottle_contents_ids: bottleContentsIds,
      bottle_contents_hex: bottleContentsHex,
      bottles,
      layout: { rows: layoutRows },
    },
    debug: {
      bottles: boxes.map((b, idx) => ({ ...b, idx, rock: !!rockFlags[idx] })),
      rockBoxes: rockBoxes || [],
      samplePoints,
      badgeBoxes,
    },
  };
}

self.onmessage = async (e) => {
  const msg = e.data || {};
  if (msg.type !== 'analyze') return;

  // Prevent accidental concurrent analyses (async/await can yield to the event loop).
  const reqId = (typeof msg.requestId === 'number') ? msg.requestId : null;
  if (isBusy) {
    self.postMessage({ type: 'error', requestId: reqId, error: 'Worker is busy; please retry.' });
    return;
  }

  isBusy = true;
  currentRequestId = reqId;

  const { width, height, rgba, capacity, dbscanEps, cvVer } = msg;

  try {
    log(`analyze: frame ${width}×${height}, capacity=${capacity ?? 4}, eps=${dbscanEps ?? '??'}, cv=${cvVer ?? '??'}`);

    if (cvPromise && cvVersionLoaded && cvVer && cvVersionLoaded !== cvVer) {
      log(`OpenCV already loaded (${cvVersionLoaded}); cannot switch to ${cvVer} without restarting worker.`);
    }

    const tCv0 = nowMs();
    const cv = await loadOpenCv(cvVersionLoaded || cvVer || '4.9.0');
    log('OpenCV ready');
    const tCv1 = nowMs();
    log(`opencv: ready in ${(tCv1 - tCv0).toFixed(1)}ms`);

    const imgData = new ImageData(new Uint8ClampedArray(rgba), width, height);
    const rgbaMat = cv.matFromImageData(imgData);
    const bgr = new cv.Mat();
    cv.cvtColor(rgbaMat, bgr, cv.COLOR_RGBA2BGR);

    const cap = (typeof capacity === 'number' && capacity > 0) ? capacity : 4;
    const eps = (typeof dbscanEps === 'number' && dbscanEps > 0) ? dbscanEps : 16.0;

    const { result, debug } = buildResult(cv, bgr, cap, eps, !!msg.debugSamples);

    rgbaMat.delete();
    bgr.delete();

    self.postMessage({ type: 'result', requestId: currentRequestId, result, debug });
  } catch (err) {
    self.postMessage({ type: 'error', requestId: currentRequestId, error: err?.message || String(err) });
  } finally {
    isBusy = false;
    currentRequestId = null;
  }
};

