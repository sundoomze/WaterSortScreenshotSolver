// colors.js
// Human-friendly names + 3-letter abbreviations for clustered colors.

export function hexToRgb(hex) {
  const h = hex.replace('#', '').trim();
  if (h.length !== 6) return { r: 0, g: 0, b: 0 };
  return {
    r: parseInt(h.slice(0, 2), 16),
    g: parseInt(h.slice(2, 4), 16),
    b: parseInt(h.slice(4, 6), 16),
  };
}

export function rgbToHsv(r, g, b) {
  // r,g,b in [0..255]
  const rf = r / 255, gf = g / 255, bf = b / 255;
  const max = Math.max(rf, gf, bf);
  const min = Math.min(rf, gf, bf);
  const d = max - min;

  let h = 0;
  if (d !== 0) {
    if (max === rf) h = ((gf - bf) / d) % 6;
    else if (max === gf) h = (bf - rf) / d + 2;
    else h = (rf - gf) / d + 4;
    h *= 60;
    if (h < 0) h += 360;
  }

  const s = max === 0 ? 0 : d / max;
  const v = max;
  return { h, s, v };
}

function isBrownish({ h, s, v }) {
  // Brown tends to be orange hue but lower value.
  return s >= 0.25 && v <= 0.55 && (h >= 10 && h <= 45);
}

export function guessColorName(hex) {
  const { r, g, b } = hexToRgb(hex);
  const hsv = rgbToHsv(r, g, b);

  if (hsv.v < 0.10) return 'Black';
  if (hsv.s < 0.10 && hsv.v > 0.90) return 'White';
  if (hsv.s < 0.12) return 'Gray';

  if (isBrownish(hsv)) return 'Brown';

  // --- Game-tuned tweaks ---
  // In this game UI, the lighter blue water often reads better as *Periwinkle*.
  // It tends to have a similar hue to Blue, but noticeably lower saturation.
  if (hsv.h >= 200 && hsv.h < 255 && hsv.s <= 0.55 && hsv.v >= 0.55) return 'Periwinkle';

  // Bright magenta-ish purples look like *Pink* in-game (even if the hue is in the purple band).
  if (hsv.h >= 295 && hsv.h < 345 && hsv.v >= 0.75) return 'Pink';

  const h = hsv.h;
  if (h < 12 || h >= 345) return 'Red';
  if (h < 35) return 'Orange';
  if (h < 65) return 'Yellow';
  if (h < 95) return 'Lime';
  if (h < 150) return 'Green';
  if (h < 190) return 'Cyan';
  if (h < 220) return 'Sky';
  if (h < 255) return 'Blue';
  if (h < 285) return 'Periwinkle';
  if (h < 315) return 'Purple';
  if (h < 345) return 'Pink';
  return 'Color';
}

export function abbrForName(name) {
  const n = name.toUpperCase();
  if (n.startsWith('ORANGE')) return 'ORA';
  if (n.startsWith('YELLOW')) return 'YEL';
  if (n.startsWith('LIME')) return 'LIM';
  if (n.startsWith('GREEN')) return 'GRE';
  if (n.startsWith('CYAN')) return 'CYA';
  if (n.startsWith('SKY')) return 'SKY';
  if (n.startsWith('BLUE')) return 'BLU';
  if (n.startsWith('PERIWINKLE')) return 'PER';
  if (n.startsWith('PURPLE')) return 'PUR';
  if (n.startsWith('PINK')) return 'PIN';
  if (n.startsWith('MAGENTA')) return 'MAG';
  if (n.startsWith('RED')) return 'RED';
  if (n.startsWith('BROWN')) return 'BRO';
  if (n.startsWith('GRAY')) return 'GRY';
  if (n.startsWith('WHITE')) return 'WHT';
  if (n.startsWith('BLACK')) return 'BLK';
  // fallback: first 3 letters
  return n.replace(/[^A-Z]/g, '').slice(0, 3).padEnd(3, 'X');
}

export function buildColorLabels(colorsById) {
  // colorsById: { "0": "#RRGGBB", ... }
  // returns:
  //  {
  //    namesById: {0:"Orange", ...},
  //    abbrById: {0:"ORA", ...},
  //    orderedIds: [0,1,2,...]
  //  }
  const ids = Object.keys(colorsById).map((k) => parseInt(k, 10)).sort((a, b) => a - b);

  const rawNames = new Map();
  for (const id of ids) rawNames.set(id, guessColorName(colorsById[String(id)]));

// Resolve the only two "legit" near-duplicate naming cases we expect in this game:
//   - Blue vs Periwinkle
//   - Green vs Lime
// Everything else should ideally be unique; if not, we still fall back to numbering.
const hsvById = new Map();
for (const id of ids) {
  const { r, g, b } = hexToRgb(colorsById[String(id)]);
  hsvById.set(id, rgbToHsv(r, g, b));
}

const blueGroup = ids.filter((id) => {
  const n = rawNames.get(id);
  return n === 'Blue' || n === 'Periwinkle';
});

if (blueGroup.length >= 2) {
  // Least saturated => Periwinkle, most saturated => Blue
  blueGroup.sort((a, b) => (hsvById.get(a).s - hsvById.get(b).s) || (hsvById.get(b).v - hsvById.get(a).v));
  const perId = blueGroup[0];
  const blueId = blueGroup[blueGroup.length - 1];
  rawNames.set(perId, 'Periwinkle');
  rawNames.set(blueId, 'Blue');
  for (let i = 1; i < blueGroup.length - 1; i++) {
    const id = blueGroup[i];
    const { s } = hsvById.get(id);
    rawNames.set(id, s <= 0.55 ? 'Periwinkle' : 'Blue');
  }
}

const greenGroup = ids.filter((id) => {
  const n = rawNames.get(id);
  return n === 'Green' || n === 'Lime';
});

if (greenGroup.length >= 2) {
  // Brightest => Lime, darkest => Green
  greenGroup.sort((a, b) => (hsvById.get(b).v - hsvById.get(a).v) || (hsvById.get(a).h - hsvById.get(b).h));
  const limeId = greenGroup[0];
  const greenId = greenGroup[greenGroup.length - 1];
  rawNames.set(limeId, 'Lime');
  rawNames.set(greenId, 'Green');
  for (let i = 1; i < greenGroup.length - 1; i++) {
    const id = greenGroup[i];
    const { v } = hsvById.get(id);
    rawNames.set(id, v >= 0.6 ? 'Lime' : 'Green');
  }
}

  // ensure uniqueness: if duplicates, append numbers
  const counts = new Map();
  for (const id of ids) {
    const n = rawNames.get(id);
    counts.set(n, (counts.get(n) ?? 0) + 1);
  }

  const seen = new Map();
  const namesById = {};
  const abbrById = {};
  for (const id of ids) {
    const base = rawNames.get(id);
    const total = counts.get(base) ?? 1;
    if (total <= 1) {
      namesById[id] = base;
      abbrById[id] = abbrForName(base);
    } else {
      const idx = (seen.get(base) ?? 0) + 1;
      seen.set(base, idx);
      const name = `${base} ${idx}`;
      namesById[id] = name;
      // keep abbreviation stable (BLU, etc.) + digit
      const baseAbbr = abbrForName(base);
      const suffix = String(idx);
      abbrById[id] = (baseAbbr.slice(0, 2) + suffix).padEnd(3, ' ');
    }
  }

  return { namesById, abbrById, orderedIds: ids };
}
