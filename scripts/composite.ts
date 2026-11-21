// Made-up photos of a card on a surface, with the true corners, for the finder tests
// (eval-finder.ts, eval-corners.ts). The card is a made-up specimen; the surface is plain, patterned,
// or a busy background with fingers over one edge ("held").
import sharp from 'sharp';
import type { Pt } from '../src/core/scan';

let seed = 7;
export const reseed = (s: number) => { seed = s; };
export const rnd = () => (seed = (seed * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff;

export const SURFACES: Record<string, [number, number, number]> = { white: [250, 250, 248], cream: [236, 230, 214], grey: [196, 198, 202], 'dark wood': [92, 64, 44], black: [28, 28, 30] };
/** patterned surfaces drawn over the base colour: the classical finders' known weak spots */
export const PATTERNS: Record<string, (W: number, H: number) => string> = {
  stripes: (W, H) => { const s = 18 + rnd() * 30, a = rnd() * 180; return `<defs><pattern id="p" width="${s * 2}" height="${s * 2}" patternUnits="userSpaceOnUse" patternTransform="rotate(${a})"><rect width="${s}" height="${s * 2}" fill="rgb(70,90,140)"/></pattern></defs><rect width="${W}" height="${H}" fill="url(#p)"/>`; },
  grid: (W, H) => { const s = 40 + rnd() * 60; return `<defs><pattern id="p" width="${s}" height="${s}" patternUnits="userSpaceOnUse" patternTransform="rotate(${rnd() * 40})"><path d="M ${s} 0 L 0 0 0 ${s}" fill="none" stroke="rgb(60,60,60)" stroke-width="${2 + rnd() * 4}"/></pattern></defs><rect width="${W}" height="${H}" fill="url(#p)"/>`; },
  'wood grain': (W, H) => Array.from({ length: 70 }, (_, i) => { const y = (i / 70) * H + rnd() * 20; return `<path d="M 0 ${y} C ${W * 0.3} ${y + (rnd() - 0.5) * 60}, ${W * 0.7} ${y + (rnd() - 0.5) * 60}, ${W} ${y}" stroke="rgb(${(60 + rnd() * 60) | 0},${(40 + rnd() * 30) | 0},20)" stroke-width="${1 + rnd() * 6}" fill="none" opacity="0.7"/>`; }).join(''),
};

export async function composite(card: string, bg: [number, number, number], W = 1200, H = 1600, held = false, pattern?: keyof typeof PATTERNS, withCard = true) {
  const ang = (rnd() - 0.5) * 30;
  let cw = Math.round(W * (0.55 + rnd() * 0.25));
  // a tall page turned at an angle must still fit in the photo
  const meta = await sharp(card).metadata(), asp = meta.height! / meta.width!, ca = Math.abs(Math.cos((ang * Math.PI) / 180)), sa = Math.abs(Math.sin((ang * Math.PI) / 180));
  cw = Math.round(Math.min(cw, (W * 0.95) / (ca + asp * sa), (H * 0.95) / (sa + asp * ca)));
  const img = await sharp(card).resize(cw).toBuffer({ resolveWithObject: true });
  const ch = img.info.height;
  const cx = W / 2 + (rnd() - 0.5) * W * 0.1, cy = H / 2 + (rnd() - 0.5) * H * 0.3;
  const a = (ang * Math.PI) / 180;
  const truth = ([[-cw / 2, -ch / 2], [cw / 2, -ch / 2], [cw / 2, ch / 2], [-cw / 2, ch / 2]] as Pt[]).map(([x, y]) => [cx + x * Math.cos(a) - y * Math.sin(a), cy + x * Math.sin(a) + y * Math.cos(a)] as Pt);
  const rot = await sharp(img.data).ensureAlpha().rotate(ang, { background: { r: 0, g: 0, b: 0, alpha: 0 } }).png().toBuffer({ resolveWithObject: true });
  // a soft shadow under the card, and light falling off towards one side
  const shadow = await sharp(rot.data).ensureAlpha().extractChannel('alpha').blur(6).linear(0.25).toBuffer();
  const shade = await sharp({ create: { width: rot.info.width, height: rot.info.height, channels: 3, background: '#000' } }).joinChannel(shadow).png().toBuffer();
  const grad = Buffer.from(`<svg width="${W}" height="${H}"><defs><linearGradient id="g" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#000" stop-opacity="0"/><stop offset="1" stop-color="#000" stop-opacity="0.12"/></linearGradient></defs><rect width="100%" height="100%" fill="url(#g)"/></svg>`);
  const left = Math.round(cx - rot.info.width / 2), top = Math.round(cy - rot.info.height / 2);
  // "held in a hand": a busy background of coloured shapes behind the card, and three finger-like blobs over one of its edges
  const shapes = held ? Array.from({ length: 14 }, () => { const x = rnd() * W, y = rnd() * H, r = 60 + rnd() * 260, c = `rgb(${(40 + rnd() * 200) | 0},${(40 + rnd() * 200) | 0},${(40 + rnd() * 200) | 0})`; return rnd() < 0.5 ? `<ellipse cx="${x}" cy="${y}" rx="${r}" ry="${r * (0.4 + rnd())}" fill="${c}"/>` : `<rect x="${x - r}" y="${y - r / 2}" width="${r * 2}" height="${r * (0.5 + rnd())}" fill="${c}" transform="rotate(${rnd() * 90} ${x} ${y})"/>`; }).join('') : pattern ? PATTERNS[pattern](W, H) : '';
  const clutter = Buffer.from(`<svg width="${W}" height="${H}">${shapes}</svg>`);
  const side = truth[1], next = truth[2]; // fingers over the right-hand edge
  const fingers = held ? [0.25, 0.5, 0.75].map((f) => { const x = side[0] + (next[0] - side[0]) * f, y = side[1] + (next[1] - side[1]) * f; return `<ellipse cx="${x + 30}" cy="${y}" rx="95" ry="38" fill="rgb(214,168,140)" transform="rotate(${ang + (rnd() - 0.5) * 20} ${x} ${y})"/>`; }).join('') : '';
  const hand = Buffer.from(`<svg width="${W}" height="${H}">${fingers}</svg>`);
  const noise = Buffer.alloc(W * H * 3);
  for (let i = 0; i < noise.length; i++) noise[i] = 128 + (rnd() - 0.5) * 12;
  const layers = withCard
    ? [{ input: clutter }, { input: shade, left: left + 5, top: top + 7 }, { input: rot.data, left, top }, { input: hand }, { input: grad }]
    : [{ input: clutter }, { input: grad }];
  const buf = await sharp({ create: { width: W, height: H, channels: 3, background: { r: bg[0], g: bg[1], b: bg[2] } } })
    .composite([...layers, { input: noise, raw: { width: W, height: H, channels: 3 }, blend: 'soft-light' as const }])
    .jpeg({ quality: 85 }).toBuffer();
  return { buf, truth, W, H };
}

export const load = async (src: Buffer | string) => { const { data, info } = await sharp(src).rotate().flatten({ background: '#fff' }).ensureAlpha().raw().toBuffer({ resolveWithObject: true }); return { rgba: new Uint8ClampedArray(data.buffer, data.byteOffset, data.length), W: info.width, H: info.height }; };
/** mean distance of each true corner to the nearest found corner, in % of the photo diagonal */
export const err = (q: Pt[], t: Pt[], diag: number) => (100 * t.reduce((s, p) => s + Math.min(...q.map((c) => Math.hypot(c[0] - p[0], c[1] - p[1]))), 0)) / t.length / diag;
/** every true corner within 4% of the photo diagonal of a found one (corners may come back in another order) */
export const close = (q: Pt[], t: Pt[], diag: number) => t.every((p) => Math.min(...q.map((c) => Math.hypot(c[0] - p[0], c[1] - p[1]))) < diag * 0.04);
