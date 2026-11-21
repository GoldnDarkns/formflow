// Camera capture for phones: a live view that finds the document while you aim and draws its
// outline, a large shutter, and a strip of what has been taken so far. Several pages can be shot in a
// row and handed over together. The picture never leaves the device. If the camera cannot be opened
// the person is sent to the ordinary photo picker instead.
//
// Finding the document (about nine looks a second): the classical finders in core/scan.ts on a small
// copy of the frame, and a small learned corner finder (core/corners.ts, 1.8 MB) on a 224 x 224 copy;
// core/track.ts decides which to believe. The chosen corners are then snapped onto the real edges in a
// larger copy. Between looks the outline is moved smoothly (QuadFilter) and redrawn every screen frame.
// When the document is held still, sharp, without glare and big enough, the picture is taken by itself
// ("Auto"); the shutter button always works too.
import { createPortal } from 'react-dom';
import { useEffect, useRef, useState } from 'react';
import { motion } from 'motion/react';
import { ICheck, IX } from './icons';
import { detectCard, detectObject, detectPage, refineQuad, type Quad } from '../core/scan';
import { rgbaToGray } from '../core/preprocess';
import { CORNER_SIZE, edgeContrast, plausibleQuad, type CornerModel } from '../core/corners';
import { chooseOutline, frameProblem, QuadFilter, type FrameQuality, type How } from '../core/track';
import { photoQuality } from '../core/quality';
import { cornerModel } from '../engine/engine';

const TRACK_SIDE = 320; // the classical finders look at a copy this many pixels on its long side
const TRACK_EVERY = 110; // milliseconds between looks
const AUTO_STILL = 7; // looks in a row the document must hold still (about 0.8 s) before Auto takes it
const AUTO_KEY = 'ff.autoCapture';

interface Canvases { small: HTMLCanvasElement; big: HTMLCanvasElement; net: HTMLCanvasElement }
const ctx2d = (c: HTMLCanvasElement) => c.getContext('2d', { willReadFrequently: true })!;
function draw(c: HTMLCanvasElement, src: CanvasImageSource, w: number, h: number) {
  if (c.width !== w || c.height !== h) { c.width = w; c.height = h; }
  const x = ctx2d(c);
  x.drawImage(src, 0, 0, w, h);
  return x.getImageData(0, 0, w, h).data;
}

interface Found { q: Quad; how: How; strong: boolean; quality: FrameQuality | null }
/**
 * Find the document in a video frame or canvas; corners as fractions (0..1) of the picture. `sharp` is
 * the long side of the copy the corners are snapped on (960 while aiming, 2400 for the picture taken).
 */
async function locate(src: CanvasImageSource, sw: number, sh: number, cv: Canvases, net: CornerModel | null, sharp = 960, measure = true): Promise<Found | null> {
  const k = TRACK_SIDE / Math.max(sw, sh);
  const w = Math.max(8, Math.round(sw * k)), h = Math.max(8, Math.round(sh * k));
  const d = draw(cv.small, src, w, h);
  const frac = (q: Quad | null) => (q ? (q.map(([x, y]) => [x / w, y / h]) as Quad) : null);
  const outline = frac(detectObject(d, w, h));
  const edges = outline ? null : frac(detectCard(d, w, h));
  const page = outline || edges ? null : frac(detectPage({ width: w, height: h, data: rgbaToGray(d, w, h).data }));
  let seen: Parameters<typeof chooseOutline>[0]['net'] = null;
  if (net && !outline) {
    try {
      const r = await net.find(draw(cv.net, src, CORNER_SIZE, CORNER_SIZE));
      seen = { quad: r.quad, score: r.score, plausible: plausibleQuad(r.quad), contrast: edgeContrast(d, w, h, r.quad.map(([x, y]) => [x * w, y * h]) as Quad) };
    } catch { /* the network failed on this frame: the classical finders still answer */ }
  }
  const pick = chooseOutline({ outline, edges, page, net: seen });
  if (!pick) return null;
  const K = Math.min(1, sharp / Math.max(sw, sh)), bw = Math.round(sw * K), bh = Math.round(sh * K);
  const bd = draw(cv.big, src, bw, bh);
  const r = refineQuad(bd, bw, bh, pick.q.map(([x, y]) => [x * bw, y * bh]) as Quad);
  return { q: r.map(([x, y]) => [x / bw, y / bh]) as Quad, how: pick.how, strong: pick.strong, quality: measure ? cropQuality(bd, bw, bh, r) : null };
}
/** blur, glare and brightness of the document only (the box around its corners), as in the reading checks */
function cropQuality(d: Uint8ClampedArray, w: number, h: number, q: Quad): FrameQuality | null {
  const x0 = Math.max(0, Math.floor(Math.min(...q.map((p) => p[0])))), x1 = Math.min(w, Math.ceil(Math.max(...q.map((p) => p[0]))));
  const y0 = Math.max(0, Math.floor(Math.min(...q.map((p) => p[1])))), y1 = Math.min(h, Math.ceil(Math.max(...q.map((p) => p[1]))));
  const cw = x1 - x0, ch = y1 - y0;
  if (cw < 40 || ch < 40) return null;
  const g = new Uint8Array(cw * ch);
  for (let y = 0; y < ch; y++) for (let x = 0; x < cw; x++) { const i = ((y + y0) * w + x + x0) * 4; g[y * cw + x] = (d[i] * 77 + d[i + 1] * 150 + d[i + 2] * 29) >> 8; }
  const p = photoQuality({ width: cw, height: ch, data: g });
  return { sharpness: p.sharpness, glary: p.glary, blurry: p.blurry, paper: p.paper };
}

interface Props {
  onShots: (files: { blob: Blob; name: string; quad?: Quad }[]) => void;
  onClose: () => void;
  onFallback: () => void; // open the system photo picker
}

export const cameraAvailable = () => typeof navigator !== 'undefined' && !!navigator.mediaDevices?.getUserMedia;

export function Camera({ onShots, onClose, onFallback }: Props) {
  const video = useRef<HTMLVideoElement>(null);
  const stream = useRef<MediaStream | null>(null);
  const [state, setState] = useState<'starting' | 'live' | 'error'>('starting');
  const [error, setError] = useState('');
  const [shape, setShape] = useState<'card' | 'page'>('card');
  const [shots, setShots] = useState<{ blob: Blob; url: string; quad?: Quad }[]>([]);
  // what the outline is doing: shown, settled, what stops a good picture, how far Auto has counted
  const [track, setTrack] = useState<{ steady: boolean; problem: string | null; strong: boolean; progress: number; taken: boolean; w: number; h: number } | null>(null);
  const follow = useRef({ last: null as Quad | null, miss: 0, still: 0, jump: null as Quad | null, sharp: [] as number[], armed: true, shotAt: null as Quad | null });
  const filter = useRef(new QuadFilter());
  const poly = useRef<SVGPolygonElement>(null);
  const dots = useRef<SVGGElement>(null);
  const cv = useRef<Canvases | null>(null);
  const net = useRef<CornerModel | null>(null);
  const [auto, setAuto] = useState(() => { try { return localStorage.getItem(AUTO_KEY) !== 'off'; } catch { return true; } });
  const autoRef = useRef(auto);
  const [flash, setFlash] = useState(0);
  const [torch, setTorch] = useState<'none' | 'off' | 'on'>('none'); // the phone's light, where the browser lets a page switch it
  const shootRef = useRef<() => void>(() => {});

  useEffect(() => {
    let cancelled = false;
    navigator.mediaDevices
      .getUserMedia({ audio: false, video: { facingMode: { ideal: 'environment' }, width: { ideal: 2560 }, height: { ideal: 1440 } } })
      .then((s) => {
        if (cancelled) { s.getTracks().forEach((t) => t.stop()); return; }
        stream.current = s;
        if (video.current) { video.current.srcObject = s; video.current.play().catch(() => {}); }
        setState('live');
        // Android Chrome can switch the light on; Safari on iPhone does not offer it to web pages
        const caps = s.getVideoTracks()[0]?.getCapabilities?.() as (MediaTrackCapabilities & { torch?: boolean }) | undefined;
        if (caps?.torch) setTorch('off');
      })
      .catch((e) => { if (!cancelled) { setError(e?.name === 'NotAllowedError' ? 'The camera was not allowed.' : 'The camera could not be opened.'); setState('error'); } });
    // the learned corner finder loads while the camera starts; until then (or if it cannot load) the classical finders work alone
    cornerModel().then((m) => { if (!cancelled) net.current = m; }).catch(() => {});
    const o = document.body.style.overflow;
    document.body.style.overflow = 'hidden';
    return () => { cancelled = true; stream.current?.getTracks().forEach((t) => t.stop()); document.body.style.overflow = o; };
  }, []);

  // follow the document while the camera is live
  useEffect(() => {
    if (state !== 'live') return;
    cv.current ??= { small: document.createElement('canvas'), big: document.createElement('canvas'), net: document.createElement('canvas') };
    let busy = false, raf = 0;
    const dist = (a: Quad, b: Quad) => Math.max(...a.map((p, i) => Math.hypot(p[0] - b[i][0], p[1] - b[i][1])));
    const lost = () => {
      const f = follow.current;
      if (++f.miss > 3) { f.last = null; f.still = 0; f.sharp = []; f.armed = true; filter.current.reset(); setTrack(null); }
    };
    const id = setInterval(async () => {
      const v = video.current, f = follow.current;
      if (busy || !v || !v.videoWidth || v.readyState < 2) return;
      busy = true;
      try {
        const r = await locate(v, v.videoWidth, v.videoHeight, cv.current!, net.current);
        if (!r) return lost();
        f.miss = 0;
        const move = f.last ? dist(r.q, f.last) : 1;
        // a jump to somewhere else is only believed when the next look finds the same place:
        // one stray frame (a hand, a shadow, a box in the background) does not throw the outline around
        if (f.last && move >= 0.12) {
          const confirmed = f.jump && dist(r.q, f.jump) < 0.05;
          f.jump = confirmed ? null : r.q;
          if (!confirmed) return;
          filter.current.reset();
        } else f.jump = null;
        filter.current.update(r.q, performance.now());
        f.still = move < 0.015 ? f.still + 1 : 0;
        f.last = r.q;
        if (r.quality) f.sharp = [...f.sharp.slice(-9), r.quality.sharpness];
        const problem = frameProblem(r.q, r.quality, Math.max(0, ...f.sharp));
        // Auto: once per document; it is ready again when the document leaves the view or moves well away
        if (!f.armed && f.shotAt && dist(r.q, f.shotAt) > 0.15) f.armed = true;
        const ready = !problem && r.strong && f.armed;
        const progress = autoRef.current && ready ? Math.min(1, f.still / AUTO_STILL) : 0;
        const fire = autoRef.current && ready && f.still >= AUTO_STILL;
        if (fire) { f.armed = false; f.shotAt = r.q; }
        setTrack({ steady: f.still >= 4 && !problem, problem, strong: r.strong, progress: fire ? 0 : progress, taken: autoRef.current && !f.armed, w: v.videoWidth, h: v.videoHeight });
        if (fire) shootRef.current();
      } catch { /* a frame that cannot be read is skipped */ } finally { busy = false; }
    }, TRACK_EVERY);
    // the outline is drawn every screen frame from the smoother, so it glides instead of jumping between looks
    const paint = () => {
      const q = filter.current.at(performance.now()), v = video.current;
      if (q && v?.videoWidth && poly.current) {
        poly.current.setAttribute('points', q.map(([x, y]) => `${(x * v.videoWidth).toFixed(1)},${(y * v.videoHeight).toFixed(1)}`).join(' '));
        dots.current?.querySelectorAll('circle').forEach((c, i) => { c.setAttribute('cx', String(q[i][0] * v.videoWidth)); c.setAttribute('cy', String(q[i][1] * v.videoHeight)); });
      }
      raf = requestAnimationFrame(paint);
    };
    raf = requestAnimationFrame(paint);
    return () => { clearInterval(id); cancelAnimationFrame(raf); };
  }, [state]);

  const toggleAuto = () => {
    const on = !auto;
    setAuto(on); autoRef.current = on;
    try { localStorage.setItem(AUTO_KEY, on ? 'on' : 'off'); } catch { /* private mode: the choice lasts for this visit */ }
  };
  const light = () => {
    const on = torch !== 'on';
    stream.current?.getVideoTracks()[0]?.applyConstraints({ advanced: [{ torch: on } as MediaTrackConstraintSet] }).then(() => setTorch(on ? 'on' : 'off')).catch(() => setTorch('none'));
  };
  const native = () => {
    stream.current?.getTracks().forEach((t) => t.stop());
    if (shots.length) onShots(shots.map((s, i) => ({ blob: s.blob, name: `Photo ${i + 1}`, quad: s.quad })));
    onFallback();
    onClose();
  };
  const shoot = async () => {
    const v = video.current;
    if (!v || !v.videoWidth) return;
    const c = document.createElement('canvas');
    c.width = v.videoWidth; c.height = v.videoHeight;
    c.getContext('2d')!.drawImage(v, 0, 0);
    setFlash((n) => n + 1);
    navigator.vibrate?.(20);
    // the corners in this very picture, sharpened on a large copy; failing that, the outline that was on screen
    let quad: Quad | undefined;
    const tmp: Canvases = { small: document.createElement('canvas'), big: document.createElement('canvas'), net: document.createElement('canvas') };
    try { quad = (await locate(c, c.width, c.height, tmp, net.current, 2400, false))?.q ?? follow.current.last ?? undefined; } catch { quad = follow.current.last ?? undefined; }
    c.toBlob((blob) => { if (blob) setShots((s) => [...s, { blob, url: URL.createObjectURL(blob), quad }]); }, 'image/jpeg', 0.92);
  };
  useEffect(() => { shootRef.current = () => { void shoot(); }; }); // the tracking loop always calls the latest shoot
  const finish = () => {
    onShots(shots.map((s, i) => ({ blob: s.blob, name: `Photo ${i + 1}`, quad: s.quad })));
    shots.forEach((s) => URL.revokeObjectURL(s.url));
    onClose();
  };

  const hint = !track
    ? shape === 'card' ? 'Looking for the card. Hold it flat, in good light.' : 'Looking for the page. Fit all of it in view, flat and well lit.'
    : track.taken ? 'Picture taken. Show the next document, or press Read.'
    : track.problem ?? (track.progress > 0 ? 'Hold still… taking the picture.' : track.steady ? 'Document found. Take the picture.' : 'Document found. Hold still.');

  return createPortal(
    <motion.div className="camera" role="dialog" aria-modal="true" aria-label="Camera" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
      <video ref={video} playsInline muted autoPlay />
      {state === 'live' && (
        <>
          {track
            ? <svg className={'cam-track' + (track.steady ? ' steady' : '') + (track.problem ? ' trouble' : '')} viewBox={`0 0 ${track.w} ${track.h}`} preserveAspectRatio="xMidYMid slice" aria-hidden="true">
                <polygon ref={poly} />
                <g ref={dots}>{[0, 1, 2, 3].map((i) => <circle key={i} r={Math.max(track.w, track.h) / 110} />)}</g>
              </svg>
            : <div className={'cam-frame as-' + shape}><i /><i /><i /><i /></div>}
          <p className={'cam-hint' + (track?.problem ? ' warn' : '')} role="status">{hint}</p>
        </>
      )}
      {flash > 0 && <motion.div key={flash} className="cam-flash" initial={{ opacity: 0.85 }} animate={{ opacity: 0 }} transition={{ duration: 0.25 }} />}
      {state === 'starting' && <p className="cam-msg">Opening the camera…</p>}
      {state === 'error' && (
        <div className="cam-msg">
          <p>{error}</p>
          <button className="btn primary" onClick={() => { onClose(); onFallback(); }}>Choose a photo instead</button>
        </div>
      )}
      <div className="cam-top">
        <button className="cam-round" onClick={onClose} aria-label="Close the camera"><IX /></button>
        <div className="cam-seg" role="radiogroup" aria-label="What are you scanning?">
          <button role="radio" aria-checked={shape === 'card'} className={shape === 'card' ? 'on' : ''} onClick={() => setShape('card')}>Card</button>
          <button role="radio" aria-checked={shape === 'page'} className={shape === 'page' ? 'on' : ''} onClick={() => setShape('page')}>Page</button>
        </div>
        {torch === 'none' ? (
          // Safari on iPhone does not let a web page switch the light on, but the phone's own camera has
          // its flash: hand over to it (the photos taken so far are kept and read too)
          <button className="cam-flashbtn" onClick={native} title="Opens the phone's own camera, which has the flash">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinejoin="round"><path d="M13 2 4 14h7l-1 8 9-12h-7z" /></svg> Flash
          </button>
        ) : (
          <button className={'cam-round' + (torch === 'on' ? ' on' : '')} onClick={light} aria-pressed={torch === 'on'} aria-label={torch === 'on' ? 'Switch the light off' : 'Switch the light on'} title="Light">
            <svg width="22" height="22" viewBox="0 0 24 24" fill={torch === 'on' ? 'currentColor' : 'none'} stroke="currentColor" strokeWidth="2" strokeLinejoin="round"><path d="M13 2 4 14h7l-1 8 9-12h-7z" /></svg>
          </button>
        )}
      </div>
      <div className="cam-bottom">
        <div className="cam-left">
          <button className={'cam-auto' + (auto ? ' on' : '')} onClick={toggleAuto} aria-pressed={auto} title="Take the picture by itself when the document is steady and sharp">Auto {auto ? 'on' : 'off'}</button>
          <div className="cam-strip">{shots.slice(-3).map((s) => <motion.img key={s.url} src={s.url} alt="" initial={{ scale: 0.4, opacity: 0 }} animate={{ scale: 1, opacity: 1 }} />)}{shots.length > 3 && <span>+{shots.length - 3}</span>}</div>
        </div>
        <button className="cam-shutter" onClick={() => void shoot()} disabled={state !== 'live'} aria-label="Take the picture">
          <i />
          {/* Auto counting down: the ring fills while the document is held still */}
          {auto && track && track.progress > 0 && <svg className="cam-ring" viewBox="0 0 84 84" aria-hidden="true"><circle cx="42" cy="42" r="39" pathLength={1} style={{ strokeDashoffset: 1 - track.progress }} /></svg>}
        </button>
        <button className="cam-done" onClick={finish} disabled={!shots.length}><ICheck /> {shots.length ? `Read ${shots.length}` : 'Read'}</button>
      </div>
    </motion.div>,
    document.body,
  );
}
