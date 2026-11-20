// Camera capture for phones: a live view that finds the document while you aim and
// draws its outline (the same finder that cuts the page out afterwards, run on a small
// copy of each frame), a large shutter, and a strip of what has been taken so far. Several pages
// can be shot in a row and handed over together. The picture never leaves the
// device. If the camera cannot be opened the person is sent to the ordinary
// photo picker instead.
import { createPortal } from 'react-dom';
import { useEffect, useRef, useState } from 'react';
import { motion } from 'motion/react';
import { ICheck, IX } from './icons';
import { detectCard, detectObject, detectPage, type Pt, type Quad } from '../core/scan';
import { rgbaToGray } from '../core/preprocess';

const TRACK_SIDE = 320; // the finder looks at a copy this many pixels on its long side
const TRACK_EVERY = 110; // milliseconds between looks

/** Find the document in a video frame or canvas; corners as fractions (0..1) of the picture. */
function locate(src: CanvasImageSource, sw: number, sh: number, small: HTMLCanvasElement): Quad | null {
  const k = TRACK_SIDE / Math.max(sw, sh);
  const w = Math.max(8, Math.round(sw * k)), h = Math.max(8, Math.round(sh * k));
  if (small.width !== w || small.height !== h) { small.width = w; small.height = h; }
  const ctx = small.getContext('2d', { willReadFrequently: true })!;
  ctx.drawImage(src, 0, 0, w, h);
  const d = ctx.getImageData(0, 0, w, h).data;
  // a clean outline on a plain surface first; then four straight edges (a card held in a hand); then a bright page on a grainy surface
  const q = detectObject(d, w, h) ?? detectCard(d, w, h) ?? detectPage({ width: w, height: h, data: rgbaToGray(d, w, h).data });
  return q ? (q.map(([x, y]) => [x / w, y / h]) as Quad) : null;
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
  // the outline being followed: where it is, whether it has stopped moving, and the frame size it refers to
  const [track, setTrack] = useState<{ q: Quad; steady: boolean; w: number; h: number } | null>(null);
  const follow = useRef<{ q: Quad | null; miss: number; still: number }>({ q: null, miss: 0, still: 0 });
  const small = useRef<HTMLCanvasElement | null>(null);
  const [flash, setFlash] = useState(0);
  const [torch, setTorch] = useState<'none' | 'off' | 'on'>('none'); // the phone's light, where the browser lets a page switch it

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
    const o = document.body.style.overflow;
    document.body.style.overflow = 'hidden';
    return () => { cancelled = true; stream.current?.getTracks().forEach((t) => t.stop()); document.body.style.overflow = o; };
  }, []);

  // follow the document while the camera is live
  useEffect(() => {
    if (state !== 'live') return;
    small.current ??= document.createElement('canvas');
    const id = setInterval(() => {
      const v = video.current, f = follow.current;
      if (!v || !v.videoWidth || v.readyState < 2) return;
      let q: Quad | null = null;
      try { q = locate(v, v.videoWidth, v.videoHeight, small.current!); } catch { /* a frame that cannot be read is skipped */ }
      if (!q) {
        if (++f.miss > 3) { f.q = null; f.still = 0; setTrack(null); }
        return;
      }
      f.miss = 0;
      const move = f.q ? Math.max(...q.map((p, i) => Math.hypot(p[0] - f.q![i][0], p[1] - f.q![i][1]))) : 1;
      // small movements are smoothed so the outline does not shiver; a jump is followed at once
      f.q = f.q && move < 0.12 ? (q.map((p, i) => [f.q![i][0] * 0.4 + p[0] * 0.6, f.q![i][1] * 0.4 + p[1] * 0.6] as Pt) as Quad) : q;
      f.still = move < 0.015 ? f.still + 1 : 0;
      setTrack({ q: f.q, steady: f.still >= 4, w: v.videoWidth, h: v.videoHeight });
    }, TRACK_EVERY);
    return () => clearInterval(id);
  }, [state]);

  const light = () => {
    const on = torch !== 'on';
    stream.current?.getVideoTracks()[0]?.applyConstraints({ advanced: [{ torch: on } as MediaTrackConstraintSet] }).then(() => setTorch(on ? 'on' : 'off')).catch(() => setTorch('none'));
  };
  const shoot = () => {
    const v = video.current;
    if (!v || !v.videoWidth) return;
    const c = document.createElement('canvas');
    c.width = v.videoWidth; c.height = v.videoHeight;
    c.getContext('2d')!.drawImage(v, 0, 0);
    setFlash((n) => n + 1);
    navigator.vibrate?.(20);
    // the corners in this very picture; failing that, the outline that was on screen
    let quad: Quad | undefined;
    try { quad = locate(c, c.width, c.height, small.current ?? (small.current = document.createElement('canvas'))) ?? follow.current.q ?? undefined; } catch { quad = follow.current.q ?? undefined; }
    c.toBlob((blob) => { if (blob) setShots((s) => [...s, { blob, url: URL.createObjectURL(blob), quad }]); }, 'image/jpeg', 0.92);
  };
  const finish = () => {
    onShots(shots.map((s, i) => ({ blob: s.blob, name: `Photo ${i + 1}`, quad: s.quad })));
    shots.forEach((s) => URL.revokeObjectURL(s.url));
    onClose();
  };

  return createPortal(
    <motion.div className="camera" role="dialog" aria-modal="true" aria-label="Camera" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
      <video ref={video} playsInline muted autoPlay />
      {state === 'live' && (
        <>
          {track
            ? <svg className={'cam-track' + (track.steady ? ' steady' : '')} viewBox={`0 0 ${track.w} ${track.h}`} preserveAspectRatio="xMidYMid slice" aria-hidden="true">
                <polygon points={track.q.map(([x, y]) => `${(x * track.w).toFixed(1)},${(y * track.h).toFixed(1)}`).join(' ')} />
                {track.q.map(([x, y], i) => <circle key={i} cx={x * track.w} cy={y * track.h} r={Math.max(track.w, track.h) / 110} />)}
              </svg>
            : <div className={'cam-frame as-' + shape}><i /><i /><i /><i /></div>}
          <p className="cam-hint">{track ? (track.steady ? 'Document found. Take the picture.' : 'Document found. Hold still.') : shape === 'card' ? 'Looking for the card. Lay it flat on a plain surface, away from glare.' : 'Looking for the page. Fit all of it in view, flat and well lit.'}</p>
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
        {torch === 'none' ? <span style={{ width: 44 }} /> : (
          <button className={'cam-round' + (torch === 'on' ? ' on' : '')} onClick={light} aria-pressed={torch === 'on'} aria-label={torch === 'on' ? 'Switch the light off' : 'Switch the light on'} title="Light">
            <svg width="22" height="22" viewBox="0 0 24 24" fill={torch === 'on' ? 'currentColor' : 'none'} stroke="currentColor" strokeWidth="2" strokeLinejoin="round"><path d="M13 2 4 14h7l-1 8 9-12h-7z" /></svg>
          </button>
        )}
      </div>
      <div className="cam-bottom">
        <div className="cam-strip">{shots.slice(-3).map((s) => <motion.img key={s.url} src={s.url} alt="" initial={{ scale: 0.4, opacity: 0 }} animate={{ scale: 1, opacity: 1 }} />)}{shots.length > 3 && <span>+{shots.length - 3}</span>}</div>
        <button className="cam-shutter" onClick={shoot} disabled={state !== 'live'} aria-label="Take the picture"><i /></button>
        <button className="cam-done" onClick={finish} disabled={!shots.length}><ICheck /> {shots.length ? `Read ${shots.length}` : 'Read'}</button>
      </div>
    </motion.div>,
    document.body,
  );
}
