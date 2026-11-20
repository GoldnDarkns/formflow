// Camera capture for phones: a live view with a frame to line the document up
// in, a large shutter, and a strip of what has been taken so far. Several pages
// can be shot in a row and handed over together. The picture never leaves the
// device. If the camera cannot be opened the person is sent to the ordinary
// photo picker instead.
import { createPortal } from 'react-dom';
import { useEffect, useRef, useState } from 'react';
import { motion } from 'motion/react';
import { ICheck, IX } from './icons';

interface Props {
  onShots: (files: { blob: Blob; name: string }[]) => void;
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
  const [shots, setShots] = useState<{ blob: Blob; url: string }[]>([]);
  const [flash, setFlash] = useState(0);

  useEffect(() => {
    let cancelled = false;
    navigator.mediaDevices
      .getUserMedia({ audio: false, video: { facingMode: { ideal: 'environment' }, width: { ideal: 2560 }, height: { ideal: 1440 } } })
      .then((s) => {
        if (cancelled) { s.getTracks().forEach((t) => t.stop()); return; }
        stream.current = s;
        if (video.current) { video.current.srcObject = s; video.current.play().catch(() => {}); }
        setState('live');
      })
      .catch((e) => { if (!cancelled) { setError(e?.name === 'NotAllowedError' ? 'The camera was not allowed.' : 'The camera could not be opened.'); setState('error'); } });
    const o = document.body.style.overflow;
    document.body.style.overflow = 'hidden';
    return () => { cancelled = true; stream.current?.getTracks().forEach((t) => t.stop()); document.body.style.overflow = o; };
  }, []);

  const shoot = () => {
    const v = video.current;
    if (!v || !v.videoWidth) return;
    const c = document.createElement('canvas');
    c.width = v.videoWidth; c.height = v.videoHeight;
    c.getContext('2d')!.drawImage(v, 0, 0);
    setFlash((n) => n + 1);
    navigator.vibrate?.(20);
    c.toBlob((blob) => { if (blob) setShots((s) => [...s, { blob, url: URL.createObjectURL(blob) }]); }, 'image/jpeg', 0.92);
  };
  const finish = () => {
    onShots(shots.map((s, i) => ({ blob: s.blob, name: `Photo ${i + 1}` })));
    shots.forEach((s) => URL.revokeObjectURL(s.url));
    onClose();
  };

  return createPortal(
    <motion.div className="camera" role="dialog" aria-modal="true" aria-label="Camera" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
      <video ref={video} playsInline muted autoPlay />
      {state === 'live' && (
        <>
          <div className={'cam-frame as-' + shape}><i /><i /><i /><i /></div>
          <p className="cam-hint">{shape === 'card' ? 'Fill the frame with the card. Avoid glare.' : 'Fit the whole page in the frame, flat and well lit.'}</p>
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
        <span style={{ width: 44 }} />
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
