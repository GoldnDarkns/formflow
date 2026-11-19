// Review mode: one doubtful field at a time, built for a thumb. The top half
// shows that part of the page, enlarged; the bottom half shows what was read
// and three large choices. Nothing is accepted without the person saying so.
import { createPortal } from 'react-dom';
import { useEffect, useMemo, useRef, useState } from 'react';
import { AnimatePresence, motion } from 'motion/react';
import type { DocState, EField } from './state';
import { ICheck, IX } from './icons';

interface Props {
  s: DocState;
  cropOf: (f: EField) => number[] | null;
  onChange: (fields: EField[]) => void;
  onClose: () => void;
}

export function Review({ s, cropOf, onChange, onClose }: Props) {
  // the list is fixed when review starts, so fields do not jump about as they are confirmed
  const queue = useMemo(() => s.fields.filter((f) => f.status === 'review' && !f.confirmed).map((f) => f.uid), []); // eslint-disable-line react-hooks/exhaustive-deps
  const [i, setI] = useState(0);
  const [done, setDone] = useState(0);
  const input = useRef<HTMLTextAreaElement>(null);
  const f = s.fields.find((x) => x.uid === queue[i]);
  const finished = i >= queue.length || !queue.length;
  const patch = (p: Partial<EField>) => f && onChange(s.fields.map((x) => (x.uid === f.uid ? { ...x, ...p } : x)));
  const next = () => setI((n) => n + 1);
  const correct = () => { patch({ confirmed: true, status: 'ok' }); setDone((n) => n + 1); next(); };
  const remove = () => { if (f) onChange(s.fields.filter((x) => x.uid !== f.uid)); setDone((n) => n + 1); next(); };

  useEffect(() => {
    const key = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
      if (e.key === 'Enter' && !e.shiftKey && !finished) { e.preventDefault(); correct(); }
    };
    window.addEventListener('keydown', key);
    return () => window.removeEventListener('keydown', key);
  });
  // stop the page behind from scrolling while the sheet is open
  useEffect(() => { const o = document.body.style.overflow; document.body.style.overflow = 'hidden'; return () => { document.body.style.overflow = o; }; }, []);

  const crop = f ? cropOf(f) : null;
  // show some of the page around the field, so the person can tell where they are
  const wide = crop ? [crop[0] - crop[2] * 0.25, crop[1] - crop[3] * 1.2, crop[2] * 1.5, crop[3] * 3.4] : null;

  return createPortal(
    <motion.div className="reviewer" role="dialog" aria-modal="true" aria-label="Check the doubtful fields" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
      <motion.div className="review-sheet" initial={{ y: 40 }} animate={{ y: 0 }} exit={{ y: 40 }} transition={{ type: 'spring', stiffness: 300, damping: 30 }}>
        <div className="review-top">
          <span className="review-count">{finished ? 'Done' : `Check ${i + 1} of ${queue.length}`}</span>
          <div className="review-bar"><motion.div animate={{ width: `${(100 * Math.min(i, queue.length)) / Math.max(1, queue.length)}%` }} /></div>
          <button className="btn icon ghost" onClick={onClose} aria-label="Close review"><IX /></button>
        </div>
        <AnimatePresence mode="wait">
          {finished ? (
            <motion.div key="done" className="review-done" initial={{ opacity: 0, scale: 0.96 }} animate={{ opacity: 1, scale: 1 }}>
              <div className="review-tick"><ICheck /></div>
              <h2>{queue.length ? `${done} of ${queue.length} checked` : 'Nothing to check'}</h2>
              <p>{queue.length - done > 0 ? `${queue.length - done} skipped. They stay marked "to check".` : 'Every doubtful field has been looked at by a person.'}</p>
              <button className="btn primary" onClick={onClose}>Back to the document</button>
            </motion.div>
          ) : f && (
            <motion.div key={f.uid} className="review-body" initial={{ opacity: 0, x: 30 }} animate={{ opacity: 1, x: 0 }} exit={{ opacity: 0, x: -30 }} transition={{ duration: 0.18 }}>
              <div className="review-crop">
                {wide ? (
                  <svg viewBox={wide.join(' ')} preserveAspectRatio="xMidYMid meet">
                    <image href={s.doc.view} width={s.doc.width} height={s.doc.height} />
                    <rect x={crop![0]} y={crop![1]} width={crop![2]} height={crop![3]} fill="none" stroke="var(--marker)" strokeWidth={Math.max(2, crop![3] * 0.06)} rx={4} />
                  </svg>
                ) : <p className="muted small">This field has no place on the page.</p>}
              </div>
              <label className="review-label" htmlFor="review-value">{f.key || 'Unnamed field'}</label>
              <textarea id="review-value" ref={input} className="review-value" rows={Math.min(4, Math.max(1, Math.ceil(f.value.length / 22)))} value={f.value} onChange={(e) => patch({ value: e.target.value, edited: true })} />
              <div className="review-why">
                {f.reasons.map((r) => <span key={r} className="tag warn">{r}</span>)}
                {f.alt && <button className="btn sm" onClick={() => patch({ value: f.alt!, edited: true })}>Second reader saw "{f.alt}"</button>}
              </div>
            </motion.div>
          )}
        </AnimatePresence>
        {!finished && (
          <div className="review-actions">
            <button className="btn ghost" onClick={remove}>Not a field</button>
            <button className="btn" onClick={next}>Skip</button>
            <button className="btn primary grow" onClick={correct}><ICheck /> Correct</button>
          </div>
        )}
      </motion.div>
    </motion.div>,
    document.body,
  );
}
