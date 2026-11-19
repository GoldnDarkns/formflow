// Guided tour: a spotlight walks a first-time visitor (or an examiner) through
// the app on the sample scans. Each step may prepare the screen (switch tab,
// open a document), then points at one element and explains it in a sentence.
import { useEffect, useLayoutEffect, useRef, useState } from 'react';
import { AnimatePresence, motion } from 'motion/react';

export interface TourStep {
  title: string;
  body: string;
  target?: string; // CSS selector of the element to point at; none = a centred card
  before?: () => void; // make the screen ready for this step
  ready?: () => boolean; // "Next" waits for this (e.g. the samples have been read)
  waiting?: () => string; // what to show while waiting
}

interface Props {
  steps: TourStep[];
  index: number;
  onIndex: (i: number) => void;
  onClose: () => void;
}

interface Rect { x: number; y: number; w: number; h: number }

export function Tour({ steps, index, onIndex, onClose }: Props) {
  const step = steps[index];
  const [rect, setRect] = useState<Rect | null>(null);
  const [, tick] = useState(0);
  const card = useRef<HTMLDivElement>(null);
  const scrolled = useRef(-1);
  const prepared = useRef(-1);

  // prepare the screen once per step
  useEffect(() => {
    if (prepared.current === index) return;
    prepared.current = index;
    scrolled.current = -1;
    setRect(null);
    step.before?.();
  }, [index, step]);

  // follow the target: screens animate in, lists grow, the window resizes
  useEffect(() => {
    const look = () => {
      tick((n) => n + 1); // re-evaluate ready()
      const el = step.target ? (document.querySelector(step.target) as HTMLElement | null) : null;
      if (!el) return setRect(null);
      if (scrolled.current !== index) {
        scrolled.current = index;
        // on a phone the card docks at the bottom, so bring the target to the top of the screen
        if (window.innerWidth < 720) window.scrollTo({ top: Math.max(0, window.scrollY + el.getBoundingClientRect().top - 22), behavior: 'smooth' });
        else el.scrollIntoView({ block: 'center', behavior: 'smooth' });
      }
      const r = el.getBoundingClientRect();
      setRect((p) => (p && Math.abs(p.x - r.left) < 0.5 && Math.abs(p.y - r.top) < 0.5 && Math.abs(p.w - r.width) < 0.5 && Math.abs(p.h - r.height) < 0.5 ? p : { x: r.left, y: r.top, w: r.width, h: r.height }));
    };
    look();
    const id = setInterval(look, 200);
    window.addEventListener('resize', look);
    window.addEventListener('scroll', look, true);
    return () => { clearInterval(id); window.removeEventListener('resize', look); window.removeEventListener('scroll', look, true); };
  }, [index, step]);

  const ready = step.ready ? step.ready() : true;
  const last = index === steps.length - 1;
  const next = () => (last ? onClose() : ready && onIndex(index + 1));

  useEffect(() => {
    const key = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
      if (e.key === 'ArrowRight' || e.key === 'Enter') next();
      if (e.key === 'ArrowLeft' && index > 0) onIndex(index - 1);
    };
    window.addEventListener('keydown', key);
    return () => window.removeEventListener('keydown', key);
  });

  // place the card under the target, or above it when there is no room; on a phone it docks at the bottom
  const [pos, setPos] = useState<{ left: number; top: number } | null>(null);
  useLayoutEffect(() => {
    const c = card.current;
    if (!c) return;
    const vw = window.innerWidth, vh = window.innerHeight, cw = c.offsetWidth, ch = c.offsetHeight;
    if (!rect || vw < 720) return setPos(null);
    const below = rect.y + rect.h + 14 + ch < vh - 8;
    const above = rect.y - 14 - ch > 8;
    if (!below && !above) {
      // a tall target: stand beside it so the card does not cover what it is pointing at
      const mid = Math.min(vh - ch - 12, Math.max(12, Math.max(rect.y, 0) + Math.min(rect.h, vh) / 2 - ch / 2));
      if (rect.x - cw - 16 > 8) return setPos({ left: rect.x - cw - 16, top: mid });
      if (rect.x + rect.w + 16 + cw < vw - 8) return setPos({ left: rect.x + rect.w + 16, top: mid });
    }
    const top = below ? rect.y + rect.h + 14 : above ? rect.y - 14 - ch : Math.max(8, vh - ch - 16);
    const left = Math.min(vw - cw - 12, Math.max(12, rect.x + rect.w / 2 - cw / 2));
    setPos({ left, top });
  }, [rect, index, ready]);

  const pad = 6;
  return (
    <div className="tour" role="dialog" aria-modal="true" aria-label="Guided tour">
      {rect ? (
        <motion.div className="tour-spot" initial={false} animate={{ left: rect.x - pad, top: rect.y - pad, width: rect.w + pad * 2, height: rect.h + pad * 2 }} transition={{ type: 'spring', stiffness: 260, damping: 30 }} />
      ) : (
        <div className="tour-dim" />
      )}
      <AnimatePresence mode="wait">
        <motion.div key={index} ref={card} className={'tour-card' + (pos ? '' : rect ? (rect.y > window.innerHeight * 0.42 ? ' docked top' : ' docked') : ' centre')} style={pos ? { left: pos.left, top: pos.top } : undefined}
          initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }} transition={{ duration: 0.18 }}>
          <div className="tour-top">
            <span className="tour-count">{index + 1} of {steps.length}</span>
            <button className="tour-x" onClick={onClose} aria-label="End the tour">×</button>
          </div>
          <h3>{step.title}</h3>
          <p>{step.body}</p>
          {!ready && <p className="tour-wait"><span className="tour-pulse" />{step.waiting?.() ?? 'One moment…'}</p>}
          <div className="tour-actions">
            {index > 0 && <button className="btn sm ghost" onClick={() => onIndex(index - 1)}>Back</button>}
            <span className="grow" />
            <button className="btn sm primary" disabled={!ready} onClick={next}>{last ? 'Finish' : 'Next'}</button>
          </div>
          <div className="tour-dots" aria-hidden>{steps.map((_, i) => <i key={i} className={i === index ? 'on' : i < index ? 'done' : ''} />)}</div>
        </motion.div>
      </AnimatePresence>
    </div>
  );
}
