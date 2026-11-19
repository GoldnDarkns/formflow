import { useEffect, useRef, useState } from 'react';
import { AnimatePresence, animate, motion, useMotionValue, useTransform } from 'motion/react';
import type { Quad } from '../core/scan';
import { DOC_LABEL, type DocType } from '../core/classify';

export const typeColor = (t: DocType) => `var(--t-${t})`;
export const pct = (x: number) => `${Math.round(x * 100)}%`;
export const trustColor = (v: number) => (v >= 0.85 ? 'var(--good)' : v >= 0.65 ? 'var(--warn)' : 'var(--bad)');

/** Circular trust gauge. The arc and the number count up together when the value changes. */
export function Ring({ value, size = 56, stroke = 6, label }: { value: number; size?: number; stroke?: number; label?: string }) {
  const r = (size - stroke) / 2;
  const C = 2 * Math.PI * r;
  const mv = useMotionValue(0);
  const offset = useTransform(mv, (v) => C * (1 - v));
  const num = useRef<HTMLSpanElement>(null);
  useEffect(() => {
    const c = animate(mv, value, { duration: 0.9, ease: [0.2, 0.8, 0.2, 1] });
    const un = mv.on('change', (v) => num.current && (num.current.textContent = String(Math.round(v * 100))));
    return () => { c.stop(); un(); };
  }, [value, mv]);
  return (
    <span className="ring" style={{ width: size, height: size }} role="img" aria-label={`${label ?? 'Trust score'} ${pct(value)}`}>
      <svg width={size} height={size}>
        <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke="var(--line)" strokeWidth={stroke} />
        <motion.circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke={trustColor(value)} strokeWidth={stroke} strokeLinecap="round" strokeDasharray={C} style={{ strokeDashoffset: offset }} />
      </svg>
      <span className="n" style={{ fontSize: size * 0.34 }}><span ref={num}>0</span><small>%</small></span>
    </span>
  );
}

/** Rubber stamp that lands on a document once its type is recognised. */
export function Stamp({ type, confidence, big, animateIn = true }: { type: DocType; confidence: number; big?: boolean; animateIn?: boolean }) {
  return (
    <motion.div
      className={'stamp' + (big ? ' big' : '')}
      style={{ ['--c' as any]: typeColor(type) }}
      initial={animateIn ? { scale: 2.6, opacity: 0, rotate: -22 } : false}
      animate={{ scale: 1, opacity: 1, rotate: -7 }}
      transition={{ type: 'spring', stiffness: 520, damping: 22, mass: 0.9 }}
    >
      {DOC_LABEL[type]}
      <small>{pct(confidence)} sure</small>
    </motion.div>
  );
}

/**
 * The cut-out moment. The photo appears as taken; when the page corners are
 * found, a frame travels from the photo's edges to the page, the background
 * dims, and the flattened page takes its place.
 */
export function CutOut({ original, quad, flat, scanning }: { original?: string; quad?: Quad | null; flat?: string; scanning: boolean }) {
  const [stage, setStage] = useState<'photo' | 'cut' | 'flat'>('photo');
  useEffect(() => {
    if (!flat) return setStage('photo');
    if (!quad) return setStage('flat');
    setStage('cut');
    const t = setTimeout(() => setStage('flat'), 780);
    return () => clearTimeout(t);
  }, [flat, quad]);
  const q = quad ?? [[0, 0], [1, 0], [1, 1], [0, 1]];
  const P = (pts: number[][]) => pts.map(([x, y], i) => `${i ? 'L' : 'M'} ${(x * 100).toFixed(2)} ${(y * 100).toFixed(2)}`).join(' ') + ' Z';
  const full = P([[0, 0], [1, 0], [1, 1], [0, 1]]);
  const outer = 'M 0 0 L 100 0 L 100 100 L 0 100 Z ';
  return (
    <div className="cutout">
      <AnimatePresence mode="popLayout" initial={false}>
        {stage !== 'flat' ? (
          <motion.div key="photo" className="shot" exit={{ opacity: 0, scale: 1.04 }} transition={{ duration: 0.25 }}>
            {original && <img src={original} alt="" />}
            {stage === 'cut' && (
              <svg viewBox="0 0 100 100" preserveAspectRatio="none">
                <motion.path fillRule="evenodd" fill="rgb(8 12 30 / 0.72)" initial={{ d: outer + full }} animate={{ d: outer + P(q) }} transition={{ type: 'spring', stiffness: 170, damping: 20 }} />
                <motion.path fill="none" stroke="var(--marker)" strokeWidth="2.5" vectorEffect="non-scaling-stroke" strokeLinejoin="round" initial={{ d: full }} animate={{ d: P(q) }} transition={{ type: 'spring', stiffness: 170, damping: 20 }} />
                {q.map(([x, y], i) => (
                  <motion.circle key={i} r="1.6" fill="var(--marker)" initial={{ cx: [0, 100, 100, 0][i], cy: [0, 0, 100, 100][i] }} animate={{ cx: x * 100, cy: y * 100 }} transition={{ type: 'spring', stiffness: 170, damping: 20 }} />
                ))}
              </svg>
            )}
          </motion.div>
        ) : (
          <motion.div key="flat" className="shot" initial={{ opacity: 0, scale: quad ? 0.86 : 1, rotate: quad ? -2 : 0 }} animate={{ opacity: 1, scale: 1, rotate: 0 }} transition={{ type: 'spring', stiffness: 260, damping: 24 }}>
            <img src={flat} alt="" />
            {scanning && <div className="beam" />}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
