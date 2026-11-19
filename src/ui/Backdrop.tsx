// The desk. The whole app sits on a drafting mat: a fine grid, and outlines
// marking the real sizes of the things this app reads (an A4 page, a passport
// page, an ID card), drawn to the same scale as each other. Inside each
// outline, faint lines of "text" write themselves in and are then picked out
// in the label and value colours, the way the reader finds them on a real
// page; this runs faster while a document is being read. A warm pool of light
// follows the pointer, like a desk lamp.
import { useEffect, useRef } from 'react';

// real sizes in millimetres, drawn at 0.2 vmin per mm so they stay in proportion
const SIZES: { id: string; label: string; w: number; h: number; lines: number[]; style: React.CSSProperties }[] = [
  { id: 'a4', label: 'A4 · 210 × 297 mm', w: 210, h: 297, lines: [46, 78, 64, 0, 88, 72, 90, 58, 0, 82, 40], style: { right: '4vmin', bottom: '-16vmin', rotate: '-3deg' } },
  { id: 'a5', label: 'A5 · 148 × 210 mm', w: 148, h: 210, lines: [52, 84, 70, 0, 90, 62, 76], style: { left: '-5vmin', bottom: '7vmin', rotate: '4deg' } },
  { id: 'td3', label: 'PASSPORT PAGE · TD3 · 125 × 88 mm', w: 125, h: 88, lines: [40, 66, 54, 0, 94, 94], style: { right: '30vmin', top: '10vmin', rotate: '2deg' } },
  { id: 'id1', label: 'ID CARD · ID-1 · 85.6 × 54 mm', w: 85.6, h: 54, lines: [50, 72, 60], style: { left: '38vw', bottom: '11vmin', rotate: '-5deg' } },
];

export function Backdrop({ busy }: { busy: boolean }) {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const el = ref.current!;
    let raf = 0, x = innerWidth * 0.3, y = innerHeight * 0.3;
    const move = (e: PointerEvent) => {
      x = e.clientX; y = e.clientY;
      if (!raf) raf = requestAnimationFrame(() => { raf = 0; el.style.setProperty('--mx', `${x}px`); el.style.setProperty('--my', `${y}px`); });
    };
    window.addEventListener('pointermove', move, { passive: true });
    return () => { window.removeEventListener('pointermove', move); cancelAnimationFrame(raf); };
  }, []);
  return (
    <div ref={ref} className={'backdrop' + (busy ? ' busy' : '')} aria-hidden>
      <div className="mat" />
      <div className="desklamp" />
      {SIZES.map((s, k) => (
        <div key={s.id} className={'size s-' + s.id} style={{ width: `${s.w * 0.2}vmin`, height: `${s.h * 0.2}vmin`, ['--k' as any]: k, ...s.style }}>
          <span>{s.label}</span>
          <i className="tl" /><i className="tr" /><i className="bl" /><i className="br" />
          <div className="inklines">
            {s.lines.map((w, i) => (w ? <b key={i} className={i % 3 === 1 ? 'v' : i % 3 === 0 ? 'q' : ''} style={{ width: `${w}%`, ['--i' as any]: i }} /> : <em key={i} />))}
          </div>
        </div>
      ))}
      <div className="vignette" />
    </div>
  );
}
