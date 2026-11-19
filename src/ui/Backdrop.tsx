// The desk. The whole app sits on a cutting mat, the kind a records clerk
// works on: a fine grid, and outlines marking the real sizes of the things
// this app reads (an A4 page, a passport page, an ID card), drawn to the same
// scale as each other. A scanner lamp travels down the mat; it moves faster
// and glows brighter while a page is being read. Pure CSS, nothing to compute.

// real sizes in millimetres, drawn at 0.2 vmin per mm so they stay in proportion
const SIZES: { id: string; label: string; w: number; h: number; style: React.CSSProperties }[] = [
  { id: 'a4', label: 'A4 · 210 × 297 mm', w: 210, h: 297, style: { right: '4vmin', bottom: '-18vmin', rotate: '-3deg' } },
  { id: 'a5', label: 'A5 · 148 × 210 mm', w: 148, h: 210, style: { left: '-6vmin', bottom: '6vmin', rotate: '4deg' } },
  { id: 'td3', label: 'PASSPORT PAGE · TD3 · 125 × 88 mm', w: 125, h: 88, style: { right: '30vmin', top: '9vmin', rotate: '2deg' } },
  { id: 'id1', label: 'ID CARD · ID-1 · 85.6 × 54 mm', w: 85.6, h: 54, style: { left: '38vw', bottom: '10vmin', rotate: '-5deg' } },
];

export function Backdrop({ busy }: { busy: boolean }) {
  return (
    <div className={'backdrop' + (busy ? ' busy' : '')} aria-hidden>
      <div className="mat" />
      {SIZES.map((s) => (
        <div key={s.id} className="size" style={{ width: `${s.w * 0.2}vmin`, height: `${s.h * 0.2}vmin`, ...s.style }}>
          <span>{s.label}</span>
          <i className="tl" /><i className="tr" /><i className="bl" /><i className="br" />
        </div>
      ))}
      <div className="lamp" />
      <div className="vignette" />
    </div>
  );
}
