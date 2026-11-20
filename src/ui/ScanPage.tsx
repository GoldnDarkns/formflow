import { useEffect, useRef, useState } from 'react';
import { AnimatePresence, motion } from 'motion/react';
import type { Live, Step } from '../engine/engine';
import type { PreprocessOptions } from '../core/preprocess';
import type { DocState } from './state';
import { CutOut, Stamp } from './bits';
import { ICamera, IShield, ISpark, IUpload } from './icons';
import { Camera, cameraAvailable } from './Camera';
import type { Quad } from '../core/scan';

export interface SecondReaderState {
  enabled: boolean;
  status: 'off' | 'loading' | 'ready' | 'error';
  device?: string;
  loaded?: number;
  total?: number;
  error?: string;
}
export interface Active extends Live {
  name: string;
  steps: Step[];
}

interface Props {
  pre: PreprocessOptions;
  setPre: (p: PreprocessOptions) => void;
  second: SecondReaderState;
  toggleSecond: (on: boolean) => void;
  critic: boolean;
  setCritic: (on: boolean) => void;
  active: Active | null;
  queue: number;
  recent: DocState[];
  onFiles: (files: { blob: Blob; name: string; quad?: Quad }[]) => void;
  onOpen: (id: string) => void;
  onSorted: () => void;
  onTour: () => void;
}

const SAMPLES = [
  ['samples/specimen-visa.jpg', 'Scan 1'],
  ['samples/form-fax-cover.png', 'Scan 2'],
  ['samples/specimen-emirates-id-front.jpg', 'Scan 3'],
  ['samples/specimen-passport.jpg', 'Scan 4'],
  ['samples/specimen-driving-licence.jpg', 'Scan 5'],
  ['samples/specimen-emirates-id-back.jpg', 'Scan 6'],
  ['samples/form-report.png', 'Scan 7'],
  ['samples/form-w9-filled.jpg', 'Scan 8'],
  ['samples/photo-on-desk.jpg', 'Phone photo'],
];
export const SAMPLE_COUNT = SAMPLES.length;
/** The made-up sample documents shipped with the app. */
export async function sampleFiles(): Promise<{ blob: Blob; name: string; quad?: Quad }[]> {
  return Promise.all(SAMPLES.map(async ([f, name]) => ({ blob: await (await fetch(import.meta.env.BASE_URL + f)).blob(), name })));
}
const FAN = [
  { r: -16, c: 'var(--t-visa)' },
  { r: -5, c: 'var(--t-passport)' },
  { r: 6, c: 'var(--t-emirates_id)' },
  { r: 17, c: 'var(--t-driving_licence)' },
];

export function ScanPage(p: Props) {
  const fileRef = useRef<HTMLInputElement>(null);
  const camRef = useRef<HTMLInputElement>(null);
  const [over, setOver] = useState(false);
  const [camera, setCamera] = useState(false);
  const busy = !!p.active;
  // on a phone the reading panel sits below the intake sheet: bring it into view when work starts
  const liveRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (busy && window.innerWidth <= 920) liveRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }, [busy]);

  const pick = async (fl: FileList | null) => {
    if (!fl?.length) return;
    const out: { blob: Blob; name: string; quad?: Quad }[] = [];
    for (const f of [...fl]) {
      if (/\.zip$/i.test(f.name) || f.type === 'application/zip' || f.type === 'application/x-zip-compressed') {
        // a ZIP of scans: every image and PDF inside is read, in name order
        const { unzipScans, MAX_ZIP_PAGES } = await import('./bundle');
        const { pdfToImages } = await import('../engine/pdf');
        for (const e of await unzipScans(f)) {
          if (out.length >= MAX_ZIP_PAGES) break;
          if (e.pdf) out.push(...(await pdfToImages(e.blob, e.name)));
          else out.push({ blob: e.blob, name: e.name });
        }
      } else if (f.type === 'application/pdf' || /\.pdf$/i.test(f.name)) {
        // PDFs are split into pages; each page is sorted and read on its own
        const { pdfToImages } = await import('../engine/pdf');
        out.push(...(await pdfToImages(f, f.name)));
      } else if (f.type.startsWith('image/')) out.push({ blob: f, name: f.name });
    }
    p.onFiles(out);
  };
  const samples = async () => p.onFiles(await sampleFiles());
  const mb = (b?: number) => ((b ?? 0) / 1e6).toFixed(0);

  return (
    <div className="scan">
      <div
        className={'tray' + (over ? ' over' : '')}
        onDragOver={(e) => { e.preventDefault(); setOver(true); }}
        onDragLeave={() => setOver(false)}
        onDrop={(e) => { e.preventDefault(); setOver(false); pick(e.dataTransfer.files); }}
      >
        <h1>Drop the whole pile. <span className="shine">It sorts itself.</span></h1>
        <p className="lede" style={{ marginTop: '0.9rem' }}>
          Passports, Emirates IDs, licences, visas and forms, in any order. FormFlow works out what each page is, reads it, checks the numbers that can be checked, and tells you how sure it is.
        </p>
        <div className="fan" aria-hidden>
          {FAN.map((f, i) => (
            <motion.div key={i} className="pg" initial={{ rotate: 0, y: 30, opacity: 0 }} animate={{ rotate: over ? f.r * 1.5 : f.r, y: 0, opacity: 1 }} transition={{ type: 'spring', stiffness: 160, damping: 16, delay: over ? 0 : 0.08 * i }}>
              <b style={{ background: f.c }} /><i /><i style={{ width: '70%' }} /><i /><i style={{ width: '55%' }} /><i style={{ width: '82%' }} />
            </motion.div>
          ))}
        </div>
        <div className="row actions">
          <button className="btn primary choose" onClick={() => fileRef.current?.click()}><IUpload /> Choose images, PDFs or a ZIP</button>
          <button className="btn cam" onClick={() => (cameraAvailable() ? setCamera(true) : camRef.current?.click())}><ICamera /> Take a photo</button>
          <button className="btn ghost" disabled={busy} onClick={samples}>Try {SAMPLE_COUNT} sample scans</button>
          <button className="btn ghost tourbtn" disabled={busy} onClick={p.onTour}><ISpark /> Take the 60-second tour</button>
        </div>
        <input ref={camRef} type="file" accept="image/*" capture="environment" hidden onChange={(e) => { pick(e.target.files); e.target.value = ''; }} />
        <input ref={fileRef} type="file" accept="image/*,application/pdf,.zip,application/zip" multiple hidden onChange={(e) => { pick(e.target.files); e.target.value = ''; }} />
        <AnimatePresence>{camera && <Camera onShots={p.onFiles} onClose={() => setCamera(false)} onFallback={() => camRef.current?.click()} />}</AnimatePresence>
        <p className="row small muted hint"><IShield /> Everything is read on this device. Nothing is uploaded.</p>
      </div>

      <div className="live" ref={liveRef}>
        <AnimatePresence mode="popLayout">
          {p.active && (
            <motion.div key="active" className="sheet" layout initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, scale: 0.97 }}>
              <div className="row" style={{ marginBottom: '0.8rem' }}>
                <h2 className="grow">Reading {p.active.name}</h2>
                {p.queue > 0 && <span className="tag">{p.queue} waiting</span>}
              </div>
              <div className="scanner">
                <CutOut original={p.active.original} quad={p.active.quad} flat={p.active.flat} scanning />
                <ul className="steps">
                  {p.active.steps.map((s) => (
                    <li key={s.name} className={s.state}>
                      <span className="dot">{s.state === 'done' ? '✓' : ''}</span>
                      <span>{s.name}</span>
                      <span className="meta">{s.detail}</span>
                    </li>
                  ))}
                </ul>
              </div>
            </motion.div>
          )}
          {p.recent.length > 0 && (
            <motion.div key="recent" className="sheet" layout initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
              <div className="row">
                <h2 className="grow">{p.recent.length} sorted</h2>
                <button className="btn dark sm" onClick={p.onSorted}>Open the trays</button>
              </div>
              <div className="strip">
                <AnimatePresence initial={false}>
                  {p.recent.map((d) => (
                    <motion.button key={d.doc.id} className="mini" layout initial={{ opacity: 0, y: -70, scale: 1.35, rotate: -5 }} animate={{ opacity: 1, y: 0, scale: 1, rotate: 0 }} transition={{ type: 'spring', stiffness: 260, damping: 22 }} onClick={() => p.onOpen(d.doc.id)}>
                      <div className="thumb"><img src={d.doc.view} alt="" /><Stamp type={d.doc.docType} confidence={d.doc.classification.confidence} /></div>
                      <span>{d.doc.facts.find((f) => f.key === 'Full name')?.value ?? d.doc.fileName}</span>
                    </motion.button>
                  ))}
                </AnimatePresence>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        <details className="sheet settings" open={p.second.enabled && p.second.status === 'loading'}>
          <summary><ISpark /> Second reader, critic and scan clean-up</summary>
          <label className="toggle" style={{ marginTop: '0.7rem' }}>
            <input type="checkbox" checked={p.second.enabled} onChange={(e) => p.toggleSecond(e.target.checked)} />
            <div>
              <b>Second reader: Florence-2</b>
              <span>A vision-language model reads each page again. Where both readers agree the value is trusted; where they differ the field is flagged and you see both versions. Downloads about 275 MB once.</span>
            </div>
          </label>
          {p.second.enabled && p.second.status === 'loading' && (
            <div className="stack small" style={{ gap: '0.35rem', paddingBottom: '0.6rem' }}>
              <span className="muted">Downloading the model, {mb(p.second.loaded)} of {mb(p.second.total) || '275'} MB</span>
              <div className="progress"><div style={{ width: `${p.second.total ? (100 * (p.second.loaded ?? 0)) / p.second.total : 3}%` }} /></div>
            </div>
          )}
          {p.second.enabled && p.second.status === 'ready' && <p className="small" style={{ paddingBottom: '0.6rem' }}><span className="tag good">Ready on {p.second.device === 'webgpu' ? 'the GPU' : 'the CPU'}</span>{p.second.device !== 'webgpu' && <span className="muted"> Slower without WebGPU: 20 to 60 seconds a page.</span>}</p>}
          {p.second.enabled && p.second.status === 'error' && <p className="small" style={{ paddingBottom: '0.6rem' }}><span className="tag bad">Could not load: {p.second.error}</span></p>}
          <div className="toggle" style={{ flexDirection: 'column', gap: '0.4rem' }}>
            <div className="row" style={{ width: '100%' }}>
              <b className="grow">Text reader</b>
              <div className="seg" role="radiogroup" aria-label="Text reader">
                {([['ppocr', 'PP-OCRv5'], ['tesseract', 'Tesseract']] as const).map(([k, l]) => (
                  <button key={k} role="radio" aria-checked={(p.pre.reader ?? 'tesseract') === k} className={(p.pre.reader ?? 'tesseract') === k ? 'on' : ''} onClick={() => p.setPre({ ...p.pre, reader: k })}>{l}</button>
                ))}
              </div>
            </div>
            <span className="muted small">PP-OCRv5 is a neural reader (12 MB) and makes fewer mistakes, but is slower on phones without WebGPU. Tesseract is the classical baseline and the faster choice there.</span>
          </div>
          <label className="toggle">
            <input type="checkbox" checked={p.critic} onChange={(e) => p.setCritic(e.target.checked)} />
            <div><b>Critic: a second opinion on the type</b><span>A small vision model looks only at the picture and says what it thinks the page is. Disagreement with the sorter is flagged. Downloads about 46 MB once.</span></div>
          </label>
          {([
            ['magic', 'Magic scan', 'Finds the page in a photo, cuts away the background, flattens it and evens out shadows.'],
            ['deskew', 'Straighten', 'Finds the tilt of the text lines and levels the page.'],
            ['binarise', 'Black and white', 'One threshold for flat scans, a local one when the light is uneven.'],
            ['removeLines', 'Remove ruled lines', 'Erases form lines and table borders that confuse the reader.'],
          ] as ['magic' | 'deskew' | 'binarise' | 'removeLines', string, string][]).map(([k, t, d]) => (
            <label className="toggle" key={k}>
              <input type="checkbox" checked={!!p.pre[k]} disabled={k === 'removeLines' && !p.pre.binarise} onChange={(e) => p.setPre({ ...p.pre, [k]: e.target.checked })} />
              <div><b>{t}</b><span>{d}</span></div>
            </label>
          ))}
        </details>
      </div>
    </div>
  );
}
