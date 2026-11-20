import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { AnimatePresence, MotionConfig, motion } from 'motion/react';
import { applyTemplate, criticLook, deleteUserTemplate, loadCritic, loadPpReader, loadSecondReader, processDocument, reanalyse, retype, secondLook, factsOf, saveUserTemplate, secondRead, warmUp } from './engine/engine';
import type { Template } from './core/template';
import type { Quad } from './core/scan';
import { downloadReport } from './ui/report';
import { downloadSearchablePdf, downloadTypePdf, downloadZip, niceName } from './ui/bundle';
import { DEFAULT_PRE, type PreprocessOptions } from './core/preprocess';
import type { DocType } from './core/classify';
import { SAMPLE_COUNT, ScanPage, sampleFiles, type Active, type SecondReaderState } from './ui/ScanPage';
import { Tour, type TourStep } from './ui/Tour';
import { Backdrop } from './ui/Backdrop';
import { Sorted } from './ui/Sorted';
import { ResultView } from './ui/ResultView';
import { ModelCard } from './ui/ModelCard';
import { Help } from './ui/Help';
import { IChart, IHelp, IList, IScan } from './ui/icons';
import { download, exportCsv, exportFunsd, exportJson, markMrz, needsReview, sidesOf, shown, toEditable, type DocState, type EField } from './ui/state';

type Tab = 'scan' | 'sorted' | 'model' | 'help';

const load = <T,>(k: string, d: T): T => {
  try { const v = localStorage.getItem(k); return v ? JSON.parse(v) : d; } catch { return d; }
};
const save = (k: string, v: unknown) => { try { localStorage.setItem(k, JSON.stringify(v)); } catch {} };

export default function App() {
  const [tab, setTab] = useState<Tab>('scan');
  const [pre, setPre] = useState<PreprocessOptions>(() => ({ ...DEFAULT_PRE, ...load('ff.pre', {}) }));
  const [critic, setCritic] = useState<boolean>(() => load('ff.critic', true));
  const criticRef = useRef(critic);
  criticRef.current = critic;
  useEffect(() => { save('ff.critic', critic); if (critic) loadCritic().catch(() => {}); }, [critic]);
  const [second, setSecond] = useState<SecondReaderState>(() => ({ enabled: load('ff.second', false), status: 'off' }));
  const [docs, setDocs] = useState<DocState[]>([]);
  const [open, setOpen] = useState<string | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [active, setActive] = useState<Active | null>(null);
  const [queue, setQueue] = useState<{ blob: Blob; name: string; quad?: Quad }[]>([]);
  const [toast, setToast] = useState('');
  const secondRef = useRef(second);
  secondRef.current = second;
  const docsRef = useRef(docs);
  docsRef.current = docs;

  useEffect(() => { warmUp(); }, []);
  useEffect(() => { save('ff.pre', pre); if (pre.reader === 'ppocr') loadPpReader().catch(() => {}); }, [pre]);
  useEffect(() => { if (toast) { const t = setTimeout(() => setToast(''), 2600); return () => clearTimeout(t); } }, [toast]);

  const startSecond = useCallback(() => {
    setSecond((s) => ({ ...s, status: 'loading' }));
    loadSecondReader((loaded, total) => setSecond((s) => ({ ...s, loaded, total })), (device) => setSecond((s) => ({ ...s, device })))
      .then((device) => setSecond((s) => ({ ...s, status: 'ready', device })))
      .catch((e) => setSecond((s) => ({ ...s, status: 'error', error: String(e.message ?? e) })));
  }, []);
  useEffect(() => { if (second.enabled && second.status === 'off') startSecond(); }, [second.enabled, second.status, startSecond]);
  const toggleSecond = (on: boolean) => { save('ff.second', on); setSecond((s) => ({ ...s, enabled: on })); };

  const patch = (id: string, fn: (s: DocState) => DocState) => setDocs((ds) => ds.map((d) => (d.doc.id === id ? fn(d) : d)));

  const runSecond = useCallback(async (s: DocState) => {
    patch(s.doc.id, (d) => ({ ...d, second: 'running' }));
    try {
      await loadSecondReader(() => {}, () => {});
      const { regions, ms } = await secondRead(s.doc);
      // read the latest document state: the clerk may have changed its type meanwhile
      const latest = docsRef.current.find((d) => d.doc.id === s.doc.id) ?? s;
      const analysis = await reanalyse(latest.doc, regions);
      patch(s.doc.id, (d) => {
        const nd: DocState = { ...d, doc: { ...d.doc, second: regions, secondMs: ms }, analysis, second: 'done', fields: toEditable(shown(d.doc, analysis.fields), d.fields) };
        return { ...nd, fields: markMrz(nd) };
      });
    } catch (e: any) {
      patch(s.doc.id, (d) => ({ ...d, second: 'error', secondError: String(e.message ?? e) }));
    }
  }, []);

  // Identity documents get a second look by the neural reader. It shares that reader with the
  // code-line step, so it waits until the pile has been read and then takes one document at a time.
  const looks = useRef<DocState['doc'][]>([]);
  const looking = useRef(false);
  const [lookTick, setLookTick] = useState(0);
  const lookAgain = useCallback((doc: DocState['doc']) => { if (doc.kind === 'id') { looks.current.push(doc); setLookTick((t) => t + 1); } }, []);
  useEffect(() => {
    if (active || queue.length || looking.current) return;
    const doc = looks.current.shift();
    if (!doc) return;
    looking.current = true;
    secondLook(doc)
      .then((r) => { if (r) patch(doc.id, (d) => (d.doc.kind === 'id' ? { ...d, doc: { ...d.doc, words2: r.words2, words2Ms: r.ms, facts: factsOf(d.doc, d.analysis.fields, r.words2) } } : d)); })
      .catch((e) => console.warn('second look', e))
      .finally(() => { looking.current = false; setLookTick((t) => t + 1); });
  }, [active, queue, lookTick]);

  // one document at a time, in the order they were dropped
  useEffect(() => {
    if (active || !queue.length) return;
    const [next, ...rest] = queue;
    setQueue(rest);
    setActive({ name: next.name, steps: [] });
    processDocument(next.blob, next.name, pre, (steps) => setActive((a) => a && { ...a, steps }), (live) => setActive((a) => a && { ...a, ...live }), undefined, 0, next.quad)
      .then((doc) => {
        const st: DocState = { doc, analysis: doc.analysis, fields: toEditable(shown(doc, doc.analysis.fields)), second: 'none' };
        st.fields = markMrz(st);
        setDocs((ds) => [st, ...ds]);
        lookAgain(doc);
        // Florence-2 was measured on forms; identity documents get the neural reader's second look instead
        if (secondRef.current.enabled && doc.kind !== 'id') runSecond(st);
        if (criticRef.current) criticLook(doc).then((vision) => patch(doc.id, (d) => ({ ...d, vision }))).catch((e) => console.warn('critic', e));
      })
      .catch((e) => {
        console.error(e);
        setToast(`Could not read ${next.name}: ${e.message ?? e}`);
      })
      .finally(() => setActive(null));
  }, [queue, active, pre, runSecond, lookAgain]);

  const current = docs.find((d) => d.doc.id === open);
  const totalReview = docs.reduce((a, d) => a + needsReview(d.fields), 0);
  const sides = useMemo(() => sidesOf(docs), [docs]);
  const openDoc = (id: string) => { setOpen(id); setSelected(null); setTab('sorted'); window.scrollTo({ top: 0 }); };
  const onRetype = async (t: DocType) => {
    if (!current) return;
    const doc = await retype(current.doc, t, current.doc.second);
    patch(doc.id, (d) => { const nd = { ...d, doc, analysis: doc.analysis, fields: toEditable(shown(doc, doc.analysis.fields)) }; return { ...nd, fields: markMrz(nd) }; });
    setToast('Sorted again');
  };

  const onTeach = (t: Template) => {
    if (!current) return;
    saveUserTemplate(t);
    const doc = applyTemplate(current.doc, t);
    if (!doc) { setToast('Template saved. It did not line up with this page well enough to read it.'); return; }
    patch(doc.id, (d) => ({ ...d, doc, analysis: doc.analysis, fields: toEditable(shown(doc, doc.analysis.fields)) }));
    setToast(`Template "${t.name}" saved. The next copy of this form will be read with it.`);
  };
  const onCorners = async (q: Quad) => {
    if (!current) return;
    const old = current.doc.id;
    setToast('Reading the page again…');
    try {
      const doc = await processDocument(current.doc.source, current.doc.fileName, pre, () => {}, undefined, q, current.doc.turned);
      const st: DocState = { doc, analysis: doc.analysis, fields: toEditable(shown(doc, doc.analysis.fields)), second: 'none' };
      st.fields = markMrz(st);
      setDocs((ds) => ds.map((d) => (d.doc.id === old ? st : d)));
      lookAgain(doc);
      setOpen(doc.id);
      setSelected(null);
      setToast('Read again with your corners');
      if (criticRef.current) criticLook(doc).then((vision) => patch(doc.id, (d) => ({ ...d, vision }))).catch(() => {});
    } catch (e: any) {
      setToast(`Could not read it again: ${e.message ?? e}`);
    }
  };
  const onForgetTemplate = async (id: string) => {
    if (!current) return;
    deleteUserTemplate(id);
    const doc = await retype(current.doc, 'form', current.doc.second);
    patch(doc.id, (d) => ({ ...d, doc: { ...doc, classification: current.doc.classification }, analysis: doc.analysis, fields: toEditable(shown(doc, doc.analysis.fields)) }));
    setToast('Template forgotten');
  };

  // ---- guided tour on the sample scans ----
  const [tour, setTour] = useState<number | null>(null);
  const activeRef = useRef(active);
  activeRef.current = active;
  const queueRef = useRef(queue);
  queueRef.current = queue;
  const tourSteps = useMemo<TourStep[]>(() => {
    const all = () => docsRef.current;
    const show = (pick: (d: DocState) => boolean) => () => { const d = all().find(pick); if (d) openDoc(d.doc.id); else { setOpen(null); setTab('sorted'); } };
    const busy = () => !!activeRef.current || queueRef.current.length > 0;
    return [
      { title: 'A 60-second tour', body: `FormFlow reads a mixed pile of scans on this device and sorts it. The tour uses ${SAMPLE_COUNT} made-up sample documents: no real person appears, and nothing leaves your browser.`, before: () => { setOpen(null); setTab('scan'); } },
      {
        title: 'Reading the pile', target: '.live',
        body: 'Each page is opened, straightened, read and stamped with its type. A phone photo is cut out of its background first. All of it runs here, in the browser.',
        before: () => { setOpen(null); setTab('scan'); if (!all().length && !busy()) sampleFiles().then((f) => setQueue((q) => [...q, ...f])).catch(() => setToast('Could not load the samples')); },
        ready: () => all().length > 0 && !busy(),
        waiting: () => `Reading the samples: ${all().length} of ${all().length + queueRef.current.length + (activeRef.current ? 1 : 0)} done`,
      },
      { title: 'Sorted into trays', target: '.packet', body: 'Five of the documents carry the same name, so FormFlow compares them. Here it caught a date of birth and a name spelling that do not match across documents: the kind of slip that gets an application rejected.', before: () => { setOpen(null); setTab('sorted'); window.scrollTo({ top: 0 }); } },
      { title: 'How sure is it?', target: '.trust', body: 'Every document gets a trust score. On a passport the long code lines at the bottom carry check digits, so the number, birth date and expiry are verified by arithmetic, not by a guess.', before: show((d) => d.doc.docType === 'passport') },
      { title: 'A person checks the doubtful ones', target: '.fields', body: 'Doubtful fields come first, each with a close-up from the page. Nothing is accepted on your behalf: you fix the text or tick it. On forms the percentage is calibrated, so 60% means right about 60% of the time.', before: show((d) => d.doc.docType === 'passport') },
      { title: 'A form it already knows', target: '.tplnote', body: 'This tax form matched a built-in template, so every box is read directly and keeps its proper label. You can teach it your own forms with "Teach this form".', before: show((d) => !!d.doc.template) },
      { title: 'Ticks, signatures and privacy tools', target: '.viewer', body: 'Checkboxes are read as ticked or not, and a signature line as signed or not. Below the page: "Redact" blacks out numbers and the photo on a copy, "Photo" lets you fix the page corners, and "Before / after" shows what the clean-up did.', before: show((d) => !!d.doc.template) },
      { title: 'Measured, including the failures', target: '.summary', body: 'The Model card reports every result from the evaluation scripts: reading accuracy, calibration before and after, the 20-form test set, and where the system fails. Nothing here is a promise it cannot back.', before: () => { setOpen(null); setTab('model'); window.scrollTo({ top: 0 }); } },
      { title: 'Your turn', body: 'Drop in your own scans or PDFs, or take a photo. Use only documents you have the right to use. Everything stays on this device until you export it.', before: () => { setOpen(null); setTab('scan'); window.scrollTo({ top: 0 }); } },
    ];
  }, []); // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => { if (new URLSearchParams(location.search).has('tour')) setTour(0); }, []);

  const nav: [Tab, string, typeof IScan][] = [['scan', 'Scan', IScan], ['sorted', 'Trays', IList], ['model', 'Model card', IChart], ['help', 'Help', IHelp]];

  return (
    <MotionConfig reducedMotion="user">
      <div className="app">
        <Backdrop busy={!!active} />
        <header className="topbar">
          <div className="brand"><img src={import.meta.env.BASE_URL + 'icon.svg'} alt="" />FormFlow</div>
          <nav className="nav" aria-label="Main">
            {nav.map(([t, label, Icon]) => (
              <button key={t} className={tab === t ? 'active' : ''} onClick={() => { setTab(t); if (t === 'sorted') setOpen(null); }} aria-current={tab === t} aria-label={label}>
                {tab === t && <motion.span layoutId="pill" className="pill" transition={{ type: 'spring', stiffness: 420, damping: 34 }} />}
                <Icon /><span className="lbl">{label}</span>{t === 'sorted' && totalReview > 0 && <span className="count">{totalReview}</span>}
              </button>
            ))}
          </nav>
        </header>
        <main className="main">
          <AnimatePresence mode="wait" initial={false}>
            <motion.div key={tab + (tab === 'sorted' ? open ?? '' : '')} initial={{ opacity: 0, y: 18 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -12 }} transition={{ duration: 0.2, ease: [0.2, 0.8, 0.2, 1] }}>
              {tab === 'scan' && (
                <ScanPage pre={pre} setPre={setPre} second={second} toggleSecond={toggleSecond} critic={critic} setCritic={setCritic} active={active} queue={queue.length} recent={docs}
                  onFiles={(fs) => setQueue((q) => [...q, ...fs])} onOpen={openDoc} onSorted={() => { setOpen(null); setTab('sorted'); }} onTour={() => setTour(0)} />
              )}
              {tab === 'sorted' && !current && (
                <Sorted docs={docs} onOpen={openDoc} onScan={() => setTab('scan')} onExportCsv={() => download('formflow-export.csv', exportCsv(docs), 'text/csv')} onReport={() => downloadReport(docs)}
                  onZip={() => { setToast('Packing the ZIP…'); downloadZip(docs).then(() => setToast('ZIP downloaded')).catch((e) => setToast(`Could not build the ZIP: ${e.message ?? e}`)); }}
                  onTypePdf={(t) => downloadTypePdf(docs, t).catch((e) => setToast(`Could not build the PDF: ${e.message ?? e}`))}
                  onCopyText={(text, what) => navigator.clipboard?.writeText(text).then(() => setToast(`${what} copied`)).catch(() => setToast('Could not copy'))} />
              )}
              {tab === 'sorted' && current && (
                <ResultView
                  s={current}
                  selected={selected}
                  onSelect={setSelected}
                  onBack={() => setOpen(null)}
                  side={sides.get(current.doc.id)}
                  onOpenOther={() => { const o = sides.get(current.doc.id)?.other; if (o) openDoc(o.doc.id); }}
                  onRetype={onRetype}
                  onChange={(fields: EField[]) => patch(current.doc.id, (d) => ({ ...d, fields }))}
                  onExportJson={() => download(niceName(current) + '.json', JSON.stringify(exportJson(current), null, 2), 'application/json')}
                  onCopy={() => navigator.clipboard.writeText(JSON.stringify(exportJson(current), null, 2)).then(() => setToast('Copied'))}
                  onExportLabelled={() => download(current.doc.fileName.replace(/\.[^.]+$/, '') + '.labels.json', JSON.stringify(exportFunsd(current), null, 1), 'application/json')}
                  onReport={() => downloadReport([current], niceName(current) + ' - report.pdf')}
                  onScanPdf={() => downloadSearchablePdf(current, sides.get(current.doc.id)).catch((e) => setToast(`Could not build the PDF: ${e.message ?? e}`))}
                  criticOn={critic}
                  onSecond={() => runSecond(current)}
                  secondAvailable={second.enabled && second.status !== 'error'}
                  onTeach={onTeach}
                  onForgetTemplate={onForgetTemplate}
                  onCorners={onCorners}
                  onRemove={() => { setDocs((ds) => ds.filter((d) => d !== current)); setOpen(null); }}
                />
              )}
              {tab === 'model' && <ModelCard />}
              {tab === 'help' && <Help />}
            </motion.div>
          </AnimatePresence>
        </main>
        {tour !== null && <Tour steps={tourSteps} index={tour} onIndex={setTour} onClose={() => setTour(null)} />}
        <AnimatePresence>{toast && <motion.div className="toast" role="status" initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: 16 }}>{toast}</motion.div>}</AnimatePresence>
      </div>
    </MotionConfig>
  );
}
