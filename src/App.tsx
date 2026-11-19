import { useCallback, useEffect, useRef, useState } from 'react';
import { AnimatePresence, MotionConfig, motion } from 'motion/react';
import { applyTemplate, criticLook, deleteUserTemplate, loadCritic, loadPpReader, loadSecondReader, processDocument, reanalyse, retype, saveUserTemplate, secondRead, warmUp } from './engine/engine';
import type { Template } from './core/template';
import { downloadReport } from './ui/report';
import { DEFAULT_PRE, type PreprocessOptions } from './core/preprocess';
import type { DocType } from './core/classify';
import { ScanPage, type Active, type SecondReaderState } from './ui/ScanPage';
import { Sorted } from './ui/Sorted';
import { ResultView } from './ui/ResultView';
import { ModelCard } from './ui/ModelCard';
import { Help } from './ui/Help';
import { IChart, IHelp, IList, IScan } from './ui/icons';
import { download, exportCsv, exportFunsd, exportJson, markMrz, needsReview, toEditable, type DocState, type EField } from './ui/state';

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
  const [queue, setQueue] = useState<{ blob: Blob; name: string }[]>([]);
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
        const nd: DocState = { ...d, doc: { ...d.doc, second: regions, secondMs: ms }, analysis, second: 'done', fields: toEditable(analysis.fields, d.fields) };
        return { ...nd, fields: markMrz(nd) };
      });
    } catch (e: any) {
      patch(s.doc.id, (d) => ({ ...d, second: 'error', secondError: String(e.message ?? e) }));
    }
  }, []);

  // one document at a time, in the order they were dropped
  useEffect(() => {
    if (active || !queue.length) return;
    const [next, ...rest] = queue;
    setQueue(rest);
    setActive({ name: next.name, steps: [] });
    processDocument(next.blob, next.name, pre, (steps) => setActive((a) => a && { ...a, steps }), (live) => setActive((a) => a && { ...a, ...live }))
      .then((doc) => {
        const st: DocState = { doc, analysis: doc.analysis, fields: toEditable(doc.analysis.fields), second: 'none' };
        st.fields = markMrz(st);
        setDocs((ds) => [st, ...ds]);
        if (secondRef.current.enabled) runSecond(st);
        if (criticRef.current) criticLook(doc).then((vision) => patch(doc.id, (d) => ({ ...d, vision }))).catch((e) => console.warn('critic', e));
      })
      .catch((e) => {
        console.error(e);
        setToast(`Could not read ${next.name}: ${e.message ?? e}`);
      })
      .finally(() => setActive(null));
  }, [queue, active, pre, runSecond]);

  const current = docs.find((d) => d.doc.id === open);
  const totalReview = docs.reduce((a, d) => a + needsReview(d.fields), 0);
  const openDoc = (id: string) => { setOpen(id); setSelected(null); setTab('sorted'); window.scrollTo({ top: 0 }); };
  const onRetype = async (t: DocType) => {
    if (!current) return;
    const doc = await retype(current.doc, t, current.doc.second);
    patch(doc.id, (d) => { const nd = { ...d, doc, analysis: doc.analysis, fields: toEditable(doc.analysis.fields) }; return { ...nd, fields: markMrz(nd) }; });
    setToast('Sorted again');
  };

  const onTeach = (t: Template) => {
    if (!current) return;
    saveUserTemplate(t);
    const doc = applyTemplate(current.doc, t);
    if (!doc) { setToast('Template saved. It did not line up with this page well enough to read it.'); return; }
    patch(doc.id, (d) => ({ ...d, doc, analysis: doc.analysis, fields: toEditable(doc.analysis.fields) }));
    setToast(`Template "${t.name}" saved. The next copy of this form will be read with it.`);
  };
  const onForgetTemplate = async (id: string) => {
    if (!current) return;
    deleteUserTemplate(id);
    const doc = await retype(current.doc, 'form', current.doc.second);
    patch(doc.id, (d) => ({ ...d, doc: { ...doc, classification: current.doc.classification }, analysis: doc.analysis, fields: toEditable(doc.analysis.fields) }));
    setToast('Template forgotten');
  };

  const nav: [Tab, string, typeof IScan][] = [['scan', 'Scan', IScan], ['sorted', 'Trays', IList], ['model', 'Model card', IChart], ['help', 'Help', IHelp]];

  return (
    <MotionConfig reducedMotion="user">
      <div className="app">
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
            <motion.div key={tab + (tab === 'sorted' ? open ?? '' : '')} initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} transition={{ duration: 0.14 }}>
              {tab === 'scan' && (
                <ScanPage pre={pre} setPre={setPre} second={second} toggleSecond={toggleSecond} critic={critic} setCritic={setCritic} active={active} queue={queue.length} recent={docs}
                  onFiles={(fs) => setQueue((q) => [...q, ...fs])} onOpen={openDoc} onSorted={() => { setOpen(null); setTab('sorted'); }} />
              )}
              {tab === 'sorted' && !current && (
                <Sorted docs={docs} onOpen={openDoc} onScan={() => setTab('scan')} onExportCsv={() => download('formflow-export.csv', exportCsv(docs), 'text/csv')} onReport={() => downloadReport(docs)} />
              )}
              {tab === 'sorted' && current && (
                <ResultView
                  s={current}
                  selected={selected}
                  onSelect={setSelected}
                  onBack={() => setOpen(null)}
                  onRetype={onRetype}
                  onChange={(fields: EField[]) => patch(current.doc.id, (d) => ({ ...d, fields }))}
                  onExportJson={() => download(current.doc.fileName.replace(/\.[^.]+$/, '') + '.json', JSON.stringify(exportJson(current), null, 2), 'application/json')}
                  onCopy={() => navigator.clipboard.writeText(JSON.stringify(exportJson(current), null, 2)).then(() => setToast('Copied'))}
                  onExportLabelled={() => download(current.doc.fileName.replace(/\.[^.]+$/, '') + '.labels.json', JSON.stringify(exportFunsd(current), null, 1), 'application/json')}
                  onReport={() => downloadReport([current], current.doc.fileName.replace(/\.[^.]+$/, '') + '-report.pdf')}
                  criticOn={critic}
                  onSecond={() => runSecond(current)}
                  secondAvailable={second.enabled && second.status !== 'error'}
                  onTeach={onTeach}
                  onForgetTemplate={onForgetTemplate}
                  onRemove={() => { setDocs((ds) => ds.filter((d) => d !== current)); setOpen(null); }}
                />
              )}
              {tab === 'model' && <ModelCard />}
              {tab === 'help' && <Help />}
            </motion.div>
          </AnimatePresence>
        </main>
        <AnimatePresence>{toast && <motion.div className="toast" role="status" initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: 16 }}>{toast}</motion.div>}</AnimatePresence>
      </div>
    </MotionConfig>
  );
}
