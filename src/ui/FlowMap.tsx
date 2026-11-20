// The whole path of one page, from the photo to the exported answer, as a diagram a person can
// read on a phone: one column of steps joined by arrows, grouped into stages, each step saying
// what goes in, what comes out and what does the work (plain image processing, a rule, or a model).
// Tap a step for the detail. Everything here describes code in src/core and src/engine.
import { useState } from 'react';

type Kind = 'cv' | 'model' | 'rule' | 'person';
interface Node {
  n: string;
  title: string;
  kind: Kind;
  by: string; // what does the work
  out: string; // what comes out of the step
  more: string; // the detail shown when opened
}
interface Stage {
  name: string;
  note: string;
  nodes: (Node | [Node, Node])[]; // a pair is a fork: the page takes one of the two
}

const KIND: Record<Kind, string> = { cv: 'Image processing', model: 'Learned model', rule: 'Rule or arithmetic', person: 'You' };

const STAGES: Stage[] = [
  {
    name: 'Take in', note: 'on your device, nothing uploaded',
    nodes: [
      { n: '1', title: 'Open the file', kind: 'cv', by: 'Browser, pdf.js, fflate', out: 'one image per page, at most 2400 px on its long side',
        more: 'A photo, a scan, a PDF or a ZIP. PDFs become one image per page and a ZIP is unpacked (up to 80 pages). Large photos are shrunk so a phone does not run out of memory.' },
      { n: '2', title: 'Check the photo', kind: 'cv', by: 'Edge sharpness and burnt-out blocks', out: 'blurry? glare? If so: "take it again"',
        more: 'Sharpness is measured at the strongest 4% of edges; below 0.30 the photo counts as blurry. Glare is the share of pure white 18-pixel blocks with no texture; 1.2% or more counts as glare on cards and booklets.' },
    ],
  },
  {
    name: 'Clean', note: 'classical computer vision, no model',
    nodes: [
      { n: '3', title: 'Find the page', kind: 'cv', by: 'Bright region, hull, four sides', out: 'four corners of the page',
        more: 'The page is the largest bright region the darker surface cannot reach from the edge of the photo. Its outline is wrapped in a convex hull, the four long straight sides are found, and the corners are where the sides cross. That keeps the corners right even when a dark photo or a thumb eats into one of them. If the page is not lighter than the surface, the whole photo is kept and you can drag the corners yourself.' },
      { n: '4', title: 'Flatten and light', kind: 'cv', by: 'Perspective warp (homography), paper estimate', out: 'a flat, evenly lit page',
        more: 'The four corners define a 3 by 3 perspective transform; every pixel of an upright rectangle is sampled from the photo. Then the paper brightness around each pixel is estimated and divided out, so shadows turn white and ink stays dark.' },
      { n: '5', title: 'Straighten', kind: 'cv', by: 'Projection profile, -6 to +6 degrees', out: 'upright text lines',
        more: 'The page is tried at every quarter degree. At the right angle the rows alternate most sharply between text and gap.' },
      { n: '6', title: 'Black and white, lines off', kind: 'cv', by: 'Otsu or Sauvola threshold, line removal', out: 'a clean page for the fast reader, a grey page for the neural one',
        more: 'Evenly lit scans get one threshold for the whole page (Otsu); shadowed ones get a threshold around every pixel (Sauvola). Long ruled lines are erased. If the result is covered in specks (more than 16% black, typical of fabric or wood under the page), the page goes to the neural reader instead.' },
    ],
  },
  {
    name: 'Read', note: 'three readers with different strengths',
    nodes: [
      { n: '7', title: 'Read the text', kind: 'model', by: 'Tesseract (fast) or PP-OCRv5 (neural)', out: 'words, each with a box and a confidence',
        more: 'Tesseract reads the cleaned page in a second or two. A grainy page, or one Tesseract has not finished in 20 seconds, is read by PP-OCRv5 instead. If almost no clean words come out, the page is turned 90, 270 and 180 degrees and read again.' },
      { n: '8', title: 'Read the code lines', kind: 'model', by: 'PP-OCRv5 on crops, then ICAO 9303 check digits', out: 'number, birth date, expiry: "verified" when the digits add up',
        more: 'The lines of "<<<" at the bottom of passports and ID cards are read from several crops. Each attempt is parsed and the one with the most passing check digits wins. A check digit is arithmetic: no model can fake it.' },
    ],
  },
  {
    name: 'Understand', note: 'what is this page, and what does it say',
    nodes: [
      { n: '9', title: 'Sort by type', kind: 'model', by: '31 clues, learned weights, softmax', out: 'passport, Emirates ID, ID card, licence, visa, form or other, with a percentage',
        more: 'Each clue is a yes/no question about the page ("valid passport code lines", "the words DRIVING LICENCE"). The weights were fitted on real and made-up documents and pulled towards hand-set values. A known form is recognised by its printed words instead.' },
      [
        { n: '10a', title: 'A form', kind: 'model', by: 'Two gradient-boosted tree models, or a taught template', out: 'label and value pairs',
          more: 'Words are grouped into blocks. The first tree model says whether a block is a label or a value; the second says which value belongs to which label (91% F1 against 70% for hand rules). Tick boxes and signature lines are found from their shapes.' },
        { n: '10b', title: 'An identity document', kind: 'rule', by: 'Code lines, labels, layout, checksums', out: 'key facts: name, number, birth, expiry',
          more: 'Facts come from the code lines first, then printed labels (misread labels allowed), then layout. The Emirates ID number is tested with the Luhn checksum. A name from the code line is compared with the printed name.' },
      ],
    ],
  },
  {
    name: 'Check', note: 'how far can each value be trusted',
    nodes: [
      { n: '11', title: 'How sure?', kind: 'rule', by: 'Reader certainty, link certainty, calibration curve', out: 'a percentage per field; below 80% means "to check"',
        more: 'A field score mixes how sure the reader was and how sure the linker was. On forms it is passed through a calibration curve so that 80% is right about 80% of the time. Expired documents, failed check digits and a doubtful type raise alerts.' },
      { n: '12', title: 'Second opinions', kind: 'model', by: 'PP-OCRv5 second look, Florence-2, MobileCLIP critic', out: '"seen twice", "readers differ", "critic has doubts"',
        more: 'After the first answer is showing, the neural reader reads identity documents again and the key facts from both readings are compared. Florence-2 (optional, generative) is a third witness on forms. MobileCLIP looks only at the picture and says what kind of document it resembles.' },
      { n: '13', title: 'Across documents', kind: 'rule', by: 'Name matching, date and number comparison', out: 'one profile per person, mismatches in red, card fronts paired with backs',
        more: 'Documents with the same name are grouped. Birth date, name spelling and passport number are compared, and the most trustworthy value for each detail is kept.' },
    ],
  },
  {
    name: 'Decide and save', note: 'a person stays in charge',
    nodes: [
      { n: '14', title: 'You review', kind: 'person', by: 'One doubtful field at a time', out: 'confirmed or corrected values',
        more: 'Each doubtful field is shown with that part of the page enlarged. Correct, Skip or Not a field. Nothing doubtful is accepted without you.' },
      { n: '15', title: 'Export', kind: 'rule', by: 'jsPDF, fflate', out: 'JSON, CSV, report, searchable PDF, ZIP by type, redacted copy',
        more: 'Files are named after what was read. Scan PDFs carry the text invisibly on top of the picture so they can be searched. Values nobody confirmed are marked "review" in every export.' },
    ],
  },
];

function Box({ node, open, onToggle }: { node: Node; open: boolean; onToggle: () => void }) {
  return (
    <button className={'flow-node ' + node.kind + (open ? ' open' : '')} onClick={onToggle} aria-expanded={open}>
      <span className="flow-n">{node.n}</span>
      <span className="flow-body">
        <b>{node.title}</b>
        <span className="flow-by"><i className={'flow-kind ' + node.kind}>{KIND[node.kind]}</i>{node.by}</span>
        <span className="flow-out"><em>gives</em> {node.out}</span>
        {open && <span className="flow-more">{node.more}</span>}
      </span>
    </button>
  );
}

export function FlowMap() {
  const [open, setOpen] = useState<string | null>(null);
  const toggle = (n: string) => setOpen((o) => (o === n ? null : n));
  return (
    <div className="flow">
      <div className="flow-legend">
        {(Object.keys(KIND) as Kind[]).map((k) => <span key={k}><i className={'flow-kind ' + k}>{KIND[k]}</i></span>)}
        <span className="muted">Tap a step for the detail</span>
      </div>
      <div className="flow-start">Photo, scan, PDF or ZIP</div>
      {STAGES.map((s) => (
        <section className="flow-stage" key={s.name}>
          <header><b>{s.name}</b><span>{s.note}</span></header>
          {s.nodes.map((n) => Array.isArray(n)
            ? <div className="flow-fork" key={n[0].n}>{n.map((x) => <Box key={x.n} node={x} open={open === x.n} onToggle={() => toggle(x.n)} />)}</div>
            : <Box key={n.n} node={n} open={open === n.n} onToggle={() => toggle(n.n)} />)}
        </section>
      ))}
      <div className="flow-end">The answer, with its proof and its doubts</div>
    </div>
  );
}
