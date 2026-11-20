// The project explained from A to Z in plain words: the problem, how each part of it is answered,
// and for every piece of the system what was used, why, how it works and what was measured.
// The figures are the ones in the tables further down the model card (measured 20 Nov 2026);
// when a model is retrained, update them here as well.

const PROBLEM: { pain: string; answer: string; proof: string }[] = [
  { pain: 'Retyping forms and IDs by hand is slow',
    answer: 'Take a photo; the details come out as text you can copy with one tap or save as CSV, JSON, a report or a searchable PDF.',
    proof: 'A clean page is read in a few seconds on a laptop. Phone timing has not been measured properly yet.' },
  { pain: 'Typing mistakes get applications rejected',
    answer: 'Numbers and dates are checked by arithmetic (check digits), read by two readers, and anything doubtful is shown to a person before it is saved.',
    proof: 'Key facts right on held-back ID specimens: 94% with two readers, 87% with one. No value marked "verified" was wrong.' },
  { pain: 'Uploading IDs to online tools is a privacy risk',
    answer: 'Every model runs inside the browser on your own device. The page works offline and sends no document anywhere.',
    proof: 'Open the browser network tab while reading a page: no upload. The models are downloaded once and cached.' },
];

interface Part { name: string; what: string; why: string; how: string; result: string }
const PARTS: Part[] = [
  { name: 'A. Where it runs',
    what: 'A web app (React, installable on a phone) with all models compiled to run in the browser through WebAssembly and ONNX Runtime Web.',
    why: 'The whole point is that an ID never leaves the device. It also means no server, no account and no cost.',
    how: 'Heavy work runs in background workers so the screen stays smooth. Models are downloaded once and kept on the device.',
    result: 'Total download for the default path is small enough for a phone; the optional models load only when switched on.' },
  { name: 'B. Finding the document in the photo',
    what: 'Two finders. An outline finder (colour edges, then the largest closed shape with four straight sides) and a bright-page finder (the largest light region the dark surface cannot reach).',
    why: 'If the card is small and tilted in the photo, every later step fails. The first version only found light pages on dark tables, so a pale card on a white table was not cut out at all.',
    how: 'The same finder runs on a small copy of each camera frame, about nine times a second, and draws the outline live before you take the picture. The corners seen at that moment are used for the cut.',
    result: 'Made-up cards pasted on white, cream, grey, wood and black surfaces: 95 of 100 found, against 40 of 100 for the bright-page finder alone. It cut nothing out of 70 plain form scans.' },
  { name: 'C. Cleaning the picture',
    what: 'Classical computer vision, no model: perspective warp to a flat rectangle, lighting correction, straightening, black-and-white threshold (Otsu or Sauvola), ruled-line removal.',
    why: 'Readers are trained on flat, evenly lit, upright text. Clean-up is cheap, fast and explainable, and it helps every reader after it.',
    how: 'Four corners define a 3 by 3 perspective transform. Paper brightness around each pixel is estimated and divided out. The page is tried at every quarter degree to find the angle where text lines are sharpest.',
    result: 'Character error with Tesseract on the test forms: 25.8% without clean-up, 20.4% with it.' },
  { name: 'D. Reading the text',
    what: 'Two open readers: Tesseract (classical, fast, default) and PP-OCRv5 mobile (a small neural network, slower, more accurate).',
    why: 'One is fast, the other is accurate. Using both also gives a check: when two different readers see the same value it is far more likely to be right.',
    how: 'Tesseract reads first. A grainy page, or one it has not finished in 20 seconds, goes to PP-OCRv5. On identity documents PP-OCRv5 reads again afterwards and the key facts from both are compared.',
    result: 'Character error on the test forms: Tesseract 20.4%, PP-OCRv5 13.1%. Key facts on ID specimens: 87% with one reader, 94% with two.' },
  { name: 'E. Code lines and check digits',
    what: 'The lines of "<<<" on passports and ID cards (ICAO 9303), read from several crops, with the check digits recomputed. The Emirates ID number is tested with the Luhn checksum.',
    why: 'A check digit is arithmetic. If it passes, the number is right, whatever any model thought. It is the only place where the app can say "verified".',
    how: 'Each attempt is parsed; the one with the most passing check digits wins. The name from the code line is compared with the printed name.',
    result: 'Real passport and ID pages: every check digit passes on 12 of 14 with the neural reader first, 4 of 14 with the classical reader.' },
  { name: 'F. Sorting by document type',
    what: 'A small learned classifier over 31 yes-or-no clues (softmax), plus a picture-only critic (MobileCLIP) as a second opinion.',
    why: 'The type decides what to look for. Clues are readable by a person, so every decision can be explained on screen ("Why driving licence?").',
    how: 'Weights are fitted on real and made-up documents and pulled towards hand-set values, so a handful of examples cannot swing them.',
    result: 'ID specimens sorted right: 107 of 108 with learned weights, 99 of 108 with hand rules.' },
  { name: 'G. Understanding a form',
    what: 'Two gradient-boosted tree models exported to ONNX. The first labels each block of text (label, value, heading, other); the second decides which value belongs to which label.',
    why: 'The public dataset (FUNSD) has only 149 training forms: too few for a large layout network, enough for trees on hand-made features. Trees are a few hundred kilobytes, run in milliseconds in a browser, and beat both baselines.',
    how: '82 features per block and 21 per label-value pair (position, size, distance, alignment, text shape). 300 trees of depth 4. Baselines: hand rules, and logistic regression on the same features.',
    result: 'Linking F1 on unseen forms: hand rules 70.4%, logistic regression 81.1%, trees 91.2%. Train against test: labeller 91.6% and 80.3%, linker 92.6% and 88.0%.' },
  { name: 'H. Templates for modern forms',
    what: 'A known form is recognised by its printed words and each box is read directly. A person can teach a new form once.',
    why: 'An honest failure: the tree models were trained on 1990s faxes and found only 7.7% of fields on modern forms. Rather than hide it, a different method was added for forms that repeat.',
    how: 'The printed words are matched to a stored layout, the page is aligned to it, and every box is cropped and read on its own.',
    result: 'Fields right on 20 modern forms: 7.7% without a template, 74.9% with one, 93.5% with a template and the neural reader.' },
  { name: 'I. The generative AI experiment',
    what: 'Florence-2, a small generative vision-language model, used as an optional second reader on forms.',
    why: 'The course asks for a generative experiment with a measured effect. The question was: does a generative reader help?',
    how: 'It reads the same regions; its text is compared word by word with the first reader.',
    result: 'As a reader it is worse. As a witness it is useful: when the two agree the word is right 93% of the time, when they disagree 48%. So it is used to flag, not to read.' },
  { name: 'J. How sure, and when to ask a person',
    what: 'A confidence per field, passed through a calibration curve, with a threshold of 80% below which a person is asked.',
    why: 'A confident wrong answer is the dangerous case. The number shown has to mean what it says.',
    how: 'Reader certainty and link certainty are combined, then mapped with a curve fitted on held-out forms so that "80%" is right about 80% of the time.',
    result: 'Calibration error: 42.1% for the raw score, 4.6% after calibration.' },
  { name: 'K. Review and save',
    what: 'A review screen that shows one doubtful value at a time next to its piece of the picture, then copy and export.',
    why: 'The person stays in charge. Nothing doubtful is saved without being looked at, and unconfirmed values are marked "review" in every export.',
    how: 'Exports: JSON, CSV, a PDF report, a searchable scan PDF, a ZIP sorted by type, and a redacted copy with black boxes.',
    result: 'On an identity document only real details are listed. If nothing could be read, the app says so instead of listing noise.' },
  { name: 'L. Limits, stated plainly',
    what: 'Specimens are made up, so their scores are an upper bound. There is no field-level score on real IDs. English and Latin letters only. A card must sit on a plain surface to be outlined.',
    why: 'Real IDs cannot be published or labelled in bulk for privacy reasons, so made-up specimens with known answers stand in for them.',
    how: 'Real documents are used only on the owner\'s machine and only counts are published.',
    result: 'The live outline and the reading speed on a real phone have not been measured; they were tested with a simulated camera on a laptop.' },
];

export function Explain() {
  return (
    <div className="explain">
      <div className="xp-problem">
        {PROBLEM.map((p) => (
          <div key={p.pain}>
            <span className="xp-tag">Problem</span><b>{p.pain}</b>
            <span className="xp-tag ok">What FormFlow does</span><p>{p.answer}</p>
            <span className="xp-tag">Evidence</span><p className="muted">{p.proof}</p>
          </div>
        ))}
      </div>
      <p className="small" style={{ margin: '1rem 0 0.5rem' }}><b>The idea in one line:</b> read on the device, check with arithmetic and a second reader, and ask a person only about the doubtful parts.</p>
      {PARTS.map((p) => (
        <details className="fold xp-part" key={p.name}>
          <summary><span className="grow">{p.name}</span></summary>
          <dl>
            <dt>What we used</dt><dd>{p.what}</dd>
            <dt>Why</dt><dd>{p.why}</dd>
            <dt>How it works</dt><dd>{p.how}</dd>
            <dt>What we measured</dt><dd><b>{p.result}</b></dd>
          </dl>
        </details>
      ))}
    </div>
  );
}
