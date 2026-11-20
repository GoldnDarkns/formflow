# FormFlow: handover for Claude Code

This file is the full context for continuing the project. It was built in a Claude chat session on 18–19 November 2026 and has not yet been run on the owner's own machine.

## Who and what

- **Owner:** Madhavan Vij, MAIB student at SP Jain School of Global Management, Dubai.
- **Course:** MAIB AI 221 Computer Vision with Applications. This is the final project (30% of the unit), Project 9, Public sector: "FormFlow: scanned-form digitiser". Due in exam week; presentation in Week 12.
- **Problem:** offices retype paper forms and ID documents by hand. It is slow, typos cause rejected applications, and uploading IDs to cloud tools is a privacy risk.
- **Solution:** an installable web app (PWA) that reads a mixed pile of scans on the device: sorts each page by type, extracts label and value pairs, verifies what can be verified, shows how sure it is, and has a person confirm doubtful fields before export. Nothing is uploaded.

## Hard rules from the course brief (do not break these)

- No paid services, APIs, subscriptions or sign-ups. All models are open and run in the browser.
- Deploy on GitHub Pages (static). Cloudflare Pages is out because the ONNX Runtime wasm file is about 27 MB and Cloudflare limits files to 25 MiB.
- Keep each model file under about 100 MB. Target under about 5 seconds per image.
- The generative AI component (Florence-2) is required. Without it the Innovation criterion is capped.
- Required parts: problem statement, public dataset with licence (FUNSD, non-commercial), a baseline, the main model exported to ONNX, a generative AI experiment with a measured effect, evaluation with error analysis and subgroups, an interface with "not sure" states and a help page, a live URL, responsible AI (model card, privacy, bias, human in the loop).
- Deliverables: live URL, public GitHub repo, technical report (3,000–4,000 words), 1-page model card, 3–5 minute demo video with three real cases including one failure, 10-minute presentation.
- Rubric: Research depth 25%, Innovation 20%, Application of techniques 35%, Presentation and reporting 20%.

## How the owner wants to work

- For the report and any other assessed writing, the words must be his own: work one section at a time, explain what the section needs, he dictates, type his words exactly and fix spelling only. Do not write the report for him.
- He reads replies on a phone. Keep answers short and lead with the result.
- He wants a real, polished product with good animation, not a static demo.
- Be honest about what is tested and what is not.

## First tasks (nothing below has been done yet)

1. `npm install`, then `npm run dev` and check the app loads and the 9 sample scans sort correctly ("Try 9 sample scans"; the ninth is a filled W-9 that shows template mode).
2. `git init -b main`, commit, create a **public** GitHub repo and push (`gh repo create formflow --public --source . --push`).
3. On GitHub: Settings → Pages → Source: GitHub Actions. The workflow in `.github/workflows/deploy.yml` builds and publishes on every push to `main`. Confirm the live URL works on a phone.
4. Test with the owner's real documents (his own, or with consent). Keep them outside the repo in `../data/own/<type>/`. Never commit them.

## Commands

```bash
npm install
npm run dev            # local app
npm run build          # production build in dist/ (runs scripts/copy-assets.mjs first)
npm run preview        # serve dist/ on port 4173
node scripts/e2e.mjs   # headless Chrome test of the built app (needs preview running; edit the Chrome path for Windows)
npm run evaluate       # writes public/eval/results.json, shown on the Model card tab
npm test               # unit tests for src/core (also run by the deploy workflow before the build)
```

Training and evaluation need data that is **not in the repo**. All scripts expect a sibling folder `../data`:

```
../data/funsd/dataset/{training_data,testing_data}/{images,annotations}   # from https://guillaumejaume.github.io/FUNSD/dataset.zip
../data/ocr/<split>-<mode>/        # cached reader output, made by the scripts below
../data/models/                    # local copies of Hugging Face models for Node scripts
../data/ppocr/{det.onnx,rec.onnx,dict.txt}   # copy from public/models/ppocr
../data/photos/                    # simulated desk photos for the magic-scan test
../data/own/<type>/                # the owner's labelled documents
```

Full retraining order:

```bash
npx tsx scripts/ocr-funsd.ts training full && npx tsx scripts/ocr-funsd.ts testing full   # Tesseract (also modes raw, upscale, deskew, nolines)
npx tsx scripts/ocr-funsd-pp.ts training pp 1600 && npx tsx scripts/ocr-funsd-pp.ts testing pp 1600   # PP-OCRv5
npx tsx scripts/build-dataset.ts      # features from ground truth + both readers
python scripts/train.py               # needs scikit-learn, skl2onnx, onnxruntime, onnx, joblib
npx tsx scripts/florence-funsd.ts 1x2 # Florence-2 on the test set (slow on CPU; needs ../data/models/onnx-community/Florence-2-base-ft)
npx tsx scripts/critic-embed.ts       # critic text embeddings + critic accuracy (needs ../data/models/Xenova/mobileclip_s0)
npx tsx scripts/magic-eval.ts         # magic-scan experiment (needs ../data/photos)
npx tsx scripts/evaluate.ts
npx tsx scripts/calibrate.ts train    # fit confidence curves on the training split -> src/core/calibration.json + results.json
python scripts/make-forms20.py        # rebuild testsets/forms20 (needs ../data/forms20/blank/*.pdf, PyMuPDF)
npx tsx scripts/eval-forms20.ts       # score the 20 filled forms -> results.json
npx tsx scripts/eval-templates.ts     # teach templates from the blank forms, score, write public/templates/library.json
npx tsx scripts/eval-marks.ts         # checkboxes and signatures on the 20 forms (run after eval-templates)
npx tsx scripts/eval-own.ts ../data/own   # per-type scores on the owner's labelled documents
```

The simulated desk photos and the specimen images were generated with inline Python (PIL) during the chat session and those generator scripts are not in the repo. Recreate them if needed: warp 20 FUNSD test pages in perspective onto a dark cluttered 1300×1700 background, multiply by a diagonal shadow, save corners to `corners.json`.

## Architecture

```
src/core/         pure TypeScript, shared by the browser and the Node scripts (no DOM)
  preprocess.ts   deskew (projection profile), Otsu/Sauvola (auto-chosen), line removal, resize
  scan.ts         magic scan: page detection, homography warp, lighting flattening
  ocr.ts          Tesseract result parsing, CER / hallucination / omission scoring
  ppocr.ts        PP-OCRv5 reader: DB post-processing, batched CTC recognition
  layout.ts       word grouping into text blocks, features, candidate pairs, link decoding
  pipeline.ts     labels → links → fields, confidence, reader agreement, ID layout prior
  classify.ts     document-type sorter (evidence weights + softmax)
  facts.ts        key facts per type, date parsing, Luhn check, expiry alerts
  critic.ts       critic: expectation checks, runner-up arguments, verdict
  critic-vision.ts MobileCLIP-S0 image encoder, zero-shot type opinion
  calibration.ts  isotonic calibration of the field confidence (curves in calibration.json)
  photo.ts        classical finder for the portrait photo on an ID (used by Redact)
  marks.ts        checkboxes (square outlines + ink) and signature strokes by the "Signature" caption
  template.ts     template mode: teach a form once (anchors + labelled boxes), match, align, read boxes
  mrz.ts          ICAO 9303 parser (TD1, TD2, TD3, MRV-A/B), check digits, guarded repair
  florence.ts     Florence-2 OCR_WITH_REGION wrapper with tiling
  classifier.ts   ONNX Runtime wrapper for the two tree models (web and node)
src/engine/       browser orchestration: engine.ts plus workers (preprocess, ocr, florence, critic) and pdf.ts
src/ui/           React UI: ScanPage, Sorted (trays), ResultView, DocViewer, ModelCard, Help, Tour (guided tour, also `?tour` in the URL), bits (Ring, Stamp, CutOut), state, report
scripts/          OCR caching, dataset building, training, tree2onnx.py, evaluation, browser tests
public/models/    entity_gbdt.onnx, link_gbdt.onnx, critic-text.json, ppocr/
public/eval/      results.json (evaluation output shown in the app)
public/templates/ library.json (built-in templates, written by scripts/eval-templates.ts)
testsets/forms20/ 20 public forms filled with invented data + truth.json
tests/            unit tests (npm test)
```

Processing order for one page (`engine.ts → processDocument`): open image → preprocess worker (magic scan, deskew, binarise) → reader (Tesseract or PP-OCRv5) → code lines (neural reader first, then Tesseract with an OCR-B whitelist, several crops, kept only when check digits add up) → document type → layout models → key facts. Florence-2 and the vision critic run afterwards and refine the result.

## Decisions that are easy to undo by accident

- **Tree models are compiled to plain tensor ops** by `scripts/tree2onnx.py`. ONNX Runtime Web has no `ai.onnx.ml` operators, so a normal skl2onnx export fails in the browser with "No Op registered for TreeEnsembleClassifier". The compiled models match scikit-learn exactly.
- **`onnxruntime-web` is pinned** to the exact dev version `@huggingface/transformers` depends on, so there is one copy. `vite.config.ts` uses the `onnxruntime-web-use-extern-wasm` condition and the wasm is self-hosted in `public/ort/`.
- **Mobile nav and `backdrop-filter`:** a backdrop filter on the top bar traps the fixed bottom nav. It is switched off under 720 px.
- **PP-OCR box growth ("unclip") is 0.8**, tuned on 12 FUNSD training pages. Larger values make neighbouring lines bleed into each crop. PP-OCR runs on WASM only; the WebGPU path is untested on real hardware.
- **MRZ repair is guarded:** a character swap that makes a field's check digit pass is only kept if the composite check digit also passes. Without this, repairs produced false "valid" document numbers.
- **The vision critic is the weaker witness.** It can raise a doubt but never changes the type, and it is ignored when the sorter is above 95% with all expectations met.
- **Tesseract is the default reader** because of the 5-second rule. PP-OCRv5 is more accurate but was about 11 s per page in a slow headless test browser.

## Measured results (FUNSD test set, 50 forms, unless stated)

| What | Result |
|---|---|
| Tesseract, no clean-up → full clean-up | CER 25.8% → 20.4% |
| PP-OCRv5 mobile | CER 13.1%, word accuracy 76.7%, hallucination 3.6% |
| Florence-2, whole page → two tiles | CER 53.4% → 29.9% |
| Linking F1, gold blocks: rules / logistic regression / gradient boosting | 70.4% / 81.1% / 91.2% |
| Entity label accuracy (gradient boosting) | 80.7% |
| End-to-end field F1: Tesseract / PP-OCRv5 | 28.1% / 40.8% |
| Second-reader experiment (two tiles): Tesseract word right when readers agree / disagree | 93.0% / 48.2% |
| Document type on FUNSD forms | 47 of 50 |
| Sample scans sorted correctly in the browser test | 8 of 8 |
| Vision critic alone | 43 of 50 forms, 5 of 7 samples |
| Magic scan on 20 simulated desk photos | page found 20/20, corner error 0.4% of diagonal, word recall 56.5% → 69.7% |
| Confidence calibration error (ECE), Tesseract / PP-OCRv5, before → after | 42.1% → 4.6% / 44.6% → 8.2% |
| 20 filled public forms (522 typed values): read / extracted / right label, Tesseract | 79.5% / 15.3% / 7.7% |
| Same, PP-OCRv5 | 93.7% / 11.9% / 6.1% |
| Template mode on the same forms, right label and value: Tesseract / PP-OCRv5 | 7.7% → 74.9% / 6.1% → 93.5%, right template 20/20 |
| Checkboxes on the same forms (196): found and read right, no template / with template | 93.9% / 100% |
| Signature lines (33, synthetic scribbles): judged right, no template / with template | 72.7% / 87.9% |

Exact numbers are in `public/eval/results.json`.

## Negative results (worth reporting, do not re-try blindly)

- Florence-2 captions are too generic to name a document type ("a screen with some text").
- The 8-bit quantised MobileCLIP-S0 loses its zero-shot ability. Use fp32 for the image encoder; text embeddings are precomputed.
- Florence-2 is a worse reader than Tesseract on these faxes and sometimes invents digits. It is used as a checker only.
- The layout models do not transfer to modern boxed forms (the 20-form set): labels printed inside the box above the value get glued to the value by `groupWords`, so few pairs form. Reading is fine. This is the argument for template mode.
- New VLM OCR models from late 2026 (MiniCPM-OCR, TeleOCR, PaddleOCR-VL, GLM-OCR) are far over the 100 MB limit.

## Known gaps, most important first

1. Done 19 Nov 2026: runs on the owner's Windows machine, public repo github.com/GoldnDarkns/formflow, live at https://goldndarkns.github.io/formflow/. Not yet checked on a phone.
2. Real identity documents have now been tested for sorting and code lines (see 17). Field-level accuracy on real IDs is still unmeasured (no labelled fields), and no real visas have been seen.
3. Done: `testsets/forms20` (20 US federal forms, invented data, half scan-like) with results on the Model card. Extraction on them is poor (see negative results).
4. Done for forms: confidence is calibrated (isotonic, fitted on the FUNSD training split). Not calibrated on identity documents (no labelled ID data) and no separate curve for the second-reader blend. The review flag still thresholds the raw score.
5. Magic scan still needs a page lighter than its background to find it automatically. Manual corner adjustment is built (Photo view, "Adjust corners"); tried in the browser on one sample photo with a mouse, not on a touch screen.
6. The sorter's weights are now learned (see 17), on few documents. Visa and plain ID card have no real examples and rest on the hand-set prior. A held-out test set from the owner is still to come.
7. No Arabic reading and no handwriting model. Names in MRZ lines have no check digit, so a misread name is only caught by the cross-document check.
8. Redaction now guesses the photo area on IDs and lets the person draw boxes. The guess is classical CV, tried only on the synthetic specimens (a flat grey photo box), never on a real face photo.
9. Template mode is built (`core/template.ts`, built-in library of the 20 public forms, "Teach this form" in the viewer, user templates in localStorage `ff.templates`). Checkbox and signature detection are built (`core/marks.ts`, fields with source `mark`); ticks in the test set are PDF tick marks and signatures are synthetic scribbles, so real pen marks are untested, and signatures are only reported as present or absent. Template limits: library labels are auto-derived from the PDF text above each box and some are clumsy; digit-cell boxes read poorly with Tesseract; teaching was tried in the browser on one FUNSD sample only; a template field's confidence is the lowest word confidence, not calibrated.
10. 16 unit tests for `src/core` in `tests/` (`npm test`, run in CI). No UI tests beyond the headless browser scripts in `scripts/`.
11. Report, 1-page model card document, demo video and presentation are not started.
12. Batch 3 (looks) added 19 Nov 2026: guided tour, before/after slider in the viewer, a ninth sample, page-drop animation. Checked in the built-in browser at desktop and 375 px wide; not on a real phone. Note for testing: the in-app Browser pane pauses animations when hidden, so page transitions stall and scripted clicks misfire unless the pane is shown.
14. Look (19 Nov 2026, second pass after the owner rejected a generic glow-and-glass style as "AI generated"): the app is a desk. `ui/Backdrop.tsx` is a green cutting mat with a grid, true-to-scale outlines of A4, A5, a passport page and an ID card, and a scanner lamp that speeds up while a page is read. Panels are cream paper sheets, trays are manila folders with index tabs, buttons are hard-shadow keys, fields are form lines with typed (mono) values, and the Model card and Help read as a printed record with numbered sections. One look for light and dark devices. All of it lives in the last block of `src/index.css` ("The desk"). `--desk` is now a paper shade for boxes inside a sheet; the page colour is `--mat`; text written on the mat uses `--on-desk`. Do not give `.topbar` a low z-index: the phone nav lives inside it. Keep this direction: no gradients, glows, glass or pill shapes.
15. Look, third pass (19 Nov 2026): the owner disliked the green mat and the scanner-lamp line. The mat is now warm graphite (`--mat: #24211d`), the lamp line is gone, a warm light follows the pointer, and faint lines of text write themselves inside the document outlines and turn label-blue and value-green (faster while a page is read). Motion was added across the app with CSS only (sheets dealt onto the desk, folders sliding out, stamp ink ring, lists arriving line by line, figures typed in, Model card pages and chart lines revealed on scroll where the browser supports scroll-driven animation). These use the individual `translate` / `rotate` / `scale` properties and `animation-fill-mode: backwards` so they never fight the motion library's inline transforms or its exit fades. Do not name a class `.ghost`: the app's ghost buttons use it (this broke the layout once).
16. Grouped downloads (19 Nov 2026): `ui/bundle.ts`. Each folder tab in the trays has a PDF button (all scans of that type as one PDF, one scan per page); "Everything as ZIP" gives a folder per type (scans PDF + data JSON) plus the report, the CSV and a README. A ZIP of images and PDFs can also be dropped in as input (capped at 80 pages). The two downloads were checked in the browser by capturing the generated files; ZIP input is covered by a unit test only, not tried by hand in the browser.
17. Real documents (19 Nov 2026): the owner supplied 48 real ID pages (he confirmed the holders' consent) in `../data/own/train/<type>/`; a `test` folder beside it is for documents he will add himself and must never be used for fitting. `scripts/eval-real.ts <train|test>` runs the app pipeline on them and prints counts and neutral file names only; `scripts/train-sorter.ts` fits the sorter. First run of the unchanged app: 28/48 sorted right, code lines fully valid on 4/14. After turning sideways pages (`core/orient.ts`), a wider code-line search gated by check digits, new clues and the fitted sorter: 42/48 with the shipped model, 40/48 leave-one-out, code lines valid 8/14. `classify.ts` is now a fixed clue list with a hand-set prior and learned weights in `sorter.json` (one number per clue and type, so no personal data can leak). New type `vehicle_licence`. Privacy rules for these documents: never open the images in chat, never print text read from them, never commit anything from `../data`, publish counts only. The owner attached the first 48 to the chat himself; later ones he copies into the folder.
18. Accuracy pass (20 Nov 2026), driven by measurement, after the owner called the results mediocre. `scripts/make-specimens.py` generates made-up specimen IDs (six types, 22 countries, invented data with correct check digits, shot flat / desk / sideways / glare / blur, stamped SPECIMEN, no real design copied) into `../data/synth/{train,test,final}` with truth. `final` must not be looked at while tuning: it is the number to report. Three changes mattered: (a) passport and ID code lines are read by the neural reader first (`readMrz` in engine.ts, `mrzNeuralCrops` in orient.ts; classical crops are the fallback), all check digits pass on 39/41 final specimens and 12/14 real pages (was 4/14); (b) `linesOf` in classify.ts now builds visual rows from word positions, because Tesseract in sparse mode returns label and value as separate mis-ordered fragments; (c) `facts.ts` reads numbered EU-style licences, vehicle licences, separate surname / given-name labels, and falls back to date order (earliest = birth, latest = expiry). Final specimen set: sorted 106/108, sideways turned 13/14, key facts exactly right 258/322 (80%; names weakest at 65%). Real documents: sorted 43/48, 42/48 leave-one-out. Known faults: 5 false code lines on pages without any, 1 value shown as verified but wrong, a misread letter in a name is not caught. Speed: reading is the slow part, not training. `node scripts/eval-fast.mjs <split> <own|synth> 8` reads with 8 processes and prints one report (108 pages in about 2 minutes); `EP=dml` runs the neural reader on the RTX 3080 Ti (about 1.4 s a page); `REUSE=1` re-scores from the private cache without reading. When a python heredoc writes TypeScript, `
` inside a JS string gets turned into a real newline and breaks the script: use the Edit tool for those lines.
13. Template field confidence is the mean word confidence (was the minimum); still not calibrated.

## Privacy and licences

- Only use documents the owner has the right to use. Never commit real documents or anything from `../data`.
- FUNSD is for non-commercial research and education, so the models and the app are non-commercial. See `LICENSE` for third-party notices.
- Do not build anything that edits the text on an ID image. Redaction (black boxes on a copy) is the supported feature.

19. Phone-first pass: the result screen now opens with an answer card (type, name, one status line, key facts with "verified" marks, tap a fact to copy it, one main button). "Check n fields" opens `Review.tsx`, one doubtful field at a time with an enlarged crop. "Take a photo" opens `Camera.tsx` (live view, Card/Page frame, several shots), falling back to the photo picker. Both overlays are portalled to `body` so they sit above the phone navigation. Do not name a top-level class `.review`: `.field.review` already exists. Checked in the preview at 375 px and desktop; the live camera itself is NOT tested (the preview pane blocks cameras) and nothing has been tried on a real phone.

20. Second accuracy pass (20 Nov 2026). Worked out on specimen train/test with `scripts/eval-facts.ts` (set `DEBUG=1` to list misses) and `scripts/dump-lines.ts` (specimens only, never the real documents).
    - `facts.ts`: `mendName` compares a code-line name with the words printed on the page (exact match = `twice: true`, shown as "seen twice"; a dropped letter is mended only from words read with confidence 75 or more). `fuzzyLabelled` accepts labels with a misread letter or two. Surname and given names printed side by side are split by x position (`rowsOf` in `classify.ts`). An unlabelled name row and an unlabelled ID number are last-resort fallbacks.
    - `mrz.ts` `names()`: the K/L padding clean-up used to eat the final L of RAFAEL. It now keeps up to two K/L letters that come before the first "<".
    - `orient.ts` `shapedLikeMrz`: without a row of "<" seen by the first pass, two passing check digits are no longer enough; the lines must also have the shape of code lines.
    - `quality.ts` `photoQuality` (blur and glare, measured in the preprocess worker before clean-up) feeds `alertsFor(..., quality)`. Thresholds from `scripts/eval-quality.ts`. Glare is only reported for cards and booklets, because white paper in a photo looks the same as glare.
    - Person groups on the Trays tab now show one merged profile (a check-digit value beats an unverified one), which of passport / Emirates ID / visa is missing, the first document to expire, and "Copy details".
    - Final specimen set, second scoring: key facts 258/322 -> 278/322 (86%); names 55/85 -> 68/85; false code lines 5 -> 0; verified-but-wrong 1 -> 0; sorting unchanged 106/108. The final set has now been looked at twice and is no longer a clean hold-out. The honest next test is the owner's real documents in `data/own/test`.
    - Pitfall met again: a Python patch script turned `\b` inside a TypeScript regex into a backspace character and `\n` into a real newline. Use the Edit tool for any line containing a backslash.
    - Still weak: driving licences (67%) and vehicle licences (72%); visas with badly misread labels; about half of blurred photos pass the blur check (the mild ones).

21. Third pass (20 Nov 2026): two readers for key facts, card sides, file names, searchable PDFs, explainer.
    - `facts.ts` `extractFacts(type, words, fields, mrz, second?)`: facts are worked out from each reading separately (`factsFrom`) and merged. Same value = `twice`; only one has it = used; different = the second (neural) reading wins and the other is kept as `alt` (shown as "readers differ"), except dates: birth = the earlier candidate, expiry = the later, other dates = first reader. Check-digit values are never replaced.
    - `mendName` now separates printed words from code-line words: a code line can no longer confirm its own misreading (the neural reader's own code-line word used to "confirm" UKI for YUKI). Label words (`NOT_A_NAME`) are never used to mend a name. A part seen in a code line is only mended when it is missing letters.
    - The "Convention on Road Traffic of 8 November 1968" line is excluded from the unlabelled-date fallback (`CONVENTION`).
    - Engine: `secondLook(doc)` reads an identity page with PP-OCR and `factsOf` recomputes facts. `App.tsx` queues these looks and runs them only when no page is being read and the queue is empty, one at a time, because the PP-OCR worker is shared with the code-line step (run immediately, they stalled the next page of a pile). Not run when PP-OCR is already the main reader.
    - `scripts/eval-real.ts` caches the full-page PP-OCR reading as `second`; `eval-facts.ts` uses it (`SECOND=0` switches it off, `PREFER=first|second`).
    - Held-back specimens: key facts 280/322 (87%) with one reader -> 304/322 (94%) with two. Number 98%, name 86%, birth 96%, expiry 99%. Driving licence 71% -> 88%, vehicle licence 72% -> 100%, passport 71 -> 69 of 72. Development set 84% -> 95%, train 86% -> 96%. Not measured on real documents (they have no field truth).
    - `state.ts` `sidesOf` pairs front and back of one card; `bundle.ts` `niceName`, searchable `scansPdf` (invisible text layer), `downloadSearchablePdf`, one named PDF per document in the ZIP.
    - `../report/FormFlow-explained.html` and `.pdf`: a technical explainer written by the assistant (not the assessed report). Rebuild the PDF with Edge: `msedge --headless --print-to-pdf=... --no-pdf-header-footer file:///...html`.
    - Not checked: the second look's timing on a real phone; searchable PDFs in Acrobat (only the text layer's presence was checked).

22. Slow read and wrong cut (20 Nov 2026), reported by the owner from his phone and PC on a real passport photo (stuck minutes on "Read the text", cut-out wrong). Reproduced with a made-up page pasted on noisy backgrounds (`scripts/debug-one.ts <image>` prints timings, corners and counts, never text).
    - Cause of the delay: a grainy surface (fabric, wood, or a page that is not lighter than what it lies on) comes out of the clean-up as specks; Tesseract in sparse mode then takes minutes (132 s measured, 1,822 "words"). Otsu and the grey image are just as slow. Fix in `engine.ts`: when more than 16% of the cleaned page is black (`SPECKLED`, `rgbaInk`) the page is read by PP-OCR instead; any Tesseract read over 20 s (`READ_LIMIT_MS`) is terminated and re-read by PP-OCR; the classical code-line fallback is skipped on such pages. `doc.reader` now records the reader actually used. In the preview the test photo went from minutes to 9 s.
    - Cause of the wrong cut: (a) `detectPage` shrank the photo by picking single pixels, which keeps all the grain, so on a noisy photo the outline broke up (`shrinkMean` now averages blocks); (b) corners were the extreme points of the region, so a dark portrait next to a page corner moved that corner far into the page. Corners are now the crossings of the four sides (`sidesQuad`: convex hull, sides weighted by length and by how square they run to the longest side); the old extreme points are the fallback. Fill-ratio floor lowered from 0.8 to 0.7.
    - Still true: a page that is not lighter than its surface is not found (whole photo kept, corners by hand).
    - Development specimens after the change: sorted 107/108, code lines 43/43, key facts 321/338 (was 322).
    - `ui/FlowMap.tsx`: the whole path of a page as a tap-to-open diagram, shown on Help and at the top of the Model card. Keep it in step with the pipeline when steps change.
    - Not checked: the owner's original photo (only the cut-out he sent was available), and a real phone.
    - Camera bug found by the owner on the live site: the frame guide was an opaque cream box hiding the view, because its modifier classes were `.card` / `.page` and `.card` is the app's paper-sheet style. They are now `.as-card` / `.as-page`. Do not reuse `.card`, `.sheet`, `.page`, `.review` or `.ghost` as modifier names.
23. Card finder, live camera outline, clean ID results, A-to-Z model card (20 Nov 2026). The owner photographed his licence on a white table with an iPhone: no cut, 103 s, and a list of noise pairs.
    - `core/scan.ts` `detectObject` (outline finder): colour edges on a 320 px block-mean copy, flood the surface from the border, largest unreached object, `sidesQuad`, then three guards (fills its quad, inside colour differs from just outside, under 5% edge pixels outside the quad: that last one stops it cutting a panel out of a flat scan). `findDocument` = outline finder first, then the old bright-page `detectPage` (still needed on fabric and wood grain). The worker, `eval-real.ts` and `debug-one.ts` use it (`FINDER=bright` gives the old behaviour in the scripts).
    - `scripts/eval-finder.ts`: made-up cards pasted on five surfaces: outline finder 95/100, bright-page finder 40/100; 0 of 70 form scans cut. Development specimens re-read: key facts 321/338 (unchanged), code lines 43/43, sorted 105/108 (was 107: two visas now "other"; not investigated).
    - `ui/Camera.tsx`: the same finder runs on each video frame (every 110 ms, 320 px copy), smoothed, drawn as an SVG outline (yellow, green when still). The corners found in the captured frame travel with the shot (`quad` on the queue item, `camQuad` in `processDocument`, upright reading only). Tested only with a simulated camera (canvas `captureStream` replacing `getUserMedia` in the preview); never on a real phone by Claude.
    - `core/facts.ts` `tidyIdFields` + `ui/state.ts` `shown()`: on identity documents the layout pairs are listed only when the label is one a document prints (fuzzy match to `ID_LABELS`, shown spelled properly), the value looks like text, and it does not repeat a key fact. `analysis.fields` stays raw (facts are worked out from it). An ID with no facts at all says "Could not read this document clearly" with what to do.
    - `ui/Explain.tsx` on the model card: problem to answer to evidence, then parts A to L (what, why, how, measured). Its figures are typed in, not read from `results.json`: update them when a model is retrained.
24. Hand-held cards, flash, the two percentages (20 Nov 2026). The owner held his Emirates ID up to a laptop camera: the whole photo (face and hand) was kept and went into the scan PDF.
    - `core/scan.ts` `detectCard` (four-edges finder): gradient with direction, Hough lines (each edge pixel votes within 5 degrees of its own direction), strongest 34 lines, two near-parallel pairs at right angles, side ratio 1.2 to 2, sides covered by edges (within 2 px), score = cover squared times root of area. Fingers over an edge are tolerated. A form is full of such boxes, so it is used only (a) in the live camera after `detectObject`, and (b) in `processDocument` after the first reading of an uncut page not recognised as a form, and then only if `quadHoldsWords` (under 80% of the picture, 85% of word centres inside) and the re-read is not clearly poorer. `eval-real.ts` does not mirror (b).
    - `scripts/eval-finder.ts` "held in a hand" row (busy coloured shapes behind, finger blobs over one edge): four-edges finder 23/30, the other two finders 9/30. Striped or gridded backgrounds defeat it. On the owner's photo (taken back out of the scan PDF, so a degraded copy) the card is cut exactly and sorted; two dates came out wrong but flagged "readers differ".
    - Camera: a light button appears when the browser exposes `torch` (Android Chrome). Safari on iPhone does not, so there is no button there; not tested on any phone.
    - The ring now carries the caption "trust" and the type line says "type 96% sure": they were two different numbers shown without labels (type certainty against overall trust in the extraction).
