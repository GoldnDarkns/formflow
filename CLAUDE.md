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

Processing order for one page (`engine.ts → processDocument`): open image → preprocess worker (magic scan, deskew, binarise) → reader (Tesseract or PP-OCRv5) → MRZ read (Tesseract with an OCR-B whitelist, best of several crops) → document type → layout models → key facts. Florence-2 and the vision critic run afterwards and refine the result.

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
2. All identity-document results are on synthetic specimens generated for testing. No real passports, Emirates IDs, licences or visas have been tested.
3. Done: `testsets/forms20` (20 US federal forms, invented data, half scan-like) with results on the Model card. Extraction on them is poor (see negative results).
4. Done for forms: confidence is calibrated (isotonic, fitted on the FUNSD training split). Not calibrated on identity documents (no labelled ID data) and no separate curve for the second-reader blend. The review flag still thresholds the raw score.
5. Magic scan still needs a page lighter than its background to find it automatically. Manual corner adjustment is built (Photo view, "Adjust corners"); tried in the browser on one sample photo with a mouse, not on a touch screen.
6. The sorter is rule-based, not learned. A learned sorter was considered and not built: there is no labelled identity-document data to train it on (5 synthetic specimens), so it could not be evaluated honestly.
7. No Arabic reading and no handwriting model. Names in MRZ lines have no check digit, so a misread name is only caught by the cross-document check.
8. Redaction now guesses the photo area on IDs and lets the person draw boxes. The guess is classical CV, tried only on the synthetic specimens (a flat grey photo box), never on a real face photo.
9. Template mode is built (`core/template.ts`, built-in library of the 20 public forms, "Teach this form" in the viewer, user templates in localStorage `ff.templates`). Checkbox and signature detection are built (`core/marks.ts`, fields with source `mark`); ticks in the test set are PDF tick marks and signatures are synthetic scribbles, so real pen marks are untested, and signatures are only reported as present or absent. Template limits: library labels are auto-derived from the PDF text above each box and some are clumsy; digit-cell boxes read poorly with Tesseract; teaching was tried in the browser on one FUNSD sample only; a template field's confidence is the lowest word confidence, not calibrated.
10. 16 unit tests for `src/core` in `tests/` (`npm test`, run in CI). No UI tests beyond the headless browser scripts in `scripts/`.
11. Report, 1-page model card document, demo video and presentation are not started.
12. Batch 3 (looks) added 19 Nov 2026: guided tour, before/after slider in the viewer, a ninth sample, page-drop animation. Checked in the built-in browser at desktop and 375 px wide; not on a real phone. Note for testing: the in-app Browser pane pauses animations when hidden, so page transitions stall and scripted clicks misfire unless the pane is shown.
14. Look (19 Nov 2026, second pass after the owner rejected a generic glow-and-glass style as "AI generated"): the app is a desk. `ui/Backdrop.tsx` is a green cutting mat with a grid, true-to-scale outlines of A4, A5, a passport page and an ID card, and a scanner lamp that speeds up while a page is read. Panels are cream paper sheets, trays are manila folders with index tabs, buttons are hard-shadow keys, fields are form lines with typed (mono) values, and the Model card and Help read as a printed record with numbered sections. One look for light and dark devices. All of it lives in the last block of `src/index.css` ("The desk"). `--desk` is now a paper shade for boxes inside a sheet; the page colour is `--mat`; text written on the mat uses `--on-desk`. Do not give `.topbar` a low z-index: the phone nav lives inside it. Keep this direction: no gradients, glows, glass or pill shapes.
15. Look, third pass (19 Nov 2026): the owner disliked the green mat and the scanner-lamp line. The mat is now warm graphite (`--mat: #24211d`), the lamp line is gone, a warm light follows the pointer, and faint lines of text write themselves inside the document outlines and turn label-blue and value-green (faster while a page is read). Motion was added across the app with CSS only (sheets dealt onto the desk, folders sliding out, stamp ink ring, lists arriving line by line, figures typed in, Model card pages and chart lines revealed on scroll where the browser supports scroll-driven animation). These use the individual `translate` / `rotate` / `scale` properties and `animation-fill-mode: backwards` so they never fight the motion library's inline transforms or its exit fades. Do not name a class `.ghost`: the app's ghost buttons use it (this broke the layout once).
16. Grouped downloads (19 Nov 2026): `ui/bundle.ts`. Each folder tab in the trays has a PDF button (all scans of that type as one PDF, one scan per page); "Everything as ZIP" gives a folder per type (scans PDF + data JSON) plus the report, the CSV and a README. A ZIP of images and PDFs can also be dropped in as input (capped at 80 pages). The two downloads were checked in the browser by capturing the generated files; ZIP input is covered by a unit test only, not tried by hand in the browser.
13. Template field confidence is the mean word confidence (was the minimum); still not calibrated.

## Privacy and licences

- Only use documents the owner has the right to use. Never commit real documents or anything from `../data`.
- FUNSD is for non-commercial research and education, so the models and the app are non-commercial. See `LICENSE` for third-party notices.
- Do not build anything that edits the text on an ID image. Redaction (black boxes on a copy) is the supported feature.
