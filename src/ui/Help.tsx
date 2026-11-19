export function Help() {
  return (
    <div className="prose" style={{ maxWidth: 780, margin: '0 auto' }}><div className="card">
      <h1>How to use FormFlow</h1>
      <ol>
        <li><b>Scan.</b> Choose or photograph as many pages as you like, in any order. Lay each page flat, fill the frame, avoid shadows and glare.</li>
        <li><b>Let it sort.</b> Each page is stamped with what FormFlow thinks it is and lands in that tray. If the stamp is wrong, open the document and change "Sorted as".</li>
        <li><b>Check.</b> Open a document. Doubtful fields come first. Tap one to see it highlighted on the page with a close-up, fix the text, then tick it.</li>
        <li><b>Export.</b> Download JSON for one document or CSV for all of them.</li>
      </ol>
      <h2>Magic scan</h2>
      <p>For photos, FormFlow finds the sheet of paper, cuts away everything around it, flattens the perspective and evens out shadows. Use the Photo / Cut out switch above a document to compare. It works best when the page is lighter than the surface it lies on. If no page is found, the image is used as it is.</p>
      <h2>The critic</h2>
      <p>After the sorter decides what a page is, the critic checks the decision three ways: does the page contain what that type should contain, what argues for a different type, and what does a separate vision model think when it looks only at the picture. The vision model is the weaker witness, so it can raise a doubt but never changes the type by itself.</p>
      <h2>PDFs, reports and redacted copies</h2>
      <p>You can drop PDF files: each page is read on its own. "PDF report" gives a printable summary of one document or of all of them, including the cross-document checks. "Redact" blacks out identity numbers and dates of birth on a copy of the image; tap any box to hide or show it. Photos of faces are not hidden automatically.</p>
      <h2>The trust score</h2>
      <p>The ring on each document blends three things: how confident the field extraction is, how sure the type recognition is, and whether the checksums pass (passport check digits, the Emirates ID check digit). Fields you confirm count as certain, so the ring fills as you review.</p>
      <h2>Checks across documents</h2>
      <p>When several documents carry the same name, FormFlow compares their date of birth, name spelling and passport number and tells you if they disagree. It also warns about expired documents and ones expiring within six months.</p>
      <h2>What the colours mean</h2>
      <ul>
        <li><b style={{ color: 'var(--q)' }}>Blue</b> question/label, <b style={{ color: 'var(--a)' }}>green</b> answer/value, <b style={{ color: 'var(--h)' }}>purple</b> heading. Orange arrows link each question to its answer.</li>
        <li><b>Confidence</b> combines the OCR's own certainty, the link classifier's probability and, if enabled, whether Florence-2 read the same text.</li>
        <li><b>Needs review</b> means at least one warning: low OCR confidence, an uncertain link, the two readers disagreeing, or a failed passport check digit.</li>
      </ul>
      <h2>Passports, ID cards and visas</h2>
      <p>The two or three lines of <code>&lt;&lt;&lt;</code> text at the bottom (the machine-readable zone) are read separately and every check digit is verified. Names and the document number from the printed page are cross-checked against it (“Matches MRZ”). A failed check digit almost always means a misread character, so look at that field closely.</p>
      <h2>Second reader (Florence-2)</h2>
      <p>Turn it on from the Scan tab. The model (about 275 MB) downloads once and is stored by your browser. On computers with WebGPU it takes a few seconds per page; on phones without it, expect 20–60 seconds. Results from the main reader appear first and are refined when it finishes.</p>
      <h2>Privacy</h2>
      <p>Everything runs inside your browser. Images, text and results are never sent to a server, and nothing is stored after you close the tab unless you export it. You can install FormFlow (browser menu → “Install app” / “Add to Home Screen”) and use it offline.</p>
      <h2>Limits</h2>
      <ul>
        <li>English / Latin-script print works best. Handwriting, Arabic script, stamps and checkboxes are often missed or misread.</li>
        <li>FormFlow does not verify that a document is genuine.</li>
        <li>Large photos are resized to 2400 px on the long side.</li>
      </ul>
    </div></div>
  );
}