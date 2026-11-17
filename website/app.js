'use strict';
const clarify = document.querySelector('#clarify-button');
clarify.addEventListener('click', () => {
  const panel = document.querySelector('#clarification');
  panel.hidden = !panel.hidden;
  clarify.setAttribute('aria-expanded', String(!panel.hidden));
  clarify.firstChild.textContent = panel.hidden ? 'Ask a clarifying question ' : 'Hide the clarification ';
});

async function loadExamples() {
  const select = document.querySelector('#example-select');
  try {
    const response = await fetch('assets/data/examples.json');
    if (!response.ok) throw new Error('Examples unavailable');
    const { examples } = await response.json();
    const render = () => {
      const item = examples[Number(select.value)];
      const fields = {
        'example-id': item.instance_id,
        'example-question': item.question,
        'example-default': item.default_answer,
        'example-answer': item.answer,
        'example-default-reading': Object.values(item.default_interpretation).join(' · '),
        'example-intended-reading': Object.values(item.intended_interpretation).join(' · '),
        'example-context': item.context,
        'example-evidence': item.intended_evidence_span,
        'example-default-evidence': item.default_evidence_span
      };
      for (const [id, text] of Object.entries(fields)) {
        const element = document.getElementById(id);
        element.textContent = text;
        if (id !== 'example-id') element.lang = item.language === 'zh' ? 'zh-Hans' : 'en';
      }
      document.querySelector('#example-status').textContent = 'Original dataset wording is preserved. Example ' + (Number(select.value) + 1) + ' of ' + examples.length + '.';
    };
    select.addEventListener('change', render);
    render();
  } catch (_) {
    select.disabled = true;
    document.querySelector('#example-status').textContent = 'The example selector could not load. The Meta example remains visible; download the records below to inspect all instances.';
  }
}
loadExamples();

document.querySelector('#copy-citation').addEventListener('click', async () => {
  const button = document.querySelector('#copy-citation');
  const citation = document.querySelector('#bibtex');
  const status = document.querySelector('#copy-status');
  try {
    await navigator.clipboard.writeText(citation.textContent);
    button.textContent = 'Copied';
    status.textContent = 'BibTeX copied to clipboard.';
    setTimeout(() => { button.textContent = 'Copy BibTeX'; }, 2000);
  } catch (_) {
    const range = document.createRange();
    range.selectNodeContents(citation);
    const selection = window.getSelection();
    selection.removeAllRanges();
    selection.addRange(range);
    button.textContent = 'Selected — press Ctrl/Cmd+C';
    status.textContent = 'Citation selected. Press Control C or Command C to copy.';
  }
});
