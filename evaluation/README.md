# Manuscript Review Agent

This directory contains the standalone manuscript scoring agent used by the WARA web console.

The agent accepts a manuscript PDF, extracts text from the PDF, sends only that extracted text to the selected LLM scoring profile, and writes JSON/CSV outputs with two paper-aligned scores:

- manuscript-level research validity;
- optimization research maturity.

It is independent of the three-phase WARA research pipeline. It does not read WARA run manifests, TeX sources, BibTeX files, figures, experiment logs, or hidden artifacts.

Example:

```bash
python evaluation/llm_paper_review_agent.py \
  --pdf path/to/manuscript.pdf \
  --out evaluation_results/manuscript_review.json \
  --csv evaluation_results/manuscript_review.csv \
  --model-profile openai-gpt-5.5
```
