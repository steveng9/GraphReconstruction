# Graph_Reconstruction

One body of text, two formats.

| File | Role |
|---|---|
| `main_neurips.tex` | NeurIPS 2026 wrapper (template only) |
| `main_acm.tex` | ACM `acmart` sigconf two-column wrapper (template only) |
| `macros.tex` | shared packages + macros + `\todo`/`\note`/`\idea` |
| `abstract.tex`, `content.tex`, `appendix.tex` | **the paper** -- edit only these |
| `notes.tex` | private working notes (ideas, designs, experiment log); printed as an appendix while `\shownotestrue` |
| `refs.bib` | shared bibliography |
| `neurips_2026.sty`, `acmart.cls`, `ACM-Reference-Format.bst` | class/style files |

**Overleaf:** New Project > Upload Project > this zip. Menu > *Main document* picks `main_neurips.tex` or `main_acm.tex`.

**Switch format:** change the main document; nothing else. Write `\linewidth` (not `\textwidth`) for figures and use `table*`/`figure*` sparingly so both layouts work.

**Before submitting:** set `\shownotesfalse` in `macros.tex` (hides all `\todo`s and the notes appendix); for NeurIPS add the checklist from the official template; edit the author block/venue in the relevant main file.

Note: the project was assembled without a local LaTeX install, so the first Overleaf compile is the first test -- tell me any error and I'll fix it.
