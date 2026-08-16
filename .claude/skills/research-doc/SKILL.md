---
name: research-doc
description: Write up investigation/debugging findings as a checked-in RESEARCH_<slug>.md file at the repo root, instead of leaving them only in chat or scratch files. Use this proactively, without being asked, any time a session does non-trivial context-gathering work that would be expensive to redo -- reading source across multiple files, root-causing a defect, running live experiments against a running Blender instance via tools/blender_bridge.py, or writing isolated bmesh/Python reproduction snippets to confirm a hypothesis. Also use it when asked explicitly to "write this up," "make this a doc," or "don't make me re-investigate this later." Trigger on the shape of the work (several tool calls converging on a root cause or a confirmed/ruled-out hypothesis), not only on an explicit request to document.
---

# Writing a RESEARCH_ document

## Why this exists

This project's CLAUDE.md already has a hard rule for verification code: "Every
check goes in `tests/`, never in a scratch file," because a measurement
written to answer one question gets asked again the next time that code
changes, and re-deriving it from scratch costs more than reading it back.

This skill applies the same discipline to prose findings. A live-Blender
debugging session that reads source, forms a hypothesis, and confirms or
rules it out with a handful of isolated experiments is expensive: it took
real tool calls, a live Blender connection, and careful reasoning to
reach whatever was found. If that reasoning only exists in a chat transcript,
the next session -- yours or someone else's -- pays that cost again, and may
not even reach the same answer. Writing it down once is cheap insurance
against paying for it twice.

## Where a RESEARCH_ doc fits among this project's other documents

Three kinds of checked-in record exist in this repo, and they answer
different questions. Picking the right one matters more than the writing:

| Kind | Answers | Lives in |
|---|---|---|
| `tests/*.py` | "Is this still true?" -- re-run automatically | `tests/` |
| `RESEARCH_<slug>.md` | "What did we find out, and how do we know?" -- a diagnosis | repo root |
| `DESIGN_<feature>.md` | "What are we building, and why this way?" -- a decision or direction | repo root |

A RESEARCH_ doc is diagnostic, not decisive. It records a symptom, a root
cause, and the evidence that pins it down -- and it stops there. It does
**not** pick a fix or commit to a plan. This project's CLAUDE.md has a strong
"ask before inventing" culture: a design substitution made silently, without
raising it to the user first, is the single most expensive mistake this
project keeps warning itself about. A RESEARCH_ doc that quietly slides from
"here's what's wrong" into "here's what I'm going to build" undermines that
guardrail in writing, even when the session itself asked first. Keep
"candidate directions" clearly labeled as unstarted options, the way the
existing `RESEARCH_gp_pattern_seam_gaps.md` does under its "next steps"
heading.

If a finding *does* lead to a design decision the user has signed off on,
that decision belongs in a `DESIGN_` doc (or directly in the code as a
comment explaining a non-obvious choice) -- and it should link back to the
`RESEARCH_` doc that grounds it, rather than duplicating the evidence.

## Naming and scope

`RESEARCH_<slug>.md` at the repo root, one file per **investigated question**,
not one per feature. A feature that gets investigated several times over its
life should accumulate several sibling files
(`RESEARCH_gp_pattern_seam_gaps.md`, `RESEARCH_gp_pattern_<next-question>.md`,
...), each scoped to one symptom or one root-cause hunt. Splitting this way
keeps each file readable and means a future session can find the one relevant
document instead of skimming a growing, unrelated pile inside a single
feature-wide file.

Pick `<slug>` the way you'd name a bug report: specific enough that the
filename alone tells a future session what question it answers
(`seam_gaps`, not `bugs`; `uv_island_flood_fill_order`, not `uv_issue`).

## What goes in one

Structure each document as a diagnosis, in this order. Not every section
needs to be long, but skipping the evidence section is what makes a
write-up worthless the next time someone doubts it.

1. **The symptom.** What was observed, concretely -- what the user reported,
   or what a check produced. Include enough detail (object names, a screenshot
   description, exact numbers) that someone who never saw the original report
   can picture it. Name the exact state it was observed against: which file,
   which live session/port, whether the repro is against saved or unsaved
   state.

2. **Root cause**, stated as a claim, immediately followed by how it was
   confirmed. This project's CLAUDE.md is explicit that a claim like
   "Blender cannot do X" or "this API doesn't support Y" needs a test, not a
   reading -- the same standard applies here to any claim about *why*
   something is happening. Prefer a minimal, standalone reproduction (a short
   `bmesh`/Python snippet with its measured output inlined, not just
   described in prose) over a description of what you believe is happening
   inside the real, complicated pipeline. A ten-line isolated repro that
   anyone can paste into `tools/blender_bridge.py` and get the same answer
   from is worth more than a confident paragraph.

3. **Where it shows up in the real code**, with `path:line` references (this
   repo's convention throughout `foam_gp_pattern.py`'s own comments) tying
   the isolated finding back to the actual pipeline. If real data was
   involved -- a live scene, a specific test file -- record enough of the
   trace (log lines, counts, coordinates) that the correlation is checkable,
   not asserted.

4. **What's still unknown.** State the boundary of what was actually checked.
   A finding that explains 4 of 5 observed instances should say so, rather
   than imply full coverage it doesn't have.

5. **Candidate next steps**, explicitly framed as *options to raise*, not a
   plan. If there are real trade-offs between them, name the trade-off in one
   line each rather than picking a favorite -- that choice belongs to the
   user.

Reproduction steps should be copy-pasteable. If the project's live-Blender
workflow was used (`tools/launch_blender.py` + `tools/blender_bridge.py`, per
CLAUDE.md's "Testing approach" section), show the exact commands and inline
scripts, not a paraphrase of what they did -- the point is that a future
session can re-run them verbatim and get the same numbers back.

## After writing it: make it findable

A RESEARCH_ doc that nobody knows exists gets re-derived anyway. Two
follow-ups matter more than the writing itself:

- **Reference it from CLAUDE.md.** Add or update a short pointer -- a line
  under the relevant section, or in an index of research docs -- naming the
  file and, in a few words, the question it answers. This is what lets a
  future session notice the question has already been investigated *before*
  spending tool calls re-investigating it. Leave the actual CLAUDE.md edit to
  the user if they've indicated they want to do it themselves in this
  session; otherwise make it.
- **If the investigation grew inline inside another document** (a DESIGN_
  doc, or CLAUDE.md itself) rather than starting in its own file, extract it:
  move the investigation into its own `RESEARCH_<slug>.md`, and leave a short
  pointer sentence in the original spot instead of the full text. Don't leave
  the same investigation duplicated in two places -- one canonical copy, one
  or more one-line pointers to it. `DESIGN_grease_pencil_patterning.md`'s
  "Known defect" section, which now just points at
  `RESEARCH_gp_pattern_seam_gaps.md`, is the pattern to repeat.

## A short example of the shape

```markdown
# Research: <what question this answers>

## The symptom (<date>)
...concrete description, with the exact state it was observed against...

## Root cause, confirmed by isolated experiment, not by reading
<claim>, confirmed by:

​```python
# minimal repro
​```
Result measured this session: <exact output>.

## Where it shows up in the real pipeline
`some_file.py:123-145` does X, which produces this shape whenever Y.
On the real file: <trace correlating the isolated finding to observed data>.

## What is not yet known / next steps
- <boundary of what was checked>
- Candidate directions, none tried:
  - <option A> -- <one-line trade-off>
  - <option B> -- <one-line trade-off>
- These are options to raise with the user before building, not a plan to
  execute unasked.
```
