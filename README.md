# Projects

Three independent research projects.

| Project | Domain | Compute | Status |
|---|---|---|---|
| [GPR151](#1-gpr151-structure-based-drug-discovery) | Structure-based drug discovery | Kaggle (GPU) | Pipeline in development |
| [VCC](#2-virtual-cell-challenge) | Single-cell perturbation prediction | Local Windows machine | Active entry ("Chorus") |
| [nano](#3-nano) | Brain-inspired conversational agent | Kaggle (training), Modal (serving) | Research prototype, Sept 2026 |

---

## 1. GPR151: structure-based drug discovery

**Target.** GPR151 (UniProt Q8TDV0), an orphan GPCR in the SOG superfamily with no confirmed endogenous ligand.

**Goal.** Find small-molecule ligands through a computational pipeline, starting from the recent structural and pharmacological data.

**Key references**
- Wang et al. 2026: cryo-EM structure that reveals an autoinhibitory N-terminal lid.
- Yoshida et al. 2026: first reported small-molecule agonists; their scaffold serves as the pharmacophore anchor.

**Pipeline**
1. Structure preparation (PDBFixer; AMBER force field, with histidine protonation states resolved)
2. Molecular dynamics (OpenMM on CUDA; PLUMED for enhanced sampling) to generate a conformational ensemble
3. Ensemble docking across the MD snapshots
4. Virtual screening
5. Interaction fingerprinting (ProLIF) and trajectory analysis (MDAnalysis)

**Pharmacophore / SAR findings so far**
- Furan/benzofuranone core: π-stacks with W6.52
- NH linker: hydrogen bond to D3.32
- A halogen-bonding site
- An endogenous structural analog has been identified

---

## 2. Virtual Cell Challenge

**Competition.** Arc Institute's Virtual Cell Challenge: predict transcriptome-wide single-cell responses to genetic perturbations (CRISPRi knockdowns).

**Evaluation.** Three metrics:
- DES: differential expression score
- PDS: perturbation discrimination score
- MAE: mean absolute error

The metric definitions were analysed in depth, including how earlier teams exploited the scoring formulas.

**Self-imposed constraints**
1. Overall score ≥ 0.25
2. No metric below baseline
3. Nothing pregenerated: every value and function is learned from the data

**Current entry: Chorus.** A donor-transfer model, documented in `CHORUS.md` and `SCORER.md` (working directory `Gaudi/`). Supporting architecture work: predicting expression changes (deltas) rather than absolute levels, cross-validated magnitude scaling, multi-task losses.

**Research direction.** The aim is a new approach rather than incremental changes to CellOracle, STATE, GEARS or foundation models.
- Earlier concept: a physics-inspired gene regulatory network. Genes are particles with signed directional couplings; "super-particles" act as higher-order logic gates.
- Current bet: nonlinear fluctuation-dissipation theory (FDT). The Kubo response hierarchy is extended through score-based generative modelling and applied to perturbation prediction. This grounds the super-particle idea in measurable three-point correlation functions.
- Method: falsification-first. Test the FDT hypothesis cheaply before committing to a full build.

**Compute.** Runs locally on a Windows machine. The ~172 GB corpus is on the E: drive.

---

## 3. nano

**What it is.** A chatbot you teach as you talk. Each conversation starts from zero knowledge, and nano:
- learns a fact from a single mention,
- infers facts it was never told,
- reports which brain-analogous system produced each reply.

**Live:** ericdu847--nano.modal.run

**Core constraint.** Learning uses local, synapse-level rules like those in real neurons. Each weight change depends only on activity on either side of the synapse. Backpropagation is used only as a yardstick to measure how far the brain-like rules have to go.

**One turn**
1. **Hear:** Wernicke's area repairs typos, resolves references ("it", "there") and classifies the dialogue act.
2. **Choose:** a basal-ganglia gate picks which system answers. It grows new experts for new situations and learns from what worked.
3. **Answer:** the chosen system answers from what it holds.
4. **Speak:** Broca's area turns the answer into words, and the cortex hears the whole exchange.

**Systems**

| System | Brain analogue | Function | Learned or hand-written |
|---|---|---|---|
| Cognitive maps | Hippocampus | Places things on scales and floor plans from single facts, infers the rest, consolidates during "sleep" | Learned |
| Object files | Ventral stream | One record per entity: identity, owner, properties, possessions, location | Content learned; grammar hand-written |
| Self-model | — | nano and the user are objects in its own world; its state answers "how are you?" | Readings real; wording hand-written |
| Gate | Basal ganglia | Chooses which system answers | Learned |
| Episodic memory | — | What was said earlier in the conversation | Learned |
| Reader | — | Answers world-knowledge questions from Wikipedia with a trained span-extraction model | Learned |
| News | — | Recent headlines for "right now" questions | Hand-written |
| Talking striatum | Striatum | Recalls what people actually said in real conversations | Learned |
| Cortex | Cortex | Recurrent network that scores candidate replies against the conversation so far | Learned (e-prop) |
| Understanding | — | Parsing; telling a question from a fact from a clarification request | Hand-written |

**Cortex results.** e-prop is a brain-plausible rule: each synapse keeps a fading trace of its recent input and changes when a learning signal arrives. Scored by held-out conversational perplexity (lower is better).

| Model | Perplexity |
|---|---|
| nano cortex (e-prop) | 94.5 |
| Backprop yardstick | 49.4 |
| 5-gram model | 30.4 |

The cortex judges replies but doesn't generate them until it beats the 5-gram. A built-in switch lets a better checkpoint's own drafts compete automatically. Training continues on Kaggle.

**Replacing hand-written understanding.** Two pre-registered experiments test learned readouts on the cortex, trained with the delta rule (a local rule):

| Task | Hand-written | Tuned standard classifier | Learned readout (best) |
|---|---|---|---|
| Dialogue acts (macro-F1, real nano messages; 200k labelled phone-call utterances) | **0.77** | 0.68 | 0.61 |
| Speaker facts (exact-match rate; 310k labelled sentences) | 0.6% | **20%** | 8.3% |

The hand-written rules stay for now. Phone-call data transfers poorly to typed chat, and even the standard classifier loses to the rules there. Next step: training data from typed chat.

**Open issues**
- The 5-gram scores better than the backprop network (30.4 vs 49.4). That's unexpected, and it sets the threshold for letting the cortex generate replies. Check that the yardstick is trained enough, and that vocabulary and unknown-word handling match across all three models, before treating 30.4 as the bar.
- The hand-written grammar's 0.6% exact match on speaker facts is suspiciously low next to the rules' 0.77 on dialogue acts. Check whether the grammar's output format matches the labels before concluding the grammar is the weak point.

---

## Repo layout and Kaggle sync

```
gpr151/  vcc/  nano/        notebooks/<slug>/ and datasets/<slug>/ per project
other/                      notebooks and datasets that don't belong to the three projects
routing_report.md           where every item went and why (regenerated each sync)
kaggle_manifest.json        every synced item, with its source account
scripts/kaggle_sync.py      the sync script
scripts/routes.json         routing rules and per-item overrides
.github/workflows/          runs the sync daily and on demand
```

`.github/workflows/kaggle-sync.yml` pulls every notebook (code plus `kernel-metadata.json`) and every dataset's metadata from each Kaggle account once a day. Dataset files and notebook outputs are not synced. The sync is one-way, Kaggle → GitHub.

**Routing.** Each notebook is sorted by its contents: project keywords found in its source (for example `openmm`/`rdkit` → gpr151, `scanpy`/`.h5ad`/`perturbation` → vcc, `e-prop`/`dialogue act`/`perplexity` → nano), plus name patterns on its slug and title. A notebook only goes to a project if the evidence clearly favours it; otherwise it goes to `other/`. Datasets follow the notebooks that attach them. To pin an item somewhere, add `"owner/slug": "folder"` to `overrides` in `scripts/routes.json`; the next sync moves it.

**Setup**
1. On each Kaggle account, go to kaggle.com/settings → API and create a token.
2. On GitHub, go to repo → Settings → Secrets and variables → Actions and add one secret per account (`ERIC847`, `ERICDU847`, `ERICDU4831`) holding that account's `kaggle.json` contents or bare key.
3. Go to Actions → *Kaggle sync* → *Run workflow*.
