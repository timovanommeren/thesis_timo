# Jump-Starting Evidence Synthesis

This repository contains the code and data to reproduce the results of:

> *Jump-Starting Evidence Synthesis: Initializing Active Learning Models for Systematic Reviews Using LLM-Generated Data*
> Timo van Ommeren — MSc thesis, Utrecht University (2026)

The study investigates the **cold start problem** in AI-assisted systematic review screening using [ASReview](https://asreview.nl). Four initialization conditions are compared across 23 systematic review datasets from the [SYNERGY corpus](https://github.com/asreview/synergy-dataset):

| Condition | Description |
|---|---|
| **LLM** | GPT-4o-mini–generated synthetic abstracts prepended as training data |
| **Eligibility criteria** | The review's eligibility criteria used directly as a prior |
| **Real abstract** | One true relevant paper sampled at random |
| **Cold start** | No prior — random screening until the first hit |

The outcome measure is the number of relevant papers found within the first 100 records screened.

Simulation outputs are large and stochastic (LLM calls). The raw results are archived on OSF and can be downloaded to reproduce all figures and tables **without re-running the simulation** (see [Path B](#path-b-fast-track-analysis-replication-osf-data) below).

---

## Repository Structure

```plaintext
.
├── simulation/                   # Python simulation pipeline
│   ├── run.py                    # Entry point — orchestrates the full simulation
│   ├── simulation.py             # ASReview active-learning loop (four conditions)
│   ├── llm.py                    # Builds LLM-augmented and criteria datasets
│   ├── prompting.py              # DSPy + GPT-4o-mini abstract generation
│   ├── stimulus.py               # Reads eligibility criteria from metadata
│   ├── priors.py                 # Seeded random prior sampling
│   ├── metrics.py                # Computes papers_found and ATD; appends to CSV
│   ├── recall_plots.py           # Aggregate recall curve figures
│   ├── aggregate_results.py      # Standalone CLI for recall aggregation
│   └── config.py                 # Reads pyproject.toml; defines defaults
│
├── Analysis/                     # R analysis pipeline
│   ├── analysis.Rmd              # Entry point — sources all R scripts, fits models,
│   │                             #   writes figures/tables to Report/results/
│   ├── R/
│   │   ├── 01_load_data.R        # Loads simulation CSV and metadata
│   │   ├── 02_prepare_data.R     # Pivots, joins metadata, rescales, two-part coding
│   │   ├── 03_descriptives.R     # Descriptive plots (barcharts, histograms)
│   │   └── bootstrap.R           # Pairs-bootstrap inference (B = 10,000)
│   └── percentage_relevant.csv   # Dataset-level metadata (N, % relevant, topic)
│
├── Report/                       # LaTeX manuscript
│   ├── main.tex                  # Main document
│   ├── preamble.tex              # Packages and formatting
│   ├── titlepage.tex             # Title page
│   ├── references.bib            # Bibliography (biblatex)
│   ├── results/                  # Generated figures and tables (written by R)
│   ├── images/                   # Static images
│   └── vector_space/             # Vector space visualisation figures
│
├── data/                         # Input data (git-ignored; present on disk)
│   ├── synergy_dataset/          # 23 SYNERGY systematic review CSVs
│   ├── inclusion_criteria_prepared.xlsx   # Eligibility criteria metadata
│   └── synergy_dataset_overview.xlsx      # Dataset overview table
│
├── simulation_results/           # Generated outputs (git-ignored; archived on OSF)
│   ├── run_01/                   # Main simulation (all four conditions, 3×3×3 IV grid)
│   │   └── all_simulation_results.csv
│   └── run_varying_eligibility_criteria/  # Supplementary: criteria-modification run
│       └── all_simulation_results.csv
│
├── pyproject.toml                # Simulation configuration (IVs, n_simulations, etc.)
├── requirements.txt              # Python dependencies (pinned)
├── renv.lock                     # R package snapshot (renv)
└── .env                          # API keys — create from env.example (git-ignored)
```

---

## Prerequisites

### Python

- **Python ≥ 3.10**
- All dependencies are listed in [`requirements.txt`](requirements.txt) with pinned versions
- An **OpenAI API key** is required to run the LLM simulation (Path A only)

### R

- **R ≥ 4.3**
- Package management via [`renv`](https://rstudio.github.io/renv/); exact versions are locked in [`renv.lock`](renv.lock)
- Run `renv::restore()` once to install all packages (see Path B, Step 1)

Key packages used: `here`, `dplyr`, `tidyr`, `ggplot2`, `ggtext`, `patchwork`, `emmeans`, `car`, `glmmTMB`, `ggsignif`, `viridisLite`, `forcats`, `stringr`

### LaTeX

- **TeX Live** (≥ 2022) or **MiKTeX**, with `latexmk`
- Required packages are loaded in [`Report/preamble.tex`](Report/preamble.tex); all are included in a standard TeX Live full installation

---

## Reproducing the Results

Two paths are provided. **Path B is recommended** for most readers, as it bypasses the computationally expensive LLM simulation (which requires an OpenAI API key and significant API cost).

---

### Path A: Full End-to-End Replication

> Runs the complete pipeline from raw data → simulation → analysis → compiled manuscript.  
> Requires an OpenAI API key. Estimated API cost: ~$X for the full 270-run grid.

#### 1. Clone the repository

```bash
git clone https://github.com/timovanommeren/thesis_timo.git
cd thesis_timo
```

#### 2. Set up the Python environment

```bash
python -m venv .venv

# macOS / Linux
source .venv/bin/activate

# Windows
.\.venv\Scripts\activate

pip install -r requirements.txt
```

#### 3. Add your OpenAI API key

```bash
cp env.example .env
# Open .env and set:  OPENAI_KEY=sk-...
```

#### 4. Configure the simulation

Open [`pyproject.toml`](pyproject.toml) and verify the `[tool.jumpstart]` section. The defaults reproduce the published results:

```toml
[tool.jumpstart]
n_simulations    = 20
stop_at_n        = 100
n_abstracts      = [1, 4, 7]
length_abstracts = [100, 500, 900]
llm_temperature  = [0.0, 0.4, 0.8]
stimuli_for_llm  = ["Original"]
condition        = ["random", "llm", "criteria", "no_initialisation"]
# subset_datasets  = [...]   # uncomment to test on a single dataset first
```

#### 5. Run the main simulation

```bash
python simulation/run.py run \
  data/synergy_dataset \
  simulation_results/run_01 \
  data/inclusion_criteria_prepared.xlsx
```

This runs 270 simulations × 23 datasets × 4 conditions and appends results to `simulation_results/run_01/all_simulation_results.csv`.

#### 6. Run the supplementary simulation (criteria modifications)

Update `pyproject.toml` for the supplementary run:

```toml
stimuli_for_llm = ["Original", "Abbreviations", "Expanded_terms"]
condition       = ["criteria"]
# subset_datasets = [...]   # the 12 datasets used in the paper
```

Then run:

```bash
python simulation/run.py run \
  data/synergy_dataset \
  simulation_results/run_varying_eligibility_criteria \
  data/inclusion_criteria_prepared.xlsx
```

#### 7. Run the R analysis

Open [`statistical_analysis.Rproj`](statistical_analysis.Rproj) in RStudio, then knit:

```r
renv::restore()           # first time only — installs all packages
rmarkdown::render("Analysis/analysis.Rmd")
```

This writes all figures and tables to `Report/results/`.

#### 8. Compile the manuscript

```bash
cd Report
latexmk -pdf -shell-escape main.tex
```

The compiled PDF is written to `Report/main.pdf`.

---

### Path B: Fast-Track Analysis Replication (OSF data)

> Downloads the pre-computed simulation outputs from OSF and runs only the R analysis and LaTeX compilation.  
> No Python, no API key, no simulation runtime required.

#### 1. Clone the repository and restore the R environment

```bash
git clone https://github.com/timovanommeren/thesis_timo.git
cd thesis_timo
```

Open [`statistical_analysis.Rproj`](statistical_analysis.Rproj) in RStudio, then run:

```r
renv::restore()
```

#### 2. Download the simulation data from OSF

The pre-computed simulation outputs are archived at:

> **OSF:** [https://osf.io/PLACEHOLDER](https://osf.io/PLACEHOLDER) — *DOI to be added upon publication*

Download the archive and extract it so your local directory looks like:

```
simulation_results/
├── run_01/
│   └── all_simulation_results.csv
└── run_varying_eligibility_criteria/
    └── all_simulation_results.csv
```

#### 3. Run the R analysis

In RStudio, knit the analysis notebook:

```r
rmarkdown::render("Analysis/analysis.Rmd")
```

All figures and tables are written to `Report/results/`. The notebook also writes `Report/values.tex` (in-text statistics) and `Report/results/bootstrap_results.tex` (results paragraph).

#### 4. Compile the manuscript

```bash
cd Report
latexmk -pdf -shell-escape main.tex
```

---

## Ethics and Data

The SYNERGY datasets used in this study are publicly available and distributed under their original licences via the [SYNERGY repository](https://github.com/asreview/synergy-dataset). Each dataset consists of titles and abstracts from published systematic review search results, with no personally identifiable information.

The LLM-generated synthetic abstracts produced during the simulation are entirely fabricated text and contain no real-world data.

---

## License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

---

## Permissions and Access

This repository will be publicly available at [https://github.com/timovanommeren/thesis_timo](https://github.com/timovanommeren/thesis_timo). Full responsibility for the content lies with Timo van Ommeren. For questions, contact [timovanommeren@gmail.com](mailto:timovanommeren@gmail.com).