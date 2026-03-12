from pathlib import Path
import pandas as pd
from matplotlib import pyplot as plt


### AGGREGATED RECALL PLOTS PER CONDITION #################################################################################
# Produces one plot per condition (random / llm / criteria / no_initialisation).
# Each plot overlays multiple datasets, showing the mean cumulative recall
# (aggregated across all runs) with ±1 SE shading.
###########################################################################################################################

def aggregated_condition_plots(data_dir: Path, out_dir: Path, datasets: list[str] = None) -> None:
    """Create one aggregated recall plot per condition, overlaying the
    mean (±SE) recall curve for each dataset across all runs.

    Parameters
    ----------
    data_dir : Path
        Root folder containing dataset sub-folders, each with a
        ``raw_simulations`` directory.
    out_dir : Path
        Folder where the output plots are saved (one file per condition).
    datasets : list[str], optional
        Dataset folder names to include. If None, all datasets are used.
    """

    # Wong colorblind-safe palette
    condition_colors = {
        'random': '#009E73',
        'llm': '#0072B2',
        'criteria': '#D55E00',
        'no_initialisation': '#E69F00',
    }

    dataset_linestyles = ['-', '--', '-.', ':']

    condition_titles = {
        'random': 'True example',
        'llm': 'LLM',
        'criteria': 'Criteria only',
        'no_initialisation': 'Cold start',
    }

    # {(condition, dataset_name): [cumsum_series, ...]}  — one series per run
    grouped: dict[tuple[str, str], list[pd.Series]] = {}

    # ── collect all runs ────────────────────────────────────────────────────
    for dataset_folder in sorted(data_dir.iterdir()):
        if not dataset_folder.is_dir():
            continue

        if datasets is not None and dataset_folder.name not in datasets:
            continue

        raw_output = dataset_folder / 'raw_simulations'
        if not raw_output.exists():
            continue

        dataset_name = dataset_folder.name

        for file in sorted(raw_output.glob('*.csv')):
            parts = file.stem.split('_run_', 1)
            condition = parts[0]

            df = pd.read_csv(file)
            df = df.dropna(axis=0, subset=["training_set"])

            cumsum = df.loc[df["querier"].notna(), "label"].reset_index(drop=True).cumsum()

            key = (condition, dataset_name)
            grouped.setdefault(key, []).append(cumsum)

    # ── aggregate: mean and SE across runs per (condition, dataset) ─────────
    def _aggregate(cumsums: list[pd.Series]) -> pd.DataFrame:
        df = pd.concat(cumsums, axis=1)
        df = df.ffill(axis=0)   # flat line beyond each run's natural end
        return pd.DataFrame({'mean': df.mean(axis=1), 'se': df.sem(axis=1)})

    agg: dict[tuple[str, str], pd.DataFrame] = {
        key: _aggregate(cumsums) for key, cumsums in grouped.items()
    }

    # ── determine which conditions and datasets are present ─────────────────
    conditions = sorted({cond for cond, _ in agg.keys()})
    dataset_order = sorted({ds for _, ds in agg.keys()})

    # ── plot one figure per condition ────────────────────────────────────────
    out_dir.mkdir(parents=True, exist_ok=True)

    for condition in conditions:
        color = condition_colors.get(condition, 'blue')

        plt.figure(figsize=(10, 6))

        for i, dataset_name in enumerate(dataset_order):
            key = (condition, dataset_name)
            if key not in agg:
                continue

            agg_df = agg[key].iloc[:100]   # truncate to first 100 records screened
            x = range(1, len(agg_df) + 1)
            ls = dataset_linestyles[i % len(dataset_linestyles)]

            plt.plot(x, agg_df['mean'],
                     label=dataset_name, color=color, linestyle=ls, linewidth=2)
            plt.fill_between(x,
                             agg_df['mean'] - agg_df['se'],
                             agg_df['mean'] + agg_df['se'],
                             color=color, alpha=0.3)

        plt.xlabel('Number of Records Screened', fontsize=16)
        plt.ylabel('Number of Relevant Records Found', fontsize=16)
        plt.xlim(-2, 105)
        plt.ylim(-1, 40)
        title = condition_titles.get(condition, condition)
        plt.title(title, fontweight='bold', fontsize=16)
        plt.legend(fontsize=12)
        plt.grid(False)
        plt.gca().set_facecolor('white')
        plt.gcf().set_facecolor('white')
        plt.tight_layout()

        plot_path = out_dir / f'{condition}_aggregated_recall_plot.png'
        plt.savefig(plot_path)
        plt.close()

    print(f'Done – plots saved to {out_dir}')


if __name__ == '__main__':
    data_dir = Path(r'C:\Users\timov\Desktop\Utrecht\Utrecht\MSBBSS\thesis_timo\simulation_results\run_01')
    out_dir  = Path(r'C:\Users\timov\Desktop\Utrecht\Utrecht\MSBBSS\thesis_timo\Report\results\aggregated_condition_plots')
    datasets = ['Appenzeller-Herzog_2019', 'Brouwer_2019']

    aggregated_condition_plots(data_dir=data_dir, out_dir=out_dir, datasets=datasets)

    print("Done — aggregated condition plots saved.")
