from pathlib import Path
import pandas as pd
from matplotlib import pyplot as plt


### CREATE INDIVIDUAL RECALL PLOTS FUNCTION ###############################################################################################################################################

def individual_recall_plots(data_dir: Path, out_dir: Path, datasets: list[str] = None) -> None:
    """Create one recall plot per condition / run-parameter combination,
    overlaying the different datasets in the same figure.

    Parameters
    ----------
    data_dir : Path
        Root folder that contains the dataset sub-folders, each with a
        ``raw_simulations`` directory  (e.g. ``simulation_results/run_01``).
    out_dir : Path
        Folder where the output plots are saved (sub-folders per condition).
    datasets : list[str], optional
        List of dataset folder names to include. If None, all datasets are used.
    """

    # Wong colorblind-safe palette
    condition_colors = {
        'random': '#009E73',
        'llm': '#0072B2',
        'criteria': '#D55E00',
        'no_initialisation': '#E69F00',
    }

    # line styles to distinguish datasets within the same condition
    dataset_linestyles = ['-', '--', '-.', ':']

    condition_titles = {
        'random': 'True example',
        'llm': 'LLM',
        'criteria': 'Criteria only',
        'no_initialisation': 'Cold start',
    }

    # {(condition, run_params): {dataset_name: cumsum_series}}
    grouped: dict[tuple[str, str], dict[str, pd.Series]] = {}

    # ── collect data ────────────────────────────────────────────────
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
            # e.g. stem = "llm_run_1_IVs_1_100_0.0"
            parts = file.stem.split('_run_', 1)
            condition = parts[0]                    # e.g. "llm"
            run_params = parts[1] if len(parts) > 1 else file.stem  # e.g. "1_IVs_1_100_0.0"

            df = pd.read_csv(file)
            df = df.dropna(axis=0, subset=["training_set"])

            labels = df.loc[df["querier"].notna(), "label"].reset_index(drop=True)
            cumsum = labels.cumsum()

            key = (condition, run_params)
            grouped.setdefault(key, {})[dataset_name] = cumsum

    # ── plot one figure per (condition, run_params) ─────────────────
    out_dir.mkdir(parents=True, exist_ok=True)

    for (condition, run_params), dataset_cumsums in sorted(grouped.items()):
        color = condition_colors.get(condition, 'blue')

        plt.figure(figsize=(10, 6))

        max_len = max(len(c) for c in dataset_cumsums.values()) if dataset_cumsums else 0
        x_axis = range(1, max_len + 1)

        for i, (dataset_name, cumsum) in enumerate(sorted(dataset_cumsums.items())):
            ls = dataset_linestyles[i % len(dataset_linestyles)]
            # Pad shorter datasets with their last value (flat line)
            n_pad = max_len - len(cumsum)
            if n_pad > 0:
                last_val = cumsum.iloc[-1] if len(cumsum) > 0 else 0
                cumsum = pd.concat([cumsum, pd.Series([last_val] * n_pad)], ignore_index=True)
            plt.plot(x_axis, cumsum, label=dataset_name, color=color, linestyle=ls, linewidth=2)

        plt.xlabel('Number of Records Screened', fontsize=16)
        plt.ylabel('Number of Relevant Records Found', fontsize=16)
        plt.ylim(-1, 40)
        title = condition_titles.get(condition, condition)
        plt.title(title, fontweight='bold', fontsize=16)
        plt.legend(fontsize=12)
        plt.grid(False)
        plt.gca().set_facecolor('white')
        plt.gcf().set_facecolor('white')
        plt.tight_layout()

        cond_folder = out_dir / condition
        cond_folder.mkdir(parents=True, exist_ok=True)
        plot_path = cond_folder / f'{condition}_run_{run_params}_recall_plot.png'
        plt.savefig(plot_path)
        plt.close()

    print(f'Done – plots saved to {out_dir}')


if __name__ == '__main__':
    data_dir = Path(r'C:\Users\timov\Desktop\Utrecht\Utrecht\MSBBSS\thesis_timo\simulation_results\run_01')
    out_dir  = Path(r'C:\Users\timov\Desktop\Utrecht\Utrecht\MSBBSS\thesis_timo\Report\results\individual_recall_plots')
    datasets = ['Appenzeller-Herzog_2019', 'Brouwer_2019']

    individual_recall_plots(data_dir=data_dir, out_dir=out_dir, datasets=datasets)

    print("Done — individual recall plots saved.")
