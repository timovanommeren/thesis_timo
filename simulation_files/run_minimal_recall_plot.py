from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from metrics import pad_labels


def minimal_recall_plot(data_dir: Path, datasets: list[str], out_path: Path) -> None:
    """Create a single recall plot for the random ('minimal') condition,
    overlaying multiple datasets.  The x-axis shows the proportion of papers
    screened (0–1) and the y-axis shows the proportion of relevant papers
    found (recall, 0–1).  A diagonal line represents screening without
    AI-assistance (pure random).

    Parameters
    ----------
    data_dir : Path
        Root folder containing dataset sub-folders, each with a
        ``raw_simulation_minimal.csv`` file.
    datasets : list[str]
        Dataset folder names to include in the plot.
    out_path : Path
        Full file path (including filename) where the plot is saved.
    """

    dataset_colors = ['#2ab07f', 'green', 'orange', 'red', 'purple', 'brown']
    dataset_linestyles = ['-', '--', '-.', ':']

    plt.figure(figsize=(10, 6))

    max_relevant = 0

    # ── plot each dataset ───────────────────────────────────────────
    for i, name in enumerate(datasets):
        csv_path = data_dir / name / 'raw_simulations\\random_run_1_IVs_Original_4_500_0.4.csv'
        df_full = pd.read_csv(csv_path)
        num_records = len(df_full)
        num_priors = df_full['training_set'].isna().sum()

        df = df_full.dropna(subset=['training_set'])
        labels = df.loc[df['querier'].notna(), 'label'].reset_index(drop=True)
        n_relevant = labels.sum()
        max_relevant = max(max_relevant, n_relevant)

        padded_labels = pad_labels(labels, num_priors, num_records, stop_at_n=-1)
        n_total = len(padded_labels)

        cumsum = padded_labels.cumsum().reset_index(drop=True)
        proportion_screened = (pd.Series(range(1, n_total + 1))) / n_total

        color = dataset_colors[i % len(dataset_colors)]
        ls = dataset_linestyles[i % len(dataset_linestyles)]
        plt.plot(proportion_screened, cumsum, label=name, color=color, linestyle=ls)

        # ── step diagonal: expected count under random screening ─────
        baseline_x = [j / n_relevant for j in range(n_relevant + 1)]
        baseline_y = list(range(n_relevant + 1))
        baseline_label = 'Screening without AI-assistance' if i == 0 else '_nolegend_'
        plt.step(baseline_x, baseline_y, where='post', color='black', linestyle='-',
                 alpha=1, label=baseline_label)

    # ── formatting ──────────────────────────────────────────────────
    plt.xlabel('Proportion of Papers Screened', fontsize=16)
    plt.ylabel('Number of Relevant Papers Found', fontsize=16)
    plt.title('Recall', fontsize=18)
    plt.xlim(0, 1)
    plt.ylim(0, max_relevant)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    # plt.legend(fontsize=14)
    plt.grid(False)
    plt.gca().set_facecolor('white')
    plt.gcf().set_facecolor('white')
    plt.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()
    print(f'Done – plot saved to {out_path}')


if __name__ == '__main__':
    data_dir = Path(r'C:\Users\timov\Desktop\Utrecht\Utrecht\MSBBSS\thesis_timo\simulation_results\full_run_Appel_Brouwer')
    datasets = ['Appenzeller-Herzog_2019']
    out_path = Path(r'C:\Users\timov\Desktop\Utrecht\Utrecht\MSBBSS\thesis_timo\Report\results\minimal_recall_plot.png')

    minimal_recall_plot(data_dir=data_dir, datasets=datasets, out_path=out_path)
