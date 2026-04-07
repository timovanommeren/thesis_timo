from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


# ── style constants ────────────────────────────────────────────────────────────

_CONDITION_COLORS = {
    'random':            '#009E73',
    'llm':               '#0072B2',
    'criteria':          '#D55E00',
    'no_initialisation': '#E69F00',
}

_CONDITION_LABELS = {
    'random':            'Actual paper',
    'llm':               'LLM',
    'criteria':          'Eligibility criteria',
    'no_initialisation': 'Cold start',
}

ALL_CONDITIONS = ['random', 'llm', 'criteria', 'no_initialisation']


# ── helpers ────────────────────────────────────────────────────────────────────

def _build_cumsum(labels: pd.Series, target_len: int) -> pd.Series:
    n_pad = target_len - len(labels)
    if n_pad > 0:
        labels = pd.concat(
            [labels, pd.Series(np.zeros(n_pad, dtype=labels.dtype))],
            ignore_index=True,
        )
    return labels.iloc[:target_len].cumsum().reset_index(drop=True)


def _apply_formatting(ax, fig, x_scale: str, title: str, fontsize: int = 14,
                      show_legend: bool = True) -> None:
    xlabel = ('Proportion screened' if x_scale == 'proportion'
              else 'Records screened')
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel('Relevant records found', fontsize=fontsize)
    ax.tick_params(labelsize=max(fontsize - 1, 6))
    if title:
        ax.set_title(title, fontsize=fontsize)
    if show_legend:
        ax.legend(
            fontsize     = max(fontsize - 2, 6),
            loc          = 'best',
            framealpha   = 0.85,
            handlelength = 1.5,
            borderpad    = 0.3,
            labelspacing = 0.2,
        )
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
    ax.grid(False)
    ax.set_facecolor('white')
    fig.set_facecolor('white')
    plt.tight_layout(pad=0.3)


def _draw_diagonal(ax, nr: int, ds_size: int, x_scale: str, label: str,
                   alpha: float = 1.0) -> None:
    diag_x = ([j / nr for j in range(nr + 1)] if x_scale == 'proportion'
               else [j * ds_size / nr for j in range(nr + 1)])
    ax.step(diag_x, list(range(nr + 1)), where='post',
            color='black', linestyle='-', linewidth=1.5,
            alpha=alpha, label=label)


# ── main function ──────────────────────────────────────────────────────────────

def recall_plot(
    data_dir: Path,
    out_dir: Path,
    source_dir: Path,
    out_name: str,
    datasets: list[str],
    conditions: list[str] = None,
    x_max: int = None,
    show_diagonal: bool = True,
    x_scale: str = 'count',
    show_average: bool = True,
    per_run: bool = False,
    figsize: tuple[float, float] = (10, 6),
    fontsize: int = 14,
    show_legend: bool = True,
    show_title: bool = True,
    out_format: str = 'png',
    condition_labels: dict = None,
    diagonal_label: str = 'Random screening (no AI)',
) -> None:
    """Generic recall plot.

    Parameters
    ----------
    data_dir : Path
        Root folder containing dataset sub-folders, each with a
        ``raw_simulations/`` directory.
    out_dir : Path
        Root folder where output plots are saved.

        - ``per_run=False``: one ``.png`` per dataset, saved directly in
          ``out_dir``.
        - ``per_run=True``: one ``.png`` per (dataset, run-parameter
          combination), saved under
          ``out_dir / individual_runs / {dataset} /``.

    out_name : str
        Base name for the output file(s), without extension.
        For ``per_run=False`` the dataset name is appended:
        ``{out_name}_{dataset}.png``.
        For ``per_run=True``: ``{out_name}_run_{run_params}.png``.
    source_dir : Path
        Folder containing the original dataset CSV files
        (e.g. ``Synergy/synergy_dataset``).  Each file must be named
        ``{dataset}.csv``.  Row count gives the true dataset size N used
        for x-axis padding and the diagonal.
    datasets : list[str]
        Which dataset folder names to process.  In ``per_run=False`` mode each
        dataset gets its own plot.  In ``per_run=True`` mode datasets are
        overlaid within the same plot (one per condition × run), distinguished
        by line style.
    conditions : list[str], optional
        Which conditions to include.  Defaults to all four:
        ``['random', 'llm', 'criteria', 'no_initialisation']``.
    x_max : int, optional
        Clip the x-axis at this many records screened.  When *None* (default),
        the full x-axis is shown: each recall curve is padded with zeros so
        every curve spans the full original dataset size.
    show_diagonal : bool, optional
        Draw the random-screening diagonal (expected relevant records found
        without AI assistance).  Only drawn when ``x_max`` is *None*.
        Default *True*.
    x_scale : str, optional
        ``'count'`` (default): x-axis shows number of records screened.
        ``'proportion'``: x-axis shows proportion of records screened (0–1).
    show_average : bool, optional
        When *True* (default), individual run curves are drawn at low opacity
        and a bold mean curve is overlaid.  Ignored when ``per_run=True``
        (each plot already shows exactly one run per condition).
    per_run : bool, optional
        When *False* (default), all runs for each condition are combined into
        one plot per dataset.
        When *True*, one plot per (dataset × run-parameter combination) is
        created, overlaying all conditions.  Saved under
        ``out_dir / individual_runs / {dataset} /``.
    """
    if conditions is None:
        conditions = ALL_CONDITIONS
    _labels = {**_CONDITION_LABELS, **(condition_labels or {})}

    # ── 1. load data ──────────────────────────────────────────────────────────
    # dataset_totals[dataset] = N, read from the authoritative source CSV.
    #
    # condition_sizes[(condition, dataset)] = papers available to the querier:
    #   N−1 for 'random' (1 prior is drawn from the actual dataset, removing
    #   it from the screening pool); N for all other conditions (priors are
    #   synthetic and do not reduce the pool).
    #
    # per_run=False: runs[(condition, dataset)]             = [Series, ...]
    # per_run=True:  runs[(dataset, run_params)][condition] = Series
    runs: dict = {}
    dataset_totals:  dict[str, int] = {}
    condition_sizes: dict[tuple[str, str], int] = {}

    for dataset in datasets:
        source_csv = source_dir / f'{dataset}.csv'
        if not source_csv.exists():
            print(f'Warning: source file {source_csv} not found – skipping {dataset}.')
            continue
        N = len(pd.read_csv(source_csv))
        dataset_totals[dataset] = N
        for cond in ALL_CONDITIONS:
            condition_sizes[(cond, dataset)] = N - 1 if cond == 'random' else N

        raw_dir = data_dir / dataset / 'raw_simulations'
        if not raw_dir.exists():
            print(f'Warning: {raw_dir} not found – skipping {dataset}.')
            continue

        for csv_file in sorted(raw_dir.glob('*.csv')):
            stem_parts = csv_file.stem.split('_run_', 1)
            condition  = stem_parts[0]
            run_params = stem_parts[1] if len(stem_parts) > 1 else csv_file.stem

            if condition not in conditions:
                continue

            df     = pd.read_csv(csv_file).dropna(subset=['training_set'])
            labels = df.loc[df['querier'].notna(), 'label'].reset_index(drop=True)

            if per_run:
                runs.setdefault((dataset, run_params), {})[condition] = labels
            else:
                runs.setdefault((condition, dataset), []).append(labels)

    if not runs:
        print('No data found – nothing to plot.')
        return

    # ── 2. build cumsums ──────────────────────────────────────────────────────
    # Each condition uses its own target length (condition_sizes) so that
    # random is padded to N−1 and all other conditions to N.
    # per_run=False: cumsums[(condition, dataset)]             = [Series, ...]
    # per_run=True:  cumsums[(dataset, run_params)][condition] = Series

    if per_run:
        cumsums: dict = {
            (ds, rp): {
                cond: _build_cumsum(
                    lbl,
                    x_max if x_max is not None else condition_sizes.get((cond, ds), len(lbl)),
                )
                for cond, lbl in cond_map.items()
            }
            for (ds, rp), cond_map in runs.items()
        }
    else:
        cumsums: dict = {
            (cond, ds): [
                _build_cumsum(
                    lbl,
                    x_max if x_max is not None else condition_sizes.get((cond, ds), len(lbl)),
                )
                for lbl in label_list
            ]
            for (cond, ds), label_list in runs.items()
        }

    # ── 3. derive n_relevant per dataset (for diagonal) ───────────────────────

    n_relevant: dict[str, int] = {}
    if per_run:
        for (ds, _rp), cond_map in cumsums.items():
            for s in cond_map.values():
                n_relevant[ds] = max(n_relevant.get(ds, 0), int(round(s.iloc[-1])))
    else:
        for (_cond, ds), series_list in cumsums.items():
            n_relevant[ds] = max(
                n_relevant.get(ds, 0),
                int(round(max(s.iloc[-1] for s in series_list))),
            )

    # ── 4a. draw figures (per_run=False) ──────────────────────────────────────
    # One figure per dataset; conditions overlaid in condition colours.

    out_dir.mkdir(parents=True, exist_ok=True)

    if not per_run:
        for dataset in datasets:
            if dataset not in dataset_totals:
                continue

            fig, ax = plt.subplots(figsize=figsize)

            for condition in conditions:
                key = (condition, dataset)
                if key not in cumsums:
                    continue

                series_list = cumsums[key]
                color       = _CONDITION_COLORS.get(condition, 'grey')
                cond_label  = _labels.get(condition, condition)
                n           = len(series_list[0])
                cond_size   = condition_sizes.get((condition, dataset), n)
                x = (pd.Series(range(1, n + 1)) / cond_size if x_scale == 'proportion'
                     else pd.Series(range(1, n + 1)))

                if show_average:
                    for s in series_list:
                        ax.plot(x, s, color=color, alpha=0.07, linewidth=0.5)
                    mean_curve = pd.concat(series_list, axis=1).ffill(axis=0).mean(axis=1)
                    ax.plot(x, mean_curve, label=cond_label, color=color, linewidth=2.0, linestyle='--')
                else:
                    for i, s in enumerate(series_list):
                        ax.plot(x, s,
                                label=cond_label if i == 0 else '_nolegend_',
                                color=color, alpha=1, linewidth=1)

            if show_diagonal and x_max is None and n_relevant.get(dataset, 0) > 0:
                _draw_diagonal(ax, n_relevant[dataset], dataset_totals[dataset], x_scale,
                               label=diagonal_label)

            title = dataset if show_title else ''
            _apply_formatting(ax, fig, x_scale, title=title, fontsize=fontsize,
                              show_legend=show_legend)
            out_path = out_dir / f'{out_name}_{dataset}.{out_format}'
            fig.savefig(out_path, bbox_inches='tight')
            plt.close(fig)
            print(f'Saved → {out_path}')

    # ── 4b. draw figures (per_run=True) ───────────────────────────────────────
    # One figure per (dataset, run_params); all conditions overlaid.
    # Saved to: out_dir / individual_runs / {dataset} /
    #           {out_name}_run_{run_params}.png

    else:
        for (dataset, run_params), cond_map in sorted(cumsums.items()):
            fig, ax = plt.subplots(figsize=figsize)

            for condition in conditions:
                if condition not in cond_map:
                    continue
                s         = cond_map[condition]
                color     = _CONDITION_COLORS.get(condition, 'grey')
                cond_label = _CONDITION_LABELS.get(condition, condition)
                cond_size = condition_sizes.get((condition, dataset), len(s))
                n         = len(s)
                x = (pd.Series(range(1, n + 1)) / cond_size if x_scale == 'proportion'
                     else pd.Series(range(1, n + 1)))
                ax.plot(x, s, label=cond_label, color=color, linewidth=1.5)

            if show_diagonal and x_max is None and n_relevant.get(dataset, 0) > 0:
                _draw_diagonal(ax, n_relevant[dataset], dataset_totals[dataset], x_scale,
                               label=diagonal_label)

            _apply_formatting(ax, fig, x_scale,
                              title=dataset if show_title else '',
                              fontsize=fontsize,
                              show_legend=show_legend)

            ds_folder = out_dir / 'individual_runs' / dataset
            ds_folder.mkdir(parents=True, exist_ok=True)
            out_path = ds_folder / f'{out_name}_run_{run_params}.{out_format}'
            fig.savefig(out_path, bbox_inches='tight')
            plt.close(fig)
            print(f'Saved → {out_path}')


# ── example usage ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    RUN_01_DIR    = Path(r'C:\Users\timov\Desktop\Utrecht\Utrecht\MSBBSS\thesis_timo\simulation_results\run_01')
    FULL_RUNS_DIR = Path(r'C:\Users\timov\Desktop\Utrecht\Utrecht\MSBBSS\thesis_timo\simulation_results\Appenzeller-Herzog_full')
    OUT_DIR       = Path(r'C:\Users\timov\Desktop\Utrecht\Utrecht\MSBBSS\thesis_timo\Report\results')
    SOURCE_DIR    = Path(r'C:\Users\timov\Desktop\Utrecht\Utrecht\MSBBSS\thesis_timo\Synergy\synergy_dataset')
    APPEN_BROUWER      = ['Appenzeller-Herzog_2019', 'Brouwer_2019']
    ALL_DATASETS = ['Appenzeller-Herzog_2019', 'Bos_2018', 'Brouwer_2019', 'Chou_2003', 'Donners_2021', 'Hall_2012', 'Leenaars_2019', 'Leenaars_2020', 'Meijboom_2021', 'Menon_2022', 'Moran_2021', 'Muthu_2021', 'Nelson_2002', 'Oud_2018', 'Radjenovic_2013', 'Sep_2021', 'Smid_2020', 'Walker_2018', 'Wassenaar_2017', 'Wolters_2018', 'van_Dis_2020', 'van_de_Schoot_2018', 'van_der_Valk_2021', 'van_der_Waal_2022']

    # Figure 1 – single dataset, random condition only, full x-axis with diagonal
    recall_plot(
        data_dir=FULL_RUNS_DIR,
        out_dir=OUT_DIR,
        source_dir=SOURCE_DIR,
        out_name='figure_1',
        datasets=APPEN_BROUWER,
        conditions=['random', 'no_initialisation'],
        x_max=None,
        show_diagonal=True,
        x_scale='count',
        show_average=False,
        figsize=(3.2, 2.0),
        fontsize=9,
        show_title=False,
        show_legend=True,
        out_format='pdf',
        condition_labels={
            'random':            'AI-assisted',
            'no_initialisation': 'AI-assisted (cold start)',
        },
        diagonal_label='manual screening',
    )

    # # Figure 2 – both datasets (two separate plots), all conditions, mean + traces
    # recall_plot(
    #     data_dir=FULL_RUNS_DIR,
    #     out_dir=OUT_DIR,
    #     source_dir=SOURCE_DIR,
    #     out_name='figure_2',
    #     datasets=APPEN_BROUWER,
    #     conditions=None,
    #     x_max=None,
    #     show_diagonal=True,
    #     x_scale='proportion',
    #     show_average=False,
    #     figsize=(6, 5),
    #     fontsize=16,
    # )

    # Figure 3 – per-run plots for cherry-picking: one PDF per (dataset × run)
    #            saved under out_dir / individual_runs / {dataset} /
    # recall_plot(
    #     data_dir=RUN_01_DIR,
    #     out_dir=OUT_DIR / 'individual_runs',
    #     source_dir=SOURCE_DIR,
    #     datasets=ALL_DATASETS,
    #     out_name='figure_3',
    #     conditions=None,
    #     x_max=100,
    #     show_diagonal=True,
    #     x_scale='count',
    #     per_run=True,
    #     figsize=(3.2, 2.0),
    #     fontsize=9,
    #     show_title=False,
    #     show_legend=True,
    #     out_format='pdf',
    # )

    # # Figure 4 — 8-panel aggregate recall figure for thesis
    # # figsize matches the ~3.2 in render width (0.48 * 6.3 in A4 textwidth)
    # # fontsize=8 renders at ~8 pt in the final PDF (no rescaling needed)
    # recall_plot(
    #     data_dir=RUN_01_DIR,
    #     out_dir=Path(r'C:\Users\timov\Desktop\Utrecht\Utrecht\MSBBSS\thesis_timo\Report\results\aggregate_recall_plots'),
    #     source_dir=SOURCE_DIR,
    #     datasets=ALL_DATASETS,
    #     out_name='figure_4',
    #     conditions=None,
    #     x_max=100,
    #     show_diagonal=False,
    #     x_scale='count',
    #     show_average=True,
    #     figsize=(3.2, 2.0),
    #     fontsize=9,
    #     show_title=False,
    #     show_legend=True,
    #     out_format='pdf',
    # )
