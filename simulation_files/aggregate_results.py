from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def aggregate_recall_plots(datasets, out_dir: Path, x_max: int = None) -> None:
  # accept either {name: dataframe} dict or [name, ...] list
  dataset_names_iter = datasets.keys() if isinstance(datasets, dict) else datasets

  for name in dataset_names_iter:

    raw_output = out_dir / name / 'raw_simulations'

    random_runs = []
    llm_runs = []
    criteria_runs = []
    no_initialisation_runs = []
    dataset_size = None  # derived from no_initialisation files (no priors → row count == dataset size)

    # loop over all csv files in raw_output
    for file in Path(raw_output).glob('*.csv'):
        df = pd.read_csv(file)

        # capture total record count from no_initialisation (no priors, so len == dataset size)
        if 'no_initialisation' in file.name and dataset_size is None:
            dataset_size = len(df)

        #drop the first rows if it contains NaN
        df = df.dropna(axis=0, subset=["training_set"])

        # check if the file is random priors, llm or no priors
        if 'random' in file.name:
            random_runs.append(df)
        elif 'llm' in file.name:
            llm_runs.append(df)
        elif 'criteria' in file.name:
            criteria_runs.append(df)
        elif 'no_initialisation' in file.name:
            no_initialisation_runs.append(df)

    # For each method, use all rows per run (each condition stopped naturally via its NLabeled/IsFittable
    # stopper).  Runs within a condition may have different lengths (especially no_initialisation whose
    # random-querier phase has variable length), so forward-fill shorter runs before averaging so that
    # the cumsum stays flat (no new relevant records found) rather than becoming NaN.

    def _agg_cumsums(cumsums_list):
        df = pd.concat(cumsums_list, axis=1)
        df = df.ffill(axis=0)   # flat line for rows beyond each run's natural end
        return pd.DataFrame({'Cumulative Sum': df.mean(axis=1), 'SE': df.sem(axis=1)})

    random_cumsums = [df.loc[df["querier"].notna(), "label"].reset_index(drop=True).cumsum() for df in random_runs]
    agg_random = _agg_cumsums(random_cumsums)

    llm_cumsums = [df.loc[df["querier"].notna(), "label"].reset_index(drop=True).cumsum() for df in llm_runs]
    agg_llm = _agg_cumsums(llm_cumsums)

    criteria_cumsums = [df.loc[df["querier"].notna(), "label"].reset_index(drop=True).cumsum() for df in criteria_runs]
    agg_criteria = _agg_cumsums(criteria_cumsums)

    no_initialisation_cumsums = [df.loc[df["querier"].notna(), "label"].reset_index(drop=True).cumsum() for df in no_initialisation_runs]
    agg_no_initialisation = _agg_cumsums(no_initialisation_cumsums)


    ### PLOT THE AGGREGATED RECALL CURVES ############################################################################################

    # x-axis spans the full original dataset size (from no_initialisation, which has no priors).
    # LLM/criteria datasets include extra prior records — truncate those to n_points.
    # Random uses 1 prior so its curve naturally ends one record earlier; plot it at its natural length.
    n_points = len(datasets[name]) if isinstance(datasets, dict) else dataset_size
    if x_max is not None:
        n_points = min(n_points, x_max)

    def _fit_to_length(agg_df, target_len):
        if len(agg_df) > target_len:
            return agg_df.iloc[:target_len].reset_index(drop=True)
        n_pad = target_len - len(agg_df)
        if n_pad == 0:
            return agg_df
        last_mean = agg_df['Cumulative Sum'].iloc[-1]
        padding = pd.DataFrame({'Cumulative Sum': [last_mean] * n_pad, 'SE': [0.0] * n_pad})
        return pd.concat([agg_df, padding], ignore_index=True)

    agg_random            = _fit_to_length(agg_random,            n_points)
    agg_llm               = _fit_to_length(agg_llm,               n_points)
    agg_criteria          = _fit_to_length(agg_criteria,          n_points)
    agg_no_initialisation = _fit_to_length(agg_no_initialisation, n_points)

    plt.figure(figsize=(10, 6))
    x_axis = range(1, n_points + 1)

    # Wong colorblind-safe palette
    plt.plot(x_axis, agg_random['Cumulative Sum'], label='True Example Condition', color='#009E73')
    plt.fill_between(x_axis, agg_random['Cumulative Sum'] - agg_random['SE'], agg_random['Cumulative Sum'] + agg_random['SE'], color='#009E73', alpha=0.3)

    plt.plot(x_axis, agg_llm['Cumulative Sum'], label='LLM Condition', color='#0072B2')
    plt.fill_between(x_axis, agg_llm['Cumulative Sum'] - agg_llm['SE'], agg_llm['Cumulative Sum'] + agg_llm['SE'], color='#0072B2', alpha=0.3)

    plt.plot(x_axis, agg_criteria['Cumulative Sum'], label='Inclusion Criteria Condition', color='#D55E00')
    plt.fill_between(x_axis, agg_criteria['Cumulative Sum'] - agg_criteria['SE'], agg_criteria['Cumulative Sum'] + agg_criteria['SE'], color='#D55E00', alpha=0.3)

    plt.plot(x_axis, agg_no_initialisation['Cumulative Sum'], label='Cold Start Condition', color='#E69F00')
    plt.fill_between(x_axis, agg_no_initialisation['Cumulative Sum'] - agg_no_initialisation['SE'], agg_no_initialisation['Cumulative Sum'] + agg_no_initialisation['SE'], color='#E69F00', alpha=0.3)

    # Step diagonal: expected relevant found under pure random screening
    n_relevant = int(datasets[name]['label_included'].sum()) if isinstance(datasets, dict) else None
    if n_relevant and x_max is None:
        baseline_x = [j * n_points / n_relevant for j in range(n_relevant + 1)]
        baseline_y = list(range(n_relevant + 1))
        plt.step(baseline_x, baseline_y, where='post', color='black', linestyle='-',
                 alpha=1)

    plt.xlabel('Number of Records Screened', fontsize=16)
    plt.ylabel('Number of Relevant Records Found', fontsize=16)
    plt.title('Recall', fontsize=16)
    plt.legend(fontsize=14)
    plt.grid(False)
    plt.gca().set_facecolor('white')
    plt.gcf().set_facecolor('white')
    plt.tight_layout()

    # save plot to output_path
    plot_path = out_dir / name / 'aggregate_recall_plot.png'
    plt.savefig(plot_path)
    plt.close()

# parameters for evaluation

out_dir = Path(r'C:\\Users\\timov\\Desktop\\Utrecht\\Utrecht\\MSBBSS\\thesis_timo\\simulation_results\\full_run_Appel_Brouwer')
in_dir  = Path(r'C:\\Users\\timov\\Desktop\\Utrecht\\Utrecht\\MSBBSS\\thesis_timo\\Synergy\\synergy_dataset')

dataset_names = [p.name for p in out_dir.iterdir() if p.is_dir()]
datasets = {name: pd.read_csv(in_dir / f'{name}.csv') for name in dataset_names}

print(list(datasets.keys()))

aggregate_recall_plots(
    datasets=datasets,
    out_dir=out_dir
)

print("Aggregation complete.")