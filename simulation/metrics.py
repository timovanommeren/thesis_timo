import numpy as np
import pandas as pd
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # headless, no Tk
import matplotlib.pyplot as plt


from asreview.metrics import loss
from asreview.metrics import ndcg
from asreviewcontrib.insights import algorithms
from asreviewcontrib.insights import metrics


CONDITION_LABELS = {
    'random':            'Random Initialization',
    'llm':               'LLM Initialization',
    'criteria':          'Criteria Initialization',
    'no_initialisation': 'No Initialization',
}

CONDITION_COLORS = {
    'random':            '#2ab07f',
    'llm':               '#482475',
    'criteria':          '#2d708e',
    'no_initialisation': '#bddf26',
}


def evaluate_simulation(simulation_results: dict, criterium: str, n_abstracts: int, length_abstracts: int, llm_temperature: float, papers_screened: dict, out_dir: Path, run: int, condition: list[str] = None) -> None:

    ALL_CONDITIONS = ['random', 'llm', 'criteria', 'no_initialisation']
    if condition is None:
        condition = ALL_CONDITIONS

    ### PREPARE DATA FOR EVALUATION ############################################################################################################

    (dataset_names, simulation_results), = simulation_results.items() # comma enforces unpacking single item (since were doing only one dataset at a time)

    # Compute per-condition cumsums sliced to their own papers_screened, then pad
    # shorter conditions with their final value (flat line) so all series reach max_ps.
    max_ps = max(papers_screened.values())

    def _pad_cumsum(series: pd.Series, length: int) -> pd.Series:
        s = series.reset_index(drop=True)
        n_pad = length - len(s)
        if n_pad <= 0:
            return s
        last = s.iloc[-1] if len(s) > 0 else 0
        return pd.concat([s, pd.Series([last] * n_pad)], ignore_index=True)

    df_cumsum = pd.DataFrame({
        CONDITION_LABELS[cond]: _pad_cumsum(simulation_results[cond]["label"].reset_index(drop=True).iloc[:papers_screened[cond]].cumsum(), max_ps)
        for cond in condition
    })

    print(f"Number of papers screened per condition: {papers_screened}")
    ############################################################################################################################################




    ### GENERATE PLOTS #########################################################################################################################

    recall_plot(
        df_cumsum=df_cumsum,
        dataset_names=dataset_names,
        criterium=criterium,
        n_abstracts=n_abstracts,
        length_abstracts=length_abstracts,
        llm_temperature=llm_temperature,
        out_dir=out_dir,
        run=run,
        papers_screened=papers_screened,
        condition=condition,
    )

    ############################################################################################################################################




    #### CALCULATE OUTCOME METRICS ############################################################################################################

    td, atd = {}, {}
    for cond in condition:
        pf_result = papers_found(
            {'record_id': simulation_results[cond]['record_id'].iloc[:100],
             'label':     simulation_results[cond]['label'].iloc[:100]},
            papers_screened[cond]
        )
        td[cond]  = pf_result[1]
        atd[cond] = metrics._average_time_to_discovery(pf_result[0])

    ############################################################################################################################################




    ### SAVE METRICS TO MASTER RESULTS FILE ###################################################################################################

    results_row = []

    for cond in condition:
        for metric_name, metric_value in [('papers_found', td[cond]), ('atd', atd[cond])]:

            # Determine if parameters apply to this condition
            is_llm = (cond == 'llm')
            is_criteria = (cond == 'criteria')

            results_row.append({
                'dataset': dataset_names,
                'condition': cond,
                'metric': metric_name,
                'value': metric_value,
                'criterium': criterium if (is_llm or is_criteria) else np.nan,
                'n_abstracts': n_abstracts if is_llm else np.nan,
                'length_abstracts': length_abstracts if is_llm else np.nan,
                'llm_temperature': llm_temperature if is_llm else np.nan,
                'tdd@': papers_screened[cond],
                'timestamp': pd.Timestamp.now().isoformat(),
                'run': run,  # replicate ID
                'n_trials': papers_screened[cond],
            })

    # Append to master results file
    df_results = pd.DataFrame(results_row)
    master_file = out_dir / 'all_simulation_results.csv'
    df_results.to_csv(master_file, mode='a', header=not master_file.exists(), index=False)

    ############################################################################################################################################




### HELPER FUNCTIONS ##############################################################################################################################################



def pad_labels(labels, num_priors, num_records, stop_at_n):

    # if there is a stopping criterion, then only pad until stop_at_n
    if stop_at_n != -1:

        #first check whether len(labels) is already >= stop_at_n (this may occur in no initialization condition, runs at least until the first relevant is found). If true truncate to stop_at_n
        if len(labels) >= stop_at_n:
            return pd.Series(labels.tolist()[:stop_at_n])
        else:
            return pd.Series(labels.tolist() + np.zeros(stop_at_n - len(labels)).tolist())

    else:
        return pd.Series(labels.tolist() + np.zeros(num_records - len(labels) - num_priors).tolist())



def papers_found(results, threshold):
    all_tdd = metrics._time_to_discovery(results['record_id'], results['label'])
    count = sum(iter_idx <= threshold for _, iter_idx in all_tdd)
    return all_tdd, count



def recall_plot(df_cumsum: pd.DataFrame, dataset_names: str, criterium: str, n_abstracts: int, length_abstracts: int, llm_temperature: float, out_dir: Path, run: int, papers_screened: dict, condition: list[str] = None):

    ALL_CONDITIONS = ['random', 'llm', 'criteria', 'no_initialisation']
    if condition is None:
        condition = ALL_CONDITIONS

    plt.figure(figsize=(10, 6))

    x_axis = range(1, len(df_cumsum.iloc[:, 0]) + 1)
    for cond in condition:
        label = CONDITION_LABELS[cond]
        plt.plot(x_axis, df_cumsum[label], label=label, color=CONDITION_COLORS[cond])

    # Add per-condition dashed stop lines when conditions have different screening lengths
    ps_values = list(papers_screened.values())
    if len(set(ps_values)) > 1:
        for cond, ps_val in papers_screened.items():
            plt.axvline(x=ps_val, color=CONDITION_COLORS[cond], linestyle='--', alpha=0.4)

    plt.xlabel('Number of Records Screened')
    plt.ylabel('Number of Relevant Records Found')
    plt.title('Number of Relevant Records Found vs. Number of Records Screened')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    #create subfolder for recalls plots
    recalls_folder = out_dir / dataset_names / 'recalls_plots'
    recalls_folder.mkdir(parents=True, exist_ok=True)

    # save plot to output_path
    plot_path = recalls_folder / f'recall_plot_run_{run}_IVs_{criterium}_{n_abstracts}_{length_abstracts}_{llm_temperature}.png'
    plt.savefig(plot_path)
    plt.close()
