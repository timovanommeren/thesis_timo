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


def evaluate_simulation(simulation_results: dict, dataset: pd.DataFrame, dataset_llms: pd.DataFrame, dataset_criteria: pd.DataFrame, prior_idx: list, criterium: str, n_abstracts: int, length_abstracts: int, llm_temperature: float, papers_screened: dict, out_dir: Path, run: int) -> None:

    ### PREPARE DATA FOR EVALUATION ############################################################################################################

    (dataset_names, simulation_results), = simulation_results.items() # comma enforces unpacking single item (since were doing only one dataset at a time)

    # pad the labels to ensure accurate simulation results (see Report section 4.2.1)
    # padded_labels_random = pad_labels(simulation_results['random']["label"].reset_index(drop=True), len([prior_idx]), len(dataset), stop_at_n)
    # padded_labels_llm = pad_labels(simulation_results['llm']["label"].reset_index(drop=True), 0, len(dataset_llms['dataset']), stop_at_n) # note that in the llm condition, the priors aren't part of the analyzed set so should not be considered in padding
    # padded_labels_criteria = pad_labels(simulation_results['criteria']["label"].reset_index(drop=True), 0, len(dataset_criteria['dataset']), stop_at_n) # idem for criteria condition
    # padded_labels_no_initialisation = pad_labels(simulation_results['no_initialisation']["label"].reset_index(drop=True), 0, len(dataset), stop_at_n)

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
        'Random Initialization':   _pad_cumsum(simulation_results['random']["label"].reset_index(drop=True).iloc[:papers_screened['random']].cumsum(), max_ps),
        'LLM Initialization':      _pad_cumsum(simulation_results['llm']["label"].reset_index(drop=True).iloc[:papers_screened['llm']].cumsum(), max_ps),
        'Criteria Initialization': _pad_cumsum(simulation_results['criteria']["label"].reset_index(drop=True).iloc[:papers_screened['criteria']].cumsum(), max_ps),
        'No Initialization':       _pad_cumsum(simulation_results['no_initialisation']["label"].reset_index(drop=True).iloc[:papers_screened['no_initialisation']].cumsum(), max_ps),
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
    )
    
    ############################################################################################################################################
 
 
 
 
 
    #### CALCULATE OUTCOME METRICS ############################################################################################################

    # Calculate the number of relevant records found at TDD threshold (capped at 100 rows)
    td_random = papers_found({'record_id': simulation_results['random']['record_id'].iloc[:100], 'label': simulation_results['random']['label'].iloc[:100]}, papers_screened['random'])[1]
    td_llm = papers_found({'record_id': simulation_results['llm']['record_id'].iloc[:100], 'label': simulation_results['llm']['label'].iloc[:100]}, papers_screened['llm'])[1]
    td_criteria = papers_found({'record_id': simulation_results['criteria']['record_id'].iloc[:100], 'label': simulation_results['criteria']['label'].iloc[:100]}, papers_screened['criteria'])[1]
    td_no_initialisation = papers_found({'record_id': simulation_results['no_initialisation']['record_id'].iloc[:100], 'label': simulation_results['no_initialisation']['label'].iloc[:100]}, papers_screened['no_initialisation'])[1]

    # Calculate the number of records that need to be screened to find the first relevant record (ATD)
    atd_random = metrics._average_time_to_discovery(papers_found({'record_id': simulation_results['random']['record_id'].iloc[:100], 'label': simulation_results['random']['label'].iloc[:100]}, papers_screened['random'])[0])
    atd_llm = metrics._average_time_to_discovery(papers_found({'record_id': simulation_results['llm']['record_id'].iloc[:100], 'label': simulation_results['llm']['label'].iloc[:100]}, papers_screened['llm'])[0])
    atd_criteria = metrics._average_time_to_discovery(papers_found({'record_id': simulation_results['criteria']['record_id'].iloc[:100], 'label': simulation_results['criteria']['label'].iloc[:100]}, papers_screened['criteria'])[0])
    atd_no_initialisation = metrics._average_time_to_discovery(papers_found({'record_id': simulation_results['no_initialisation']['record_id'].iloc[:100], 'label': simulation_results['no_initialisation']['label'].iloc[:100]}, papers_screened['no_initialisation'])[0])

    ############################################################################################################################################





    ### SAVE METRICS TO MASTER RESULTS FILE ###################################################################################################

    results_row = []

    for condition, metrics_dict in [
        ('random', {'papers_found': td_random, 
                    'atd': atd_random}),
        ('llm', {'papers_found': td_llm,
                'atd': atd_llm}),
        ('criteria', {'papers_found': td_criteria,
                      'atd': atd_criteria}),
        ('no_initialisation', {'papers_found': td_no_initialisation,
                    'atd': atd_no_initialisation})
    ]:
        for metric_name, metric_value in metrics_dict.items():
            
            # Determine if parameters apply to this condition
            is_llm = (condition == 'llm')
            is_criteria = (condition == 'criteria')
            
            results_row.append({
                'dataset': dataset_names,
                'condition': condition,
                'metric': metric_name,
                'value': metric_value,
                'criterium': criterium if (is_llm or is_criteria) else np.nan,
                'n_abstracts': n_abstracts if is_llm else np.nan,
                'length_abstracts': length_abstracts if is_llm else np.nan,
                'llm_temperature': llm_temperature if is_llm else np.nan,
                'tdd@': papers_screened[condition],
                'timestamp': pd.Timestamp.now().isoformat(),
                'run': run,  # replicate ID
                'n_trials': papers_screened[condition],
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



def recall_plot(df_cumsum: pd.DataFrame, dataset_names: str, criterium: str, n_abstracts: int, length_abstracts: int, llm_temperature: float, out_dir: Path, run: int, papers_screened: dict):

    plt.figure(figsize=(10, 6))

    # x-axis runs to max papers_screened across all conditions (df_cumsum already padded)
    x_axis = range(1, len(df_cumsum['Random Initialization']) + 1)
    plt.plot(x_axis, df_cumsum['Random Initialization'], label='Random Initialization', color='#2ab07f')
    plt.plot(x_axis, df_cumsum['LLM Initialization'], label='LLM Initialization', color='#482475')
    plt.plot(x_axis, df_cumsum['Criteria Initialization'], label='Criteria Initialization', color='#2d708e')
    plt.plot(x_axis, df_cumsum['No Initialization'], label='No Initialization', color='#bddf26')

    # Add per-condition dashed stop lines when conditions have different screening lengths
    ps_values = list(papers_screened.values())
    if len(set(ps_values)) > 1:
        _stop_colors = {'random': '#2ab07f', 'llm': '#482475', 'criteria': '#2d708e', 'no_initialisation': '#bddf26'}
        for cond, ps_val in papers_screened.items():
            plt.axvline(x=ps_val, color=_stop_colors[cond], linestyle='--', alpha=0.4)

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


