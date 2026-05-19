import typer 
from pathlib import Path
import pandas as pd
import itertools
import time


from simulation import run_simulation
from recall_plots import recall_plot
from config import load_pyproject_config

app = typer.Typer()


def _format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"

@app.command()
def run(
    
    in_dir: Path = typer.Argument(..., exists=True, file_okay=False, dir_okay=True, readable=True,
                                  help="Folder containing datasets."),
    out_dir: Path = typer.Argument(..., exists=False, file_okay=False, dir_okay=True, readable=True,
                                  help="Root folder for all outputs."),
    criteria_path: Path = typer.Argument(..., exists=True, file_okay=True, dir_okay=False, readable=True,
                                  help="Path to criteria file for LLM."),
    #stimuli_for_llm: str = typer.Argument(..., help="Space-separated list of stimuli for LLM.")
):
  
    ### LOAD CONFIG FROM TOML FILE ##########################################################################
    
    config = load_pyproject_config()
    
    #from simulation import pad_labels
    stimuli_for_llm = config.get("stimuli_for_llm")
    
    if isinstance(stimuli_for_llm, str):
        stimuli_for_llm = [stimuli_for_llm] # Convert single string to list if only one stimulus is provided as a string

    # Parameters for running simulations
    n_simulations = config.get("n_simulations")
    stop_at_n = config.get("stop_at_n") # set to -1 to stop when all relevant records are found
    condition = config.get("condition") # set to None to run all four conditions
    
    # Parameters for simulation (IVs)
    n_abstracts = config.get("n_abstracts")
    length_abstracts = config.get("length_abstracts")
    llm_temperature = config.get("llm_temperature")


  
    ### RETRIEVE INPUT #######################################################################################
    
    # # Resolve in_dir relative to project root (parent of simulation_files/) !!! DOESNT WORK YET
    # script_dir = Path(__file__).parent
    # project_root = script_dir.parent
    # in_dir = project_root / in_dir
    
    # import the paths of all files in the input directory
    data_paths = [f for f in Path(in_dir).iterdir() if f.is_file()]
    
    # import all the of the datasets
    datasets = {file.stem: pd.read_csv(file) for file in data_paths}

    ### Create smaller subset of datasets for testing ####################################################
    subset_keys = config.get("subset_datasets", None)
    if subset_keys is not None:
        datasets = {k: datasets[k] for k in subset_keys if k in datasets}
    
    print(f"Running simulations on datasets: {list(datasets.keys())}")

    ##########################################################################################################





    ### CREATE OUTPUT DIRECTORIES ############################################################################

    # create output directories for each dataset
    for dataset in datasets:
        (out_dir / dataset).mkdir(parents=True, exist_ok=True)
        

        
    # load synergy metadata (path relative to this script's location)
    synergy_metadata = pd.read_excel(criteria_path)

    ##########################################################################################################






    ### SIMULATE AND EVALUATE ###############################################################################
    
    # Generate all combinations of IVs
    iv_combinations = list(itertools.product(
        n_abstracts,
        length_abstracts,
        llm_temperature
    ))
    
    print(f"Running {n_simulations} simulations for {len(iv_combinations)} combinations of LLM-specific variables and {len(stimuli_for_llm)} stimuli for {len(datasets)} datasets for all four conditions")
    print(f"Total simulations: {n_simulations * len(iv_combinations) * len(stimuli_for_llm) * len(datasets) * len(condition)}")
    
    
    total_runs = n_simulations * len(stimuli_for_llm) * len(iv_combinations)
    progress_every = max(1, total_runs // 20)  # report roughly every 5%
    start_time = time.perf_counter()

    for run in range(n_simulations):
        
        for stimulus_idx, stimulus_for_llm in enumerate(stimuli_for_llm):
        
            for combo_idx, (n_abs, len_abs, temp) in enumerate(iv_combinations):
                global_run = (
                    run * len(stimuli_for_llm) * len(iv_combinations)
                    + stimulus_idx * len(iv_combinations)
                    + combo_idx
                    + 1
                )
                print(f"\nIV Combination {combo_idx + 1}/{len(iv_combinations)}: "
                    f"n_abstracts={n_abs}, length={len_abs}, temperature={temp}."
                    f"From simulation {global_run} of {total_runs}.")
            
                run_simulation(
                    datasets=datasets,
                    criterium=stimulus_for_llm,
                    condition=condition,
                    out_dir=out_dir,
                    metadata=synergy_metadata,
                    n_abstracts=n_abs,
                    length_abstracts=len_abs,
                    llm_temperature=temp,
                    run=global_run,
                    stop_at_n=stop_at_n
                )

                if global_run % progress_every == 0 or global_run == total_runs:
                    elapsed = time.perf_counter() - start_time
                    avg_time_per_run = elapsed / global_run
                    eta_seconds = avg_time_per_run * (total_runs - global_run)
                    progress_pct = (global_run / total_runs) * 100
                    print(
                        f"[Progress] {global_run}/{total_runs} ({progress_pct:.1f}%) | "
                        f"Elapsed: {_format_duration(elapsed)} | "
                        f"ETA: {_format_duration(eta_seconds)}"
                    )

    ############################################################################################################


    ### RETURN AGGREGATE RECALL PLOTS ##########################################################################
    
    # aggregate_recall_plots(
    #     datasets=datasets,
    #     out_dir=out_dir,
    # )
    
    #     # Figure 4
    # recall_plot(
    #     data_dir=out_dir,
    #     out_dir=Path(r'C:\Users\timov\Desktop\Utrecht\Utrecht\MSBBSS\thesis_timo\Report\results\aggregate_recall_plots'),
    #     datasets=datasets,
    #     conditions=None,
    #     x_max=100,
    #     show_diagonal=False,
    #     x_scale='count',
    #     show_average=True,
    # )
    
    ############################################################################################################

    return

if __name__ == "__main__":
    app()