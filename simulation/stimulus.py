import pandas as pd

### Select metadata criteria to use as stimulus for llm or as priors directly ###

def select_criteria(name: str, criterium: str, metadata: pd.ExcelFile):

    #create dictionary with the criteria depending on the selected criterium
    stimulus = {}
    
    # Check if dataset exists first
    indexed_metadata = metadata.set_index("dataset_ID")
    if name not in indexed_metadata.index:
        print(f"Warning: Dataset '{name}' not found in metadata.")
        return None
        
    if criterium not in indexed_metadata.columns:
        print(f"Warning: Criterium '{criterium}' not found in metadata columns.")
        return None

    stimulus[criterium] = indexed_metadata.loc[name, criterium]    
    
    return stimulus