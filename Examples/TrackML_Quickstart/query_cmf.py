import pandas as pd
from cmflib import cmfquery
from tabulate import tabulate

def _print_executions_in_stage(cmf_query: cmfquery.CmfQuery, stage_name: str) -> None:
    print('\n')
    print('\n')
    df: pd.DataFrame = cmf_query.get_all_executions_in_stage(stage_name)
    #df.drop(columns=['Git_Start_Commit', 'Git_End_Commit'], inplace=True, axis=1)
    print(tabulate(df, headers='keys', tablefmt='psql'))


def query(mlmd_path: str) -> None:
    cmf_query = cmfquery.CmfQuery(mlmd_path)
    stages: t.List[str] = cmf_query.get_pipeline_stages("exatrkx")
    print(stages)

    for name in [
        '1. Train Metric Learning', 
        '2. Metric Learning Inference', 
        '3. Train GNN', 
        '4. GNN Inference', 
        '5. Build Track Candidates', 
        '6. Evaluate Track Candidates']:
        _print_executions_in_stage(cmf_query, name)

if __name__ == '__main__':
    query("mlmd")