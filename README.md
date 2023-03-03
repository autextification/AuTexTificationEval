# AuTexTification Evaluation
This repo contains the script the organizers will use to evaluate the submissions.

As participant, you can use it as well to evaluate your models during the training phase of the competition.

## Getting started

Install the requirements
`pip install -r requirements.txt`

## Folder structure

The `task_submissions` folder contains two subfolders: `ground_truth` and `submissions`.

The `ground_truth` folder contains one folder per subtask, and each subtask, one folder per language. You should put the ground truth files (`truth.tsv`) in the inner folders.

The `submissions` folder contain as many folders as participants in the competition. Each participant folder has the same structure than the `ground_truth` folder (one folder per subtask, and each subtask, one folder per language). You should put all your runs (`{run_name}.tsv`) in the inner folders.

## Evaluation script

```
usage: evaluate_submissions.py [-h] [--submissions_path SUBMISSIONS_PATH] [--ground_truth_path GROUND_TRUTH_PATH] {subtask_1,subtask_2} {es,en}

positional arguments:
  {subtask_1,subtask_2}
                        Subtask to evaluate
  {es,en}               Language to evaluate

optional arguments:
  -h, --help            show this help message and exit
  --submissions_path SUBMISSIONS_PATH
                        Path to the submissions folder
  --ground_truth_path GROUND_TRUTH_PATH
                        Path to the ground_truth folder
 ```
 
For instance, you can use the evaluation script to evaluate the submissions on the spanish variant of the subtask 2 as:

```bash
python evaluate_submissions.py \
--submissions_path task_submissions/submissions/ \
--ground_truth_path task_submissions/ground_truth/ \
subtask_2 \
es
```

which will return a dataframe with four columns: *team* (team name), *run* (run name), *all_metrics* (metrics from sklearn.classification_report), *mf1* (macro-f1), and *mf1_cinterval* (confidence interval of macro-f1). If you run the evaluation script with the truths and preds of this repo, you will get something similar to:

```     
      team    run                                        all_metrics  mf1 mf1_cinterval
0  my_team  truth  {'A': {'precision': 1.0, 'recall': 1.0, 'f1-sc...  1.0    (1.0, 1.0)
```
