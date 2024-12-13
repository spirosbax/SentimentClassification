# Representing Sentences with Neural Models

## How to run the code

```bash
conda env create -f nlp_gpu.yml
conda activate nlp_gpu
```

Example training command:
```bash
python train.py --model TreeLSTM --word_embeddings glove --trainable_embeddings --supervise_nodes --batch_size 128 --patience 10 --max_epochs 100
```

Results will be saved in the `results` folder, in json format.

## Analysis

Our results can be found in the `results` folder.
To run the analysis, use the `analysis.ipynb` notebook, it generates the plots and tables in the report.

## Inspect

To inspect the code, use the `inspect.ipynb` notebook, it prints the source code of the important functions and classes.
