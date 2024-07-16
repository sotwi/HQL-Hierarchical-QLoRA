# HQL: Hierarchical-QLoRA


## Data

Because of copyright constraints we cannot provide all the data used for training. Following the provided links redirects to the right sources.

* Obtain a copy of [AMR3.0 (LDC2020T02)](https://catalog.ldc.upenn.edu/LDC2020T02) and translate it with [NLLB](https://huggingface.co/facebook/nllb-200-3.3B.).
* Obtain a copy of [AMR2.0 4 Translations (LDC2020T07)](https://catalog.ldc.upenn.edu/LDC2020T07).
* Obtain a copy of [FLORES-200](https://huggingface.co/datasets/facebook/flores) and parse the English sentences with [AMR3-structbart-L](https://github.com/IBM/transition-amr-parser).
* Reformat and place the files with the data on the corresponding path within the `data` directory. Look at the corresponding `.py` file for each dataset for details on the required format.

## Monolingual Experiment

To run this experiments use the following commands:
```
python train.py -bm google/mt5-large -ln "language" -le 21 -lf language_files/ast.csv -sn ast_21epochs -tm qlora
python train.py -bm google/mt5-large -ln "language" -le 21 -lf language_files/deu.csv -sn deu_21epochs -tm qlora
python train.py -bm google/mt5-large -ln "language" -le 21 -lf language_files/eng.csv -sn eng_21epochs -tm qlora
python train.py -bm google/mt5-large -ln "language" -le 21 -lf language_files/fra.csv -sn fra_21epochs -tm qlora
python train.py -bm google/mt5-large -ln "language" -le 21 -lf language_files/hat.csv -sn hat_21epochs -tm qlora
python train.py -bm google/mt5-large -ln "language" -le 21 -lf language_files/ita.csv -sn ita_21epochs -tm qlora
python train.py -bm google/mt5-large -ln "language" -le 21 -lf language_files/lim.csv -sn lim_21epochs -tm qlora
python train.py -bm google/mt5-large -ln "language" -le 21 -lf language_files/ltz.csv -sn ltz_21epochs -tm qlora
python train.py -bm google/mt5-large -ln "language" -le 21 -lf language_files/nld.csv -sn nld_21epochs -tm qlora
python train.py -bm google/mt5-large -ln "language" -le 21 -lf language_files/scn.csv -sn scn_21epochs -tm qlora
python train.py -bm google/mt5-large -ln "language" -le 21 -lf language_files/spa.csv -sn spa_21epochs -tm qlora
python train.py -bm google/mt5-large -ln "language" -le 21 -lf language_files/tpi.csv -sn tpi_21epochs -tm qlora
```

## Multilingual Experiment

To run this experiment use the following command:
```
python train.py -bm google/mt5-large -ln "level0" -le 4 -lf language_files/12langs.csv -sn 12langs_4epochs -tm qlora
```

## DLHQL
To run this experiment use the following command:
```
python train.py -bm google/mt5-large -ln "level0", "level1", "level2", "language" -le 1 -lf language_files/12langs_dlhql.csv -sn 12langs_dlhql -tm qlora
```

## PTHQL
To run this experiment use the following command:
```
python train.py -bm google/mt5-large -ln "level0", "level1", "level2", "language" -le 1 -lf language_files/12langs_pthql.csv -sn 12langs_pthql -tm qlora
```

