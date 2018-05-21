## SCRN Language Models
Code for the Structurally Constrained Recurrent Network models from the paper Reproducing and Regularizing the SCRN Model (COLING 2018) 

### Requirements
Code is written in Python 3 and requires TensorFlow 1.1+. It also requires the following Python modules: `numpy`, `argparse`. You can install them via:
```
pip3 install numpy argparse
```

### Data
Data should be put into the `data/` directory, split into `train.txt`, `valid.txt`, and `test.txt`. Each line of the .txt file should be a sentence. The English Penn Treebank (PTB) data is given as the default.

The non-English data (Czech, French, German, Russian, and Spanish) can be downloaded from [Jan Botha's website](https://bothameister.github.io). For ease of use you can use the [Yoon Kim's script](https://github.com/yoonkim/lstm-char-cnn/blob/master/get_data.sh), which downloads these data and saves them into the relevant folders.

#### Note on non-English data
The PTB data above does not have end-of-sentence tokens for each sentence, and by default these are
appended. The non-English data already have end-of-sentence tokens for each line so, you want to add
`--eos " "` to the command line. 

### Model
To reproduce the baseline SCRN result on English PTB for `hidden_size=40` and `context_size=10` from Table 2
```
python3 SCRN-Word-baseline.py
```
Hyperparameters can be changed in the `PTBSmallConfig` class at the beginning of the script.

To reproduce the result of the small SCRN + ND + WT on English PTB from Table 4 use
```
python3 SCRN-Word.py --remb 1
```
To reproduce the result of the medium SCRN + VD + WT on Wikitext-2 from Table 4 use
```
python3 VD-SCRN-Word.py --config WT2Medium --remb 1 --data_dir data/wikitext-2
```
To reproduce the SCRN + ND + WT on Czech data from Table 6 use
```
python3 SCRN-Word.py --config WT2Medium --remb 1 --data_dir data/cs --eos " "
```

### Other options
To see the full list of options run with the `-h` key, e.g.
```
python3 VD-SCRN-Word.py -h
```
