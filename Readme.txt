1) Put the files prescriptions.csv, diagnoses_icd.csv, labevents.csv from MIMIC-IV website in the folder Data
2) Run "mimiciv processing_24hr_morethan2visit_neurips.py" to get data file.
3) Run "Vocab creation mimiciv_neurips.py" on the data file obtaind from step 2
4) Run "Co-occurrence and interaction graph creation.py" to obtain adjacency files
5) Run "train_REFINE.py" to obtain results.



a) "train_BART.py" under the baselines folder trains BART