# Natural language processing course 2023/24: `BDG`

## Slovenian Instruction-based Corpus Generation

- Žan Horvat, 63190120, zh0444@student.uni-lj.si
- Bine Markelj, 63190184, bm9928@student.uni-lj.si
- Anže Glušič, 63170101, ag5072@student.uni-lj.si

This file is continiously getting updated as we update our project.
<br/><br/>
### ALL PROJECT REPORTS
SUBMISSION1 REPORT: [Report1](https://github.com/UL-FRI-NLP-2023-2024/ul-fri-nlp-course-project-bdg/blob/main/report/report1.pdf)  
SUBMISSION2 REPORT: [Report2](https://github.com/UL-FRI-NLP-2023-2024/ul-fri-nlp-course-project-bdg/blob/main/report/report2.pdf)  
SUBMISSION2 REPORT: TODO 

<br/><br/>
### REPOSITORY STRUCTURE
Folder `./report` includes all the files needed for our report and all the reports themselves.  
Folder `./code` includes all our runnable code to reproduce our reported results.  
Folder `./model` is a place for our fine-tuned model to be saved.  
Folder `./datasets` includes all the original datasets we scraped.  
Folder `./corpora` includes all of our final corpora.  
Folder `./corpora_model_fine-tune` includes all of our final corpora, structured specifically for our model's fine-tuning.  
  
  <br/><br/>
### INSTUCTIONS TO RUN ALL THE CODE AND REPRODUCE OUR RESULTS
1. CRAWLING: TODO
2. DATA PROCESSING:  
   a.) Run preprocessing scripts on the crawled data in `./datasets` using 3 prepered scripts in `./code/preprocessing`  
   b.) This will create 3 corpora inside each `./corpora` folder and also 3 corpora suited specifically for our model's fine-tuning in   `./corpora_model_fine-tune` 
3. COMBINING CORPORA AND CORPUS EVALUATION  
   a.) Run `./code/evaluate_corpus.py` to combine seperate corpora into a 1 large final corpus  
   b.) Script will also return corpus statistics (number of conversations, number of tokens)  
   c.) Script also randomly samples 10 conversations  
   d.) Finally script converts corpora to a .jsonl file, suitable for LLM training  
4. MODEL FINE-TUNING  
   a.) Execute cells in `./code/LLM/mistral_model_fine-tuning.ipynb`  
   b.) This step takes large amount of computing resources (GPU) and time to complete  
   c.) Using same prompts test the difference between initial model and fine-tuned model  
