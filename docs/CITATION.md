# Citing this work and the work we build on

## How to cite this framework

If you use the local-ai-bench validation framework in academic
work, please cite it as:

```bibtex
@software{vlm_llm_benchmark_2026,
  title  = {local-ai-bench: A reproducible RAG / VLM / LLM
            validation framework},
  author = {qiurui144 and contributors},
  year   = {2026},
  url    = {https://github.com/qiurui144/local-ai-bench},
  note   = {Apache-2.0}
}
```

## Foundational references the framework builds on

The implementations in `benchmark/rigor/` and `benchmark/rag/` are
applications of well-established methods. When using a specific
module in a paper, cite the original method authors as well.

### Statistical methods

```bibtex
@book{cohen1988statistical,
  title     = {Statistical Power Analysis for the Behavioral Sciences},
  author    = {Cohen, Jacob},
  year      = {1988},
  publisher = {Lawrence Erlbaum},
  edition   = {2nd}
}

@article{wilcoxon1945individual,
  title   = {Individual Comparisons by Ranking Methods},
  author  = {Wilcoxon, Frank},
  journal = {Biometrics Bulletin},
  volume  = {1},
  number  = {6},
  pages   = {80--83},
  year    = {1945}
}

@article{efron1979bootstrap,
  title   = {Bootstrap Methods: Another Look at the Jackknife},
  author  = {Efron, Bradley},
  journal = {Annals of Statistics},
  volume  = {7},
  number  = {1},
  pages   = {1--26},
  year    = {1979}
}

@article{demsar2006statistical,
  title   = {Statistical Comparisons of Classifiers over Multiple Data Sets},
  author  = {Dem{\v{s}}ar, Janez},
  journal = {Journal of Machine Learning Research},
  volume  = {7},
  pages   = {1--30},
  year    = {2006}
}
```

### Calibration

```bibtex
@inproceedings{naeini2015obtaining,
  title     = {Obtaining Well Calibrated Probabilities Using Bayesian
               Binning},
  author    = {Naeini, Mahdi Pakdaman and Cooper, Gregory and
               Hauskrecht, Milos},
  booktitle = {AAAI},
  year      = {2015}
}

@inproceedings{guo2017calibration,
  title     = {On Calibration of Modern Neural Networks},
  author    = {Guo, Chuan and Pleiss, Geoff and Sun, Yu and
               Weinberger, Kilian Q.},
  booktitle = {ICML},
  year      = {2017}
}
```

### Inter-rater agreement

```bibtex
@article{cohen1960coefficient,
  title   = {A Coefficient of Agreement for Nominal Scales},
  author  = {Cohen, Jacob},
  journal = {Educational and Psychological Measurement},
  volume  = {20},
  number  = {1},
  pages   = {37--46},
  year    = {1960}
}

@article{fleiss1971measuring,
  title   = {Measuring Nominal Scale Agreement Among Many Raters},
  author  = {Fleiss, Joseph L.},
  journal = {Psychological Bulletin},
  volume  = {76},
  number  = {5},
  pages   = {378--382},
  year    = {1971}
}

@book{krippendorff2004content,
  title     = {Content Analysis: An Introduction to Its Methodology},
  author    = {Krippendorff, Klaus},
  edition   = {2nd},
  year      = {2004},
  publisher = {Sage Publications}
}
```

### Retrieval metrics

```bibtex
@inproceedings{buckley2004retrieval,
  title     = {Retrieval Evaluation with Incomplete Information},
  author    = {Buckley, Chris and Voorhees, Ellen M.},
  booktitle = {SIGIR},
  year      = {2004}
}

@inproceedings{chapelle2009expected,
  title     = {Expected Reciprocal Rank for Graded Relevance},
  author    = {Chapelle, Olivier and Metlzer, Donald and Zhang, Ya
               and Grinspan, Pierre},
  booktitle = {CIKM},
  year      = {2009}
}

@article{moffat2008rank,
  title   = {Rank-Biased Precision for Measurement of Retrieval
             Effectiveness},
  author  = {Moffat, Alistair and Zobel, Justin},
  journal = {ACM TOIS},
  volume  = {27},
  number  = {1},
  pages   = {1--27},
  year    = {2008}
}

@article{jarvelin2002cumulated,
  title   = {Cumulated Gain-Based Evaluation of IR Techniques},
  author  = {J{\"a}rvelin, Kalervo and Kek{\"a}l{\"a}inen, Jaana},
  journal = {ACM TOIS},
  volume  = {20},
  number  = {4},
  pages   = {422--446},
  year    = {2002}
}

@inproceedings{cormack2009reciprocal,
  title     = {Reciprocal Rank Fusion outperforms Condorcet and
               individual Rank Learning Methods},
  author    = {Cormack, Gordon V. and Clarke, Charles L. A. and
               B{\"u}ttcher, Stefan},
  booktitle = {SIGIR},
  year      = {2009}
}
```

### LLM-as-judge

```bibtex
@inproceedings{liu2023geval,
  title     = {G-Eval: NLG Evaluation Using GPT-4 with Better Human
               Alignment},
  author    = {Liu, Yang and Iter, Dan and Xu, Yichong and Wang,
               Shuohang and Xu, Ruochen and Zhu, Chenguang},
  booktitle = {EMNLP},
  year      = {2023}
}

@inproceedings{zheng2023judging,
  title     = {Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena},
  author    = {Zheng, Lianmin and Chiang, Wei-Lin and Sheng, Ying
               and Zhuang, Siyuan and Wu, Zhanghao and Zhuang,
               Yonghao and Lin, Zi and Li, Zhuohan and Li, Dacheng
               and Xing, Eric and others},
  booktitle = {NeurIPS},
  year      = {2023}
}

@inproceedings{es2023ragas,
  title     = {{RAGAs}: Automated Evaluation of Retrieval Augmented
               Generation},
  author    = {Es, Shahul and James, Jithin and Espinosa-Anke, Luis
               and Schockaert, Steven},
  booktitle = {EACL Demos},
  year      = {2023}
}
```

### Reproducibility

```bibtex
@article{pineau2021reproducibility,
  title   = {Improving Reproducibility in Machine Learning Research
             (A Report from the NeurIPS 2019 Reproducibility Program)},
  author  = {Pineau, Joelle and others},
  journal = {Journal of Machine Learning Research},
  volume  = {22},
  pages   = {1--20},
  year    = {2021}
}
```

### Drift and production monitoring

```bibtex
@inproceedings{rabanser2019failing,
  title     = {Failing Loudly: An Empirical Study of Methods for
               Detecting Dataset Shift},
  author    = {Rabanser, Stephan and G{\"u}nnemann, Stephan and
               Lipton, Zachary C.},
  booktitle = {NeurIPS},
  year      = {2019}
}

@inproceedings{lipton2018detecting,
  title     = {Detecting and Correcting for Label Shift with Black
               Box Predictors},
  author    = {Lipton, Zachary C. and Wang, Yu-Xiang and Smola,
               Alexander J.},
  booktitle = {ICML},
  year      = {2018}
}
```
