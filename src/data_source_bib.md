# Biograpies

  ```bibtex
    @inproceedings{min-etal-2023-factscore,
        title = "{FA}ct{S}core: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation",
        author = "Min, Sewon  and Krishna, Kalpesh  and Lyu, Xinxi  and Lewis, Mike  and Yih, Wen-tau  and Koh, Pang  and Iyyer, Mohit  and Zettlemoyer, Luke  and Hajishirzi, Hannaneh",
        booktitle = "Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing",
        month = dec,
        year = "2023",
        address = "Singapore",
        publisher = "Association for Computational Linguistics",
        url = "https://aclanthology.org/2023.emnlp-main.741/"
    }
  ```

# Summarization

```bibtex
@article{hermann2015teaching,
  title={Teaching machines to read and comprehend},
  author={Hermann, Karl Moritz and Kocisky, Tomas and Grefenstette, Edward and Espeholt, Lasse and Kay, Will and Suleyman, Mustafa and Blunsom, Phil},
  journal={Advances in neural information processing systems},
  volume={28},
  year={2015}
}
```


# Simplification

```bibtex
@inproceedings{fang-etal-2017-learning,
    title = "Learning how to Active Learn: A Deep Reinforcement Learning Approach",
    author = "Fang, Meng  and
      Li, Yuan  and
      Cohn, Trevor",
    editor = "Palmer, Martha  and
      Hwa, Rebecca  and
      Riedel, Sebastian",
    booktitle = "Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing",
    month = sep,
    year = "2017",
    address = "Copenhagen, Denmark",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/D17-1063/",
    doi = "10.18653/v1/D17-1063",
    pages = "595--605",
    abstract = "Active learning aims to select a small subset of data for annotation such that a classifier learned on the data is highly accurate. This is usually done using heuristic selection methods, however the effectiveness of such methods is limited and moreover, the performance of heuristics varies between datasets. To address these shortcomings, we introduce a novel formulation by reframing the active learning as a reinforcement learning problem and explicitly learning a data selection policy, where the policy takes the role of the active learning heuristic. Importantly, our method allows the selection policy learned using simulation to one language to be transferred to other languages. We demonstrate our method using cross-lingual named entity recognition, observing uniform improvements over traditional active learning algorithms."
}
```

# Rationalization (Binary)


  ```bibtex
  @article{zhang2023language,
    title={How language model hallucinations can snowball},
    author={Zhang, Muru and Press, Ofir and Merrill, William and Liu, Alisa and Smith, Noah A},
    journal={ICML},
    year={2024}
  }
  ```

# Scientific Attribution

  ```bibtex
@inproceedings{wadden-etal-2020-fact,
    title = "Fact or Fiction: Verifying Scientific Claims",
    author = "Wadden, David  and
      Lin, Shanchuan  and
      Lo, Kyle  and
      Wang, Lucy Lu  and
      van Zuylen, Madeleine  and
      Cohan, Arman  and
      Hajishirzi, Hannaneh",
    editor = "Webber, Bonnie  and
      Cohn, Trevor  and
      He, Yulan  and
      Liu, Yang",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)",
    month = nov,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2020.emnlp-main.609/",
    doi = "10.18653/v1/2020.emnlp-main.609",
    pages = "7534--7550",
    abstract = "We introduce scientific claim verification, a new task to select abstracts from the research literature containing evidence that SUPPORTS or REFUTES a given scientific claim, and to identify rationales justifying each decision. To study this task, we construct SciFact, a dataset of 1.4K expert-written scientific claims paired with evidence-containing abstracts annotated with labels and rationales. We develop baseline models for SciFact, and demonstrate that simple domain adaptation techniques substantially improve performance compared to models trained on Wikipedia or political news. We show that our system is able to verify claims related to COVID-19 by identifying evidence from the CORD-19 corpus. Our experiments indicate that SciFact will provide a challenging testbed for the development of new systems designed to retrieve and reason over corpora containing specialized domain knowledge. Data and code for this new task are publicly available at \url{https://github.com/allenai/scifact}. A leaderboard and COVID-19 fact-checking demo are available at \url{https://scifact.apps.allenai.org}."
}
  ```

  ```bibtex
@inproceedings{hossain-etal-2020-covidlies,
    title = "{COVIDL}ies: Detecting {COVID}-19 Misinformation on Social Media",
    author = "Hossain, Tamanna  and
      Logan IV, Robert L.  and
      Ugarte, Arjuna  and
      Matsubara, Yoshitomo  and
      Young, Sean  and
      Singh, Sameer",
    editor = "Verspoor, Karin  and
      Cohen, Kevin Bretonnel  and
      Conway, Michael  and
      de Bruijn, Berry  and
      Dredze, Mark  and
      Mihalcea, Rada  and
      Wallace, Byron",
    booktitle = "Proceedings of the 1st Workshop on {NLP} for {COVID}-19 (Part 2) at {EMNLP} 2020",
    month = dec,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2020.nlpcovid19-2.11",
    doi = "10.18653/v1/2020.nlpcovid19-2.11",
    abstract = "The ongoing pandemic has heightened the need for developing tools to flag COVID-19-related misinformation on the internet, specifically on social media such as Twitter. However, due to novel language and the rapid change of information, existing misinformation detection datasets are not effective for evaluating systems designed to detect misinformation on this topic. Misinformation detection can be divided into two sub-tasks: (i) retrieval of misconceptions relevant to posts being checked for veracity, and (ii) stance detection to identify whether the posts Agree, Disagree, or express No Stance towards the retrieved misconceptions. To facilitate research on this task, we release COVIDLies (\url{https://ucinlp.github.io/covid19} ), a dataset of 6761 expert-annotated tweets to evaluate the performance of misinformation detection systems on 86 different pieces of COVID-19 related misinformation. We evaluate existing NLP systems on this dataset, providing initial benchmarks and identifying key challenges for future models to improve upon.",
}
  ```

  ```bibtex
@inproceedings{lin-etal-2022-truthfulqa,
    title = "{T}ruthful{QA}: Measuring How Models Mimic Human Falsehoods",
    author = "Lin, Stephanie  and
      Hilton, Jacob  and
      Evans, Owain",
    editor = "Muresan, Smaranda  and
      Nakov, Preslav  and
      Villavicencio, Aline",
    booktitle = "Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = may,
    year = "2022",
    address = "Dublin, Ireland",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.acl-long.229/",
    doi = "10.18653/v1/2022.acl-long.229",
    pages = "3214--3252",
    abstract = "We propose a benchmark to measure whether a language model is truthful in generating answers to questions. The benchmark comprises 817 questions that span 38 categories, including health, law, finance and politics. We crafted questions that some humans would answer falsely due to a false belief or misconception. To perform well, models must avoid generating false answers learned from imitating human texts. We tested GPT-3, GPT-Neo/J, GPT-2 and a T5-based model. The best model was truthful on 58{\%} of questions, while human performance was 94{\%}. Models generated many false answers that mimic popular misconceptions and have the potential to deceive humans. The largest models were generally the least truthful. This contrasts with other NLP tasks, where performance improves with model size. However, this result is expected if false answers are learned from the training distribution. We suggest that scaling up models alone is less promising for improving truthfulness than fine-tuning using training objectives other than imitation of text from the web."
}
  ```
