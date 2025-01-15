# HALOGEN🔦: Fantastic LLM Hallucinations and Where to Find Them

This repository provides the code and data of [HALOGEN🔦: Fantastic LLM Hallucinations and Where to Find Them](#) by *Abhilasha Ravichander, *Shrusti Ghela, David Wadden, and Yejin Choi

[Website](https://halogen-hallucinations.github.io/) |  [Paper](https://arxiv.org/abs/2501.08292) | [HALoGEN prompts](https://github.com/AbhilashaRavichander/HALoGEN/tree/main/prompts) |  [LLM Hallucinations](https://github.com/AbhilashaRavichander/HALoGEN/tree/main/model_hallucinations) | [Decomposers and Verifiers](https://github.com/AbhilashaRavichander/HALoGEN/tree/main/verifiers) | [Scoring Functions](https://github.com/AbhilashaRavichander/HALoGEN/tree/main/scorers)(see the following codebase)

## Overview


Despite their impressive ability to generate high-quality and fluent text, generative large language models (LLMs) also produce hallucinations: fabricated statements that contain false information, or that deviate from provided context. Understanding how often these hallucinations occur and what causes them remains a fundamental challenge in developing trustworthy AI systems.

This repository contains resources related to 🔦HALoGEN, a diverse multi-domain benchmark to measure LLM hallucinations. 

:star::star:  If you use any of our data, verifiers, or evaluations, please consider citing our work :star::star:  :

```bibtex
@misc{ravichander2025halogenfantasticllmhallucinations,
      title={HALoGEN: Fantastic LLM Hallucinations and Where to Find Them}, 
      author={Abhilasha Ravichander and Shrusti Ghela and David Wadden and Yejin Choi},
      year={2025},
      eprint={2501.08292},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2501.08292}, 
}
```

  -> If you use the biographies evaluation, please also cite,


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


  -> If you use the rationalization (binary) evaluation, please also cite,
  

  ```bibtex
  @article{zhang2023language,
    title={How language model hallucinations can snowball},
    author={Zhang, Muru and Press, Ofir and Merrill, William and Liu, Alisa and Smith, Noah A},
    journal={ICML},
    year={2024}
  }
  ```

If you have any questions, please feel free to email us at **aravicha[AT]cs.washington.edu**, **shrustighela1[AT]gmail.com**.

## Announcements

**Jan 13, 2025:** Our evaluator for scientific references uses an API that is internal to Semantic Scholar. We will release this evaluator when this API becomes publicly available. In the meantime, we have implemented a evaluator that uses the public API which is available [here](https://github.com/AbhilashaRavichander/HALoGEN/blob/main/src/scientific_attribution.py), but we found that this evaluator identifies more references as hallucinated compared to our private implementation. We also provide the references that are flagged as hallucinations by the internal API [here](https://github.com/AbhilashaRavichander/HALoGEN/tree/main/model_hallucinations/references).


**Jan 13, 2025:** We use the TogetherAI implementation of LLama-2-70B for the verifiers. This has since been deprecated, so please run the model locally if you need to match the results in our draft. This repository uses the TogetherAI implementation of Llama-3-70B instead.


## Installation

Please run the following commands:

```
export PYTHONPATH="${PYTHONPATH}:./src:./verifiers"
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## HALoGEN Prompts

You can find the prompts for all categories [here](https://github.com/AbhilashaRavichander/HALoGEN/tree/main/prompts). 
* Prompts for each task are in the associated prompt_\<TASK NAME\> file, under the `"prompts"` columnn. 
* Additionally for the historical events task, we also include metadata such as the birth and death dates of the entities mentioned in the prompt.
* If you would prefer to use huggingface datasets to load halogen prompts, you can do 

## HALoGEN Responses

You can find the model responses for all categories [here](https://github.com/AbhilashaRavichander/HALoGEN/tree/main/responses). 
* Model responses for each task are in the associated task directory , under the `"response"` columnn. 

## LLM Hallucinations

You can find LLM hallucinations for all categories [here](https://github.com/AbhilashaRavichander/HALoGEN/tree/main/model_hallucinations).
* Model hallucination for each task are in the associated \<TASK NAME\> directory.
* For each model file in this directory, `"Atomic Units"` refers to each atomic fact that can be verified for factual precision in the model response, and `"Hallucinated Atomic Units"` refers to atomic units that are determined to be non-factual according to the corresponding verifier.

## Decomposers and Verifiers

You can find verifiers [here](https://github.com/AbhilashaRavichander/HALoGEN/tree/main/verifiers). 
* Most verifiers will require the path to a CSV file containing model responses, which should have a `'response'` column which contains the generations to be verified. 
* You can look at the model responses [here](https://github.com/AbhilashaRavichander/HALoGEN/tree/main/responses) for reference.
* In addition, some of the verifiers leverage either OpenAI models or LLama models for verification, and will require a valid OpenAI key, as well as either a TogetherAI key or a local Llama implementation to run. 
* Every verifier file has a minimal example describing how to run that verifier. You can look at an example of how to run all verifiers in `src/evaluate_hallucinations.py`.


To run any configuration of HALoGEN verifiers, please
* Specify OpenAI API key, Together API Key and Semantic Scholar API key in `config.yml`

and then run

`python src/evaluate_hallucinations.py --[TASK NAME]`


## Scoring

Once you have run your verifiers and obtained hallucination files, you are ready to compute metrics to measure the extent of LLM Hallucination. You can find all the scorers [here](https://github.com/AbhilashaRavichander/HALoGEN/tree/main/scorers). 

* We compute three metrics: the hallucination score of responses, the response ratio, and the utility score of a response. Please refer to the paper for details on these scorers.
* Each scorer requires specifying a model hallucination file, which should include a `response` column, as well as columns for `atomic units` and `hallucinated atomic units` which are obtained from verifier outputs. Please find examples of this formatting [here](https://github.com/AbhilashaRavichander/HALoGEN/tree/main/model_hallucinations)
* To reproduce all results in the draft, simply run `./run_refusal_scorers.sh` and `./run_response_scorers.sh`
