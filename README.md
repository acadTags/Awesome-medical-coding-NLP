# Awesome-medical-coding-NLP
 Automated medical coding is an area in Clinical Natural Language Processing to assign diagnosis or procedure medical codes to free-text clinical notes. The domain is a sub-field of document classification and information extraction.

 Below is a curation of papers (mostly peer-reviewed) and datasets in this field, mainly since the application of deep learning to this field (around 2017). Given the many new papers and datasets published, I may have lost some of them.
 
 Stop at any time to check this collection of papers!

# Datasets
-[MIMIC-III](https://physionet.org/content/mimiciii/1.4/)

-[CodieEsp](https://temu.bsc.es/codiesp/)

# Reviews
-[A systematic literature review of automated clinical coding and classification systems](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3000748/) - one of the earliest reviews in automated clinical coding, in JAMIA, 2010

-[A Systematic Literature Review of Automated ICD Coding and Classification Systems using Discharge Summaries](https://arxiv.org/abs/2107.10652) - review of work between 2010-2020

-[A Unified Review of Deep Learning for Automated Medical Coding](https://arxiv.org/abs/2201.02797) - a focus on deep learning, 2022.

-[A review on deep neural networks for ICD coding](https://www.computer.org/csdl/journal/tk/5555/01/09705116/1AII1Yh8t44) - TKDE 2022.

-[Computer-assisted clinical coding: A narrative review of the literature on its benefits, limitations, implementation and impact on clinical coding professionals](https://journals.sagepub.com/doi/10.1177/1833358319851305) - an application-oriented review of computer-assisted clinical coding, in Health Information Management Journal, 2020.

# Paper by years

## 2021

-[Multitask Recalibrated Aggregation Network for Medical Code Prediction](https://2021.ecmlpkdd.org/wp-content/uploads/2021/07/sub_161.pdf) - multi-task learning - in ECML-PKDD 2021

-[Effective Convolutional Attention Network for Multi-label Clinical Document Classification](https://aclanthology.org/2021.emnlp-main.481/) - in EMNLP 2021

-[Meta-LMTC: Meta-Learning for Large-Scale Multi-Label Text Classification](https://aclanthology.org/2021.emnlp-main.679/) - Meta-learning for few- or zero-shot multi-label classification - in EMNLP 2021

-[CoPHE: A Count-Preserving Hierarchical Evaluation Metric in Large-Scale Multi-Label Text Classification](http://arxiv.org/abs/2109.04853) A novel metric for hierarchical multi-label classification, applied to MIMIC-III ICD coding - in EMNLP 2021

-[Description-based Label Attention Classifier for Explainable ICD-9 Classification](https://aclanthology.org/2021.wnut-1.8.pdf) - with Longformer and label descriptions - in W-NUT@EMNLP 2021

-[Read, Attend, and Code: Pushing the Limits of Medical Codes Prediction from Clinical Notes by Machines](https://arxiv.org/abs/2107.10650) - Attention-based model, human-level coding results - in MLHC 2021 - [leaderboard on paper with code](https://paperswithcode.com/sota/medical-code-prediction-on-mimic-iii) - [video](https://www.youtube.com/watch?v=Pm5DZhCOJJ0)

-[Does the Magic of BERT Apply to Medical Code Assignment? A Quantitative Study](https://www.researchgate.net/publication/350005526_Does_the_Magic_of_BERT_Apply_to_Medical_Code_Assignment_A_Quantitative_Study) Evaluation of BERT on MIMIC-III ICD coding. in Computers in Biology and Medicine, 2021

-[Counterfactual Supporting Facts Extraction for Explainable Medical Record Based Diagnosis with Graph Network](https://www.aclweb.org/anthology/2021.naacl-main.156) - in NAACL 2021

-[Automatic ICD Coding via Interactive Shared Representation Networks with Self-distillation Mechanism](https://aclanthology.org/2021.acl-long.463/) in ACL 2021

-[JLAN: medical code prediction via joint learning attention networks and denoising mechanism](https://doi.org/10.1186/s12859-021-04520-x) in BMC Bioinformatics, 2021

-[Explainable Automated Coding of Clinical Notes using Hierarchical Label-Wise Attention Networks and Label Embedding Initialisation](https://arxiv.org/abs/2010.15728) in JBI, 2021.

## 2020

-[Multi-label Few/Zero-shot Learning with Knowledge Aggregated from Multiple Label Graphs](https://aclanthology.org/2020.emnlp-main.235/). In EMNLP 2020

-[An Empirical Study on Large-Scale Multi-Label Text Classification Including Few and Zero-Shot Labels](https://arxiv.org/pdf/2010.01653.pdf) - (i) Improvement on zero-shot learning and (ii) the idea of Graph-aware Annotation Proximity (GAP), an graph-based look into the coding process, and (iii) BERTs' underpreformance on MIMIC-III. In EMNLP 2020

-[BERT-XML: Large Scale Automated ICD Coding Using BERT Pretraining](https://aclanthology.org/2020.clinicalnlp-1.3/) - in ClinicalNLP workshop at EMNLP 2020

-[A Label Attention Model for ICD Coding from Clinical Text](https://www.ijcai.org/proceedings/2020/461) - In IJCAI 2020.

-[Generalized Zero-Shot Text Classification for ICD Coding](https://www.ijcai.org/Proceedings/2020/0556.pdf) -  Generalised Zero-shot learning with Generative adversial training, the ICD hierarchy with descriptions, and Graph Recurrent Neural Networks. In IJCAI 2020.

-[Towards Interpretable Clinical Diagnosis with Bayesian Network Ensembles Stacked on Entity-Aware CNNs](https://www.aclweb.org/anthology/2020.acl-main.286/) in ACL 2020.

-[HyperCore: Hyperbolic and Co-graph Representation for Automatic ICD Coding](https://www.aclweb.org/anthology/2020.acl-main.282/) - Hyperbolic embedding + Graph Convolutional Networks. In ACL 2020.

-[Clinical-Coder: Assigning Interpretable ICD-10 Codes to Chinese Clinical Notes](https://www.aclweb.org/anthology/2020.acl-demos.33/) in Demo at ACL 2020.

-[Experimental Evaluation and Development of a Silver-Standard for the MIMIC-III Clinical Coding Dataset](https://www.aclweb.org/anthology/2020.bionlp-1.8/) in BioNLP at ACL 2020.

## 2019

-[Multimodal Machine Learning for Automated ICD Coding](http://proceedings.mlr.press/v106/xu19a.html) Ensembling models from unstructured text, semi-structured text and structured tabular data for ICD coding. (Keyang Xu, Mike Lam, Jingzhi Pang, Xin Gao, Charlotte Band, Piyush Mathur, Frank Papay, Ashish K. Khanna, Jacek B. Cywinski, Kamal Maheshwari, Pengtao Xie, Eric P. Xing ; Proceedings of the 4th Machine Learning for Healthcare Conference, PMLR 106:197-215, 2019.)

-[Ontological attention ensembles for capturing semantic concepts in ICD
code prediction from clinical text](https://www.aclweb.org/anthology/D19-6220/) - Multi-view convolution + multi-task learning. In LOUHI 2019 at EMNLP.

## 2018

-[Few-Shot and Zero-Shot Multi-Label Learning for Structured Label Spaces](https://www.aclweb.org/anthology/D18-1352/) - Few-shot and zero-shot learning with Graph Convolutional Neural Networks and the ICD hierarchy with descriptions. In EMNLP 2018.

-[Explainable Prediction of Medical Codes from Clinical Text](https://www.aclweb.org/anthology/N18-1100) - CNN with labelwise attention and the benchmark MIMIC preprocessed datasets. In NAACL-HLT 2018.

-[Towards automated clinical coding](https://discovery.ucl.ac.uk/id/eprint/10061782/) - International Journal of Medical Informatics, 2018

## 2017

-[Towards Automated ICD Coding Using Deep Learning](https://arxiv.org/abs/1711.04075)

-[Automatic Diagnosis Coding of Radiology Reports: A Comparison of Deep Learning and Conventional Classification Methods](https://aclanthology.org/W17-2342/) in BioNLP 2017
