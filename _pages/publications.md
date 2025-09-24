---
layout: archive
title: "Publications"
permalink: /publications/
author_profile: true
---


This is a list of selected publications.

A full list of publications can be found in <a href="https://scholar.google.com/citations?user=URThmtMAAAAJ&hl=en">my Google Scholar profile</a>.



OMNIGUARD: An Efficient Approach for AI Safety Moderation Across Modalities and Languages
Authors
------
Sahil Verma, Keegan Hines, Jeff Bilmes, Charlotte Siska, Luke Zettlemoyer, <u> Hila Gonen</u>, Chandan Singh. EMNLP, 2025.

<details>
<summary>Abstract</summary>

The emerging capabilities of large language models (LLMs) have sparked concerns about their immediate potential for harmful misuse. The core approach to mitigate these concerns is the detection of harmful queries to the model. Current detection approaches are fallible, and are particularly susceptible to attacks that exploit mismatched generalization of model capabilities (e.g., prompts in low-resource languages or prompts provided in non-text modalities such as image and audio). To tackle this challenge, we propose OMNIGUARD, an approach for detecting harmful prompts across languages and modalities. Our approach (i) identifies internal representations of an LLM/MLLM that are aligned across languages or modalities and then (ii) uses them to build a language-agnostic or modality-agnostic classifier for detecting harmful prompts. OMNIGUARD improves harmful prompt classification accuracy by 11.57\% over the strongest baseline in a multilingual setting, by 20.44\% for image-based prompts, and sets a new SOTA for audio-based prompts. By repurposing embeddings computed during generation, OMNIGUARD is also very efficient (≈120× faster than the next fastest baseline).

</details>

[Paper](https://arxiv.org/abs/2505.23856)



Dementia Through Different Eyes: Explainable Modeling of Human and LLM Perceptions for Early Awareness
Authors
------
Lotem Peled-Cohen, Maya Zadok, Nitay Calderon, <u> Hila Gonen</u>, Roi Reichart. Findings of EMNLP, 2025.

<details>
<summary>Abstract</summary>

Cognitive decline often surfaces in language years before diagnosis. It is frequently non-experts, such as those closest to the patient, who first sense a change and raise concern. As LLMs become integrated into daily communication and used over prolonged periods, it may even be an LLM that notices something is off. But what exactly do they notice--and should be noticing--when making that judgment? This paper investigates how dementia is perceived through language by non-experts. We presented transcribed picture descriptions to non-expert humans and LLMs, asking them to intuitively judge whether each text was produced by someone healthy or with dementia. We introduce an explainable method that uses LLMs to extract high-level, expert-guided features representing these picture descriptions, and use logistic regression to model human and LLM perceptions and compare with clinical diagnoses. Our analysis reveals that human perception of dementia is inconsistent and relies on a narrow, and sometimes misleading, set of cues. LLMs, by contrast, draw on a richer, more nuanced feature set that aligns more closely with clinical patterns. Still, both groups show a tendency toward false negatives, frequently overlooking dementia cases. Through our interpretable framework and the insights it provides, we hope to help non-experts better recognize the linguistic signs that matter.

</details>

[Paper](https://arxiv.org/abs/2505.13418)


MAGNET: Improving the Multilingual Fairness of Language Models with Adaptive Gradient-Based Tokenization
------
Orevaoghene Ahia, Sachin Kumar, <u> Hila Gonen</u>, Valentin Hoffman, Tomasz Limisiewicz, Yulia Tsvetkov, Noah A. Smith. arXiv, 2024.

<details>
<summary>Abstract</summary>

In multilingual settings, non-Latin scripts and low-resource languages are usually disadvantaged in terms of language models' utility, efficiency, and cost. Specifically, previous studies have reported multiple modeling biases that the current tokenization algorithms introduce to non-Latin script languages, the main one being over-segmentation. In this work, we propose MAGNET; multilingual adaptive gradient-based tokenization to reduce over-segmentation via adaptive gradient-based subword tokenization. MAGNET learns to predict segment boundaries between byte tokens in a sequence via sub-modules within the model, which act as internal boundary predictors (tokenizers). Previous gradient-based tokenization methods aimed for uniform compression across sequences by integrating a single boundary predictor during training and optimizing it end-to-end through stochastic reparameterization alongside the next token prediction objective. However, this approach still results in over-segmentation for non-Latin script languages in multilingual settings. In contrast, MAGNET offers a customizable architecture where byte-level sequences are routed through language-script-specific predictors, each optimized for its respective language script. This modularity enforces equitable segmentation granularity across different language scripts compared to previous methods. Through extensive experiments, we demonstrate that in addition to reducing segmentation disparities, MAGNET also enables faster language modelling and improves downstream utility.

</details>
  
[Paper](https://arxiv.org/abs/2407.08818)


MYTE: Morphology-Driven Byte Encoding for Better and Fairer Multilingual Language Modeling
------
Tomasz Limisiewicz, Terra Blevins, <u> Hila Gonen</u>, Orevaoghene Ahia, Luke Zettlemoyer. ACL, 2024.

<details>
<summary>Abstract</summary>

A major consideration in multilingual language modeling is how to best represent languages with diverse vocabularies and scripts. Although contemporary text encoding methods cover most of the world's writing systems, they exhibit bias towards the high-resource languages of the Global West. As a result, texts of underrepresented languages tend to be segmented into long sequences of linguistically meaningless units. To address the disparities, we introduce a new paradigm that encodes the same information with segments of consistent size across diverse languages. Our encoding convention (MYTE) is based on morphemes, as their inventories are more balanced across languages than characters, which are used in previous methods. We show that MYTE produces shorter encodings for all 99 analyzed languages, with the most notable improvements for non-European languages and non-Latin scripts. This, in turn, improves multilingual LM performance and diminishes the perplexity gap throughout diverse languages.

</details>
  
[Paper](https://arxiv.org/abs/2403.10691)


Breaking the Curse of Multilinguality with Cross-lingual Expert Language Models
------
Terra Blevins, Tomasz Limisiewicz, Suchin Gururangan, Margaret Li, <u> Hila Gonen</u>, Noah A. Smith, Luke Zettlemoyer. arXiv, 2024.

<details>
<summary>Abstract</summary>

Despite their popularity in non-English NLP, multilingual language models often underperform monolingual ones due to inter-language competition for model parameters. We propose Cross-lingual Expert Language Models (X-ELM), which mitigate this competition by independently training language models on subsets of the multilingual corpus. This process specializes X-ELMs to different languages while remaining effective as a multilingual ensemble. Our experiments show that when given the same compute budget, X-ELM outperforms jointly trained multilingual models across all considered languages and that these gains transfer to downstream tasks. X-ELM provides additional benefits over performance improvements: new experts can be iteratively added, adapting X-ELM to new languages without catastrophic forgetting. Furthermore, training is asynchronous, reducing the hardware requirements for multilingual training and democratizing multilingual modeling.

</details>
  
[Paper](https://arxiv.org/abs/2401.10440)


BUFFET: Benchmarking Large Language Models for Few-shot Cross-lingual Transfer
------
Akari Asai, Sneha Kudugunta, Xinyan Velocity Yu, Terra Blevins, <u> Hila Gonen</u>, Machel Reid, Yulia Tsvetkov, Sebastian Ruder, Hannaneh Hajishirzi. NAACL, 2024.

<details>
<summary>Abstract</summary>

Despite remarkable advancements in few-shot generalization in natural language processing, most models are developed and evaluated primarily in English. To establish a rigorous and equitable evaluation framework for few-shot cross-lingual transfer, we introduce a new benchmark, called BUFFET, which unifies 15 diverse tasks across 54 languages in a sequence-to-sequence format and provides a fixed set of few-shot examples and instructions. Using BUFFET, we perform thorough evaluations of ten state-of-the-art multilingual large language models with different transfer methods, namely in-context learning and fine-tuning. Our findings reveal significant room for improvement in few-shot in-context cross-lingual transfer. Strong multilingual pre-trained or instruction-tuned models such as BLOOM or ChatGPT often lag behind much smaller mT5-base models given the same number of few-shot samples, particularly in low-resource languages. Our analysis suggests avenues for future research in few-shot cross-lingual transfer.

</details>
  
[Paper](https://aclanthology.org/2024.naacl-long.100.pdf)


Demystifying prompts in language models via perplexity estimation
------
<u>Hila Gonen</u>, Srini Iyer, Terra Blevins, Noah A. Smith, Luke Zettlemoyer. Findings of EMNLP, 2023.

<details>
<summary>Abstract</summary>

Language models can be prompted to perform a wide variety of zero- and few-shot learning problems. However, performance varies significantly with the choice of prompt, and we do not yet understand why this happens or how to pick the best prompts. In this work, we analyze the factors that contribute to this variance and establish a new empirical hypothesis: the performance of a prompt is coupled with the extent to which the model is familiar with the language it contains. Over a wide range of tasks, we show that the lower the perplexity of the prompt is, the better the prompt is able to perform the task. As a result, we devise a method for creating prompts: (1) automatically extend a small seed set of manually written prompts by paraphrasing using GPT3 and backtranslation and (2) choose the lowest perplexity prompts to get significant gains in performance.
  
</details>
  
[Paper](https://aclanthology.org/2023.findings-emnlp.679.pdf)


Do All Languages Cost the Same? Tokenization in the Era of Commercial Language Models
------
Orevaoghene Ahia, Sachin Kumar, <u>Hila Gonen</u>, Jungo Kasai, David R. Mortensen, Noah A. Smith, Yulia Tsvetkov. EMNLP, 2023.

<details>
<summary>Abstract</summary>

Language models have graduated from being research prototypes to commercialized products offered as web APIs, and recent works have highlighted the multilingual capabilities of these products. The API vendors charge their users based on usage, more specifically on the number of ``tokens'' processed or generated by the underlying language models. What constitutes a token, however, is training data and model dependent with a large variance in the number of tokens required to convey the same information in different languages. In this work, we analyze the effect of this non-uniformity on the fairness of an API's pricing policy across languages. We conduct a systematic analysis of the cost and utility of OpenAI's language model API on multilingual benchmarks in 22 typologically diverse languages. We show evidence that speakers of a large number of the supported languages are overcharged while obtaining poorer results. These speakers tend to also come from regions where the APIs are less affordable to begin with. Through these analyses, we aim to increase transparency around language model APIs' pricing policies and encourage the vendors to make them more equitable.

</details>
  
[Paper](https://aclanthology.org/2023.emnlp-main.614.pdf)


Universal NER: A Gold-Standard Multilingual Named Entity Recognition Benchmark
------
Stephen Mayhew, Terra Blevins, Shuheng Liu, Marek Šuppa, <u>Hila Gonen</u>, Joseph Marvin Imperial, Börje F Karlsson, Peiqin Lin, Nikola Ljubešić, LJ Miranda, Barbara Plank, Arij Riabi, Yuval Pinter. NAACL, 2024.

<details>
<summary>Abstract</summary>

We introduce Universal NER (UNER), an open, community-driven project to develop gold-standard NER benchmarks in many languages. The overarching goal of UNER is to provide high-quality, cross-lingually consistent annotations to facilitate and standardize multilingual NER research. UNER v1 contains 18 datasets annotated with named entities in a cross-lingual consistent schema across 12 diverse languages. In this paper, we detail the dataset creation and composition of UNER; we also provide initial modeling baselines on both in-language and cross-lingual learning settings. We release the data, code, and fitted models to the public.  
</details>
  
[Paper](https://arxiv.org/abs/2311.09122)


Toward Human Readable Prompt Tuning: Kubrick's The Shining is a good movie, and a good prompt too?
------
Weijia Shi, Xiaochuang Han, <u>Hila Gonen</u>, Ari Holtzman, Yulia Tsvetkov Luke Zettlemoyer. Findings of EMNLP, 2023.

<details>
<summary>Abstract</summary>

Large language models can perform new tasks in a zero-shot fashion, given natural language prompts that specify the desired behavior. Such prompts are typically hand engineered, but can also be learned with gradient-based methods from labeled data. However, it is underexplored what factors make the prompts effective, especially when the prompts are natural language. In this paper, we investigate common attributes shared by effective prompts. We first propose a human readable prompt tuning method (FLUENT PROMPT) based on Langevin dynamics that incorporates a fluency constraint to find a diverse distribution of effective and fluent prompts. Our analysis reveals that effective prompts are topically related to the task domain and calibrate the prior probability of label words. Based on these findings, we also propose a method for generating prompts using only unlabeled data, outperforming strong baselines by an average of 7.0% accuracy across three tasks.
  
</details>
  
[Paper](https://aclanthology.org/2023.findings-emnlp.733.pdf)


XLM-V: Overcoming the Vocabulary Bottleneck in Multilingual Masked Language Models
------
Davis Liang, <u>Hila Gonen</u>, Yuning Mao, Rui Hou, Naman Goyal, Marjan Ghazvininejad, Luke Zettlemoyer, Madian Khabsa. EMNLP, 2023.
<details>
<summary>Abstract</summary>

Large multilingual language models typically rely on a single vocabulary shared across 100+ languages. As these models have increased in parameter count and depth, vocabulary size has remained largely unchanged. This vocabulary bottleneck limits the representational capabilities of multilingual models like XLM-R. In this paper, we introduce a new approach for scaling to very large multilingual vocabularies by de-emphasizing token sharing between languages with little lexical overlap and assigning vocabulary capacity to achieve sufficient coverage for each individual language. Tokenizations using our vocabulary are typically more semantically meaningful and shorter compared to XLM-R. Leveraging this improved vocabulary, we train XLM-V, a multilingual language model with a one million token vocabulary. XLM-V outperforms XLM-R on every task we tested on ranging from natural language inference (XNLI), question answering (MLQA, XQuAD, TyDiQA), and named entity recognition (WikiAnn) to low-resource tasks (Americas NLI, MasakhaNER).
  
</details>
  
[Paper](https://aclanthology.org/2023.emnlp-main.813.pdf)


Prompting Language Models for Linguistic Structure
------
Terra Blevins, <u>Hila Gonen</u>, Luke Zettlemoyer. ACL, 2022.

<details>
<summary>Abstract</summary>


Although pretrained language models (PLMs) can be prompted to perform a wide range of language tasks, it remains an open question how much this ability comes from generalizable linguistic representations versus more surface-level lexical patterns. To test this, we present a structured prompting approach that can be used to prompt for linguistic structure prediction tasks, allowing us to perform zero- and few-shot sequence tagging with autoregressive PLMs. We evaluate this approach on part-of-speech tagging, named entity recognition, and sentence chunking and demonstrate strong few-shot performance in all cases. We also find that, though the surface forms of the tags provide some signal, structured prompting can retrieve linguistic structure even with arbitrary labels, indicating that PLMs contain this knowledge in a general manner robust to label choice.

</details>
  
[Paper](https://aclanthology.org/2023.acl-long.367.pdf)


Analyzing the Mono-and Cross-Lingual Pretraining Dynamics of Multilingual Language Models
------
Terra Blevins, <u>Hila Gonen</u>, Luke Zettlemoyer. EMNLP, 2022.

<details>
<summary>Abstract</summary>
  
The emergent cross-lingual transfer seen in multilingual pretrained models has sparked significant interest in studying their behavior. However, because these analyses have focused on fully trained multilingual models, little is known about the dynamics of the multilingual pretraining process. We investigate when these models acquire their in-language and cross-lingual abilities by probing checkpoints taken from throughout XLM-R pretraining, using a suite of linguistic tasks. Our analysis shows that the model achieves high in-language performance early on, with lower-level linguistic skills acquired before more complex ones. In contrast, when the model learns to transfer cross-lingually depends on the language pair. Interestingly, we also observe that, across many languages and tasks, the final, converged model checkpoint exhibits significant performance degradation and that no one checkpoint performs best on all languages. Taken together with our other findings, these insights highlight the complexity and interconnectedness of multilingual pretraining.

</details>
    
[Paper](https://aclanthology.org/2022.emnlp-main.234.pdf) [Checkpoints](https://nlp.cs.washington.edu/xlmr-across-time/)


Analyzing Gender Representation in Multilingual Models
------
<u>Hila Gonen</u>, Shauli Ravfogel, Yoav Goldberg. <span style="color:blue">Best Paper</span> at the RepL4NLP workshop at ACL, 2022

<details>
<summary>Abstract</summary>
  
Multilingual language models were shown to allow for nontrivial transfer across scripts and languages. In this work, we study the structure of the internal representations that enable this transfer. We focus on the representation of gender distinctions as a practical case study, and examine the extent to which the gender concept is encoded in shared subspaces across different languages. Our analysis shows that gender representations consist of several prominent components that are shared across languages, alongside language-specific components. The existence of language-independent and language-specific components provides an explanation for an intriguing empirical observation we make: while gender classification transfers well across languages, interventions for gender removal, trained on a single language, do not transfer easily to others.

</details>
  
[Paper](https://aclanthology.org/2022.repl4nlp-1.8.pdf) [Code](https://github.com/gonenhila/multilingual_gender)


It's not Greek to mBERT: Inducing Word-Level Translations from Multilingual BERT
------
<u>Hila Gonen</u>, Shauli Ravfogel, Yanai Elazar, Yoav Goldberg. BlackBoxNLP workshop, 2020

<details>
<summary>Abstract</summary>
  
Recent works have demonstrated that multilingual BERT (mBERT) learns rich cross-lingual representations, that allow for transfer across languages. We study the word-level translation information embedded in mBERT and present two simple methods that expose remarkable translation capabilities with no fine-tuning. The results suggest that most of this information is encoded in a non-linear way, while some of it can also be recovered with purely linear tools. As part of our analysis, we test the hypothesis that mBERT learns representations which contain both a language-encoding component and an abstract, cross-lingual component, and explicitly identify an empirical language-identity subspace within mBERT representations.

</details>
  
[Paper](https://aclanthology.org/2020.blackboxnlp-1.5/) [Code](https://github.com/gonenhila/mbert)


Automatically Identifying Gender Issues in Machine Translation using Perturbations
------
<u>Hila Gonen</u>, Kellie Webster. Findings of EMNLP, 2020

<details>
<summary>Abstract</summary>
   
The successful application of neural methods to machine translation has realized huge quality advances for the community. With these improvements, many have noted outstanding challenges, including the modeling and treatment of gendered language. While previous studies have identified issues using synthetic examples, we develop a novel technique to mine examples from real world data to explore challenges for deployed systems. We use our method to compile an evaluation benchmark spanning examples for four languages from three language families, which we publicly release to facilitate research. The examples in our benchmark expose where model representations are gendered, and the unintended consequences these gendered representations can have in downstream application.

</details>
  
[Paper](https://aclanthology.org/2020.findings-emnlp.180/) [Dataset](https://github.com/google-research-datasets/NatGenMT)


Simple, Interpretable and Stable Method for Detecting Words with Usage Change across Corpora
------
<u>Hila Gonen</u>*, Ganesh Jawahar*, Djamé Seddah, Yoav Goldberg (* equal contribution). ACL 2020

<details>
<summary>Abstract</summary>
  
The problem of comparing two bodies of text and searching for words that differ in their usage between them arises often in digital humanities and computational social science. This is commonly approached by training word embeddings on each corpus, aligning the vector spaces, and looking for words whose cosine distance in the aligned space is large. However, these methods often require extensive filtering of the vocabulary to perform well, and - as we show in this work - result in unstable, and hence less reliable, results. We propose an alternative approach that does not use vector space alignment, and instead considers the neighbors of each word. The method is simple, interpretable and stable. We demonstrate its effectiveness in 9 different setups, considering different corpus splitting criteria (age, gender and profession of tweet authors, time of tweet) and different languages (English, French and Hebrew).

</details>
  
[Paper](https://aclanthology.org/2020.acl-main.51/) [Code](https://github.com/gonenhila/usage_change)


Null It Out: Guarding Protected Attributes by Iterative Nullspace Projection
------

Shauli Ravfogel, Yanai Elazar, <u>Hila Gonen</u>, Michael Twiton, Yoav Goldberg. ACL 2020

<details>
<summary>Abstract</summary>
    
The ability to control for the kinds of information encoded in neural representation has a variety of use cases, especially in light of the challenge of interpreting these models. We present Iterative Null-space Projection (INLP), a novel method for removing information from neural representations. Our method is based on repeated training of linear classifiers that predict a certain property we aim to remove, followed by projection of the representations on their null-space. By doing so, the classifiers become oblivious to that target property, making it hard to linearly separate the data according to it. While applicable for multiple uses, we evaluate our method on bias and fairness use-cases, and show that our method is able to mitigate bias in word embeddings, as well as to increase fairness in a setting of multi-class classification.

</details>
  
[Paper](https://aclanthology.org/2020.acl-main.647/) [Code](https://github.com/shauli-ravfogel/nullspace_projection)


How does Grammatical Gender Affect Noun Representations in Gender-Marking Languages?
------
<u>Hila Gonen</u>, Yova Kementchedjhieva, Yoav Goldberg. <span style="color:blue">Best Paper</span> at CoNLL 2019

<details>
<summary>Abstract</summary>
  
Many natural languages assign grammatical gender also to inanimate nouns in the language. In such languages, words that relate to the gender-marked nouns are inflected to agree with the noun's gender. We show that this affects the word representations of inanimate nouns, resulting in nouns with the same gender being closer to each other than nouns with different gender. While "embedding debiasing" methods fail to remove the effect, we demonstrate that a careful application of methods that neutralize grammatical gender signals from the words' context when training word embeddings is effective in removing it. Fixing the grammatical gender bias yields a positive effect on the quality of the resulting word embeddings, both in monolingual and cross-lingual settings. We note that successfully removing gender signals, while achievable, is not trivial to do and that a language-specific morphological analyzer, together with careful usage of it, are essential for achieving good results.

</details>
  
[Paper](https://aclanthology.org/W19-3622/) [Code](https://github.com/gonenhila/grammatical_gender)


Lipstick on a Pig: Debiasing Methods Cover up Systematic Gender Biases in Word Embeddings But do not Remove Them
------
<u>Hila Gonen</u>, Yoav Goldberg. NAACL 2019

<details>
<summary>Abstract</summary>
  
Word embeddings are widely used in NLP for a vast range of tasks. It was shown that word embeddings derived from text corpora reflect gender biases in society. This phenomenon is pervasive and consistent across different word embedding models, causing serious concern. Several recent works tackle this problem, and propose methods for significantly reducing this gender bias in word embeddings, demonstrating convincing results. However, we argue that this removal is superficial. While the bias is indeed substantially reduced according to the provided bias definition, the actual effect is mostly hiding the bias, not removing it. The gender bias information is still reflected in the distances between "gender-neutralized" words in the debiased embeddings, and can be recovered from them. We present a series of experiments to support this claim, for two debiasing methods. We conclude that existing bias removal techniques are insufficient, and should not be trusted for providing gender-neutral modeling.

</details>
  
[Paper](https://aclanthology.org/N19-1061/) [Code](https://github.com/gonenhila/gender_bias_lipstick)


Language Modeling for Code-Switching: Evaluation, Integration of Monolingual Data, and Discriminative Training
------
<u>Hila Gonen</u>, Yoav Goldberg. EMNLP 2019

<details>
<summary>Abstract</summary>
  
We focus on the problem of language modeling for code-switched language, in the context of automatic speech recognition (ASR). Language modeling for code-switched language is challenging for (at least) three reasons: (1) lack of available large-scale code-switched data for training; (2) lack of a replicable evaluation setup that is ASR directed yet isolates language modeling performance from the other intricacies of the ASR system; and (3) the reliance on generative modeling. We tackle these three issues: we propose an ASR-motivated evaluation setup which is decoupled from an ASR system and the choice of vocabulary, and provide an evaluation dataset for English-Spanish code-switching. This setup lends itself to a discriminative training approach, which we demonstrate to work better than generative language modeling. Finally, we explore a variety of training protocols and verify the effectiveness of training with large amounts of monolingual data followed by fine-tuning with small amounts of code-switched data, for both the generative and discriminative cases.

</details>
  
[Paper](https://aclanthology.org/D19-1427/) [Code](https://github.com/gonenhila/codeswitching-lm)


Semi Supervised Preposition-Sense Disambiguation using Multilingual Data
------
<u>Hila Gonen</u>, Yoav Goldberg. COLING 2016

<details>
<summary>Abstract</summary>
  
Prepositions are very common and very ambiguous, and understanding their sense is critical for understanding the meaning of the sentence. Supervised corpora for the preposition-sense disambiguation task are small, suggesting a semi-supervised approach to the task. We show that signals from unannotated multilingual data can be used to improve supervised preposition-sense disambiguation. Our approach pre-trains an LSTM encoder for predicting the translation of a preposition, and then incorporates the pre-trained encoder as a component in a supervised classification system, and fine-tunes it for the task. The multilingual signals consistently improve results on two preposition-sense datasets.

</details>
[Paper](https://aclanthology.org/C16-1256/) 



{% include base_path %}


