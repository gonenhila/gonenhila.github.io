---
layout: archive
title: "Publications"
permalink: /publications/
author_profile: true
---


Full list of publications can be found in <u><a href="https://scholar.google.com/citations?user=URThmtMAAAAJ&hl=en">my Google Scholar profile</a>.</u>


Demystifying prompts in language models via perplexity estimation
------
<u>Hila Gonen</u>, Srini Iyer, Terra Blevins, Noah A. Smith, Luke Zettlemoyer. arXiv, 2022.

<details>
<summary>Abstract</summary>

Language models can be prompted to perform a wide variety of zero- and few-shot learning problems. However, performance varies significantly with the choice of prompt, and we do not yet understand why this happens or how to pick the best prompts. In this work, we analyze the factors that contribute to this variance and establish a new empirical hypothesis: the performance of a prompt is coupled with the extent to which the model is familiar with the language it contains. Over a wide range of tasks, we show that the lower the perplexity of the prompt is, the better the prompt is able to perform the task. As a result, we devise a method for creating prompts: (1) automatically extend a small seed set of manually written prompts by paraphrasing using GPT3 and backtranslation and (2) choose the lowest perplexity prompts to get significant gains in performance.
  
</details>
  
[Paper](https://arxiv.org/pdf/2212.04037)


Toward Human Readable Prompt Tuning: Kubrick's The Shining is a good movie, and a good prompt too?
------
Weijia Shi, Xiaochuang Han, <u>Hila Gonen</u>, Ari Holtzman, Yulia Tsvetkov Luke Zettlemoyer. arXiv, 2022.

<details>
<summary>Abstract</summary>

Large language models can perform new tasks in a zero-shot fashion, given natural language prompts that specify the desired behavior. Such prompts are typically hand engineered, but can also be learned with gradient-based methods from labeled data. However, it is underexplored what factors make the prompts effective, especially when the prompts are natural language. In this paper, we investigate common attributes shared by effective prompts. We first propose a human readable prompt tuning method (FLUENT PROMPT) based on Langevin dynamics that incorporates a fluency constraint to find a diverse distribution of effective and fluent prompts. Our analysis reveals that effective prompts are topically related to the task domain and calibrate the prior probability of label words. Based on these findings, we also propose a method for generating prompts using only unlabeled data, outperforming strong baselines by an average of 7.0% accuracy across three tasks.
  
</details>
  
[Paper](https://arxiv.org/abs/2212.10539)


XLM-V: Overcoming the Vocabulary Bottleneck in Multilingual Masked Language Models
------
Davis Liang, <u>Hila Gonen</u>, Yuning Mao, Rui Hou, Naman Goyal, Marjan Ghazvininejad, Luke Zettlemoyer, Madian Khabsa. arXiv, 2023.
<details>
<summary>Abstract</summary>

Large multilingual language models typically rely on a single vocabulary shared across 100+ languages. As these models have increased in parameter count and depth, vocabulary size has remained largely unchanged. This vocabulary bottleneck limits the representational capabilities of multilingual models like XLM-R. In this paper, we introduce a new approach for scaling to very large multilingual vocabularies by de-emphasizing token sharing between languages with little lexical overlap and assigning vocabulary capacity to achieve sufficient coverage for each individual language. Tokenizations using our vocabulary are typically more semantically meaningful and shorter compared to XLM-R. Leveraging this improved vocabulary, we train XLM-V, a multilingual language model with a one million token vocabulary. XLM-V outperforms XLM-R on every task we tested on ranging from natural language inference (XNLI), question answering (MLQA, XQuAD, TyDiQA), and named entity recognition (WikiAnn) to low-resource tasks (Americas NLI, MasakhaNER).
  
</details>
  
[Paper](https://arxiv.org/abs/2301.10472)


Prompting Language Models for Linguistic Structure
------
Terra Blevins, <u>Hila Gonen</u>, Luke Zettlemoyer. arXiv, 2022.

<details>
<summary>Abstract</summary>


Although pretrained language models (PLMs) can be prompted to perform a wide range of language tasks, it remains an open question how much this ability comes from generalizable linguistic representations versus more surface-level lexical patterns. To test this, we present a structured prompting approach that can be used to prompt for linguistic structure prediction tasks, allowing us to perform zero- and few-shot sequence tagging with autoregressive PLMs. We evaluate this approach on part-of-speech tagging, named entity recognition, and sentence chunking and demonstrate strong few-shot performance in all cases. We also find that, though the surface forms of the tags provide some signal, structured prompting can retrieve linguistic structure even with arbitrary labels, indicating that PLMs contain this knowledge in a general manner robust to label choice.

</details>
  
[Paper](https://arxiv.org/abs/2211.07830)


Analyzing the Mono-and Cross-Lingual Pretraining Dynamics of Multilingual Language Models
------
Terra Blevins, <u>Hila Gonen</u>, Luke Zettlemoyer. EMNLP, 2022.

<details>
<summary>Abstract</summary>
  
The emergent cross-lingual transfer seen in multilingual pretrained models has sparked significant interest in studying their behavior. However, because these analyses have focused on fully trained multilingual models, little is known about the dynamics of the multilingual pretraining process. We investigate when these models acquire their in-language and cross-lingual abilities by probing checkpoints taken from throughout XLM-R pretraining, using a suite of linguistic tasks. Our analysis shows that the model achieves high in-language performance early on, with lower-level linguistic skills acquired before more complex ones. In contrast, when the model learns to transfer cross-lingually depends on the language pair. Interestingly, we also observe that, across many languages and tasks, the final, converged model checkpoint exhibits significant performance degradation and that no one checkpoint performs best on all languages. Taken together with our other findings, these insights highlight the complexity and interconnectedness of multilingual pretraining.

</details>
    
[Paper](https://arxiv.org/abs/2205.11758) [Checkpoints](https://nlp.cs.washington.edu/xlmr-across-time/)


McPhraSy: Multi context phrase similarity and clustering
------
Amir DN Cohen, <u>Hila Gonen</u>, Ori Shapira, Ran Levy, Yoav Goldberg. Findings of EMNLP, 2022.

<details>
<summary>Abstract</summary>

Phrase similarity is a key component of many NLP applications. Current phrase similarity methods focus on embedding the phrase itself and use the phrase context only during training of the pretrained model. To better leverage the information in the context, we propose McPhraSy (Multi-context Phrase Similarity), a novel algorithm for estimating the similarity of phrases based on multiple contexts. At inference time, McPhraSy represents each phrase by considering multiple contexts in which it appears and computes the similarity of two phrases by aggregating the pairwise similarities between the contexts of the phrases. Incorporating context during inference enables McPhraSy to outperform current state-of-theart models on two phrase similarity datasets by up to 13.3%. Finally, we also present a new downstream task that relies on phrase similarity – keyphrase clustering – and create a new benchmark for it in the product reviews domain. We show that McPhraSy surpasses all other baselines for this task.

</details>

[Paper](https://www.amazon.science/publications/mcphrasy-multi-context-phrase-similarity-and-clustering)


Analyzing Gender Representation in Multilingual Models
------
<u>Hila Gonen</u>, Shauli Ravfogel, Yoav Goldberg. <span style="color:blue">Best Paper</span> at the RepL4NLP workshop at ACL, 2022

<details>
<summary>Abstract</summary>
  
Multilingual language models were shown to allow for nontrivial transfer across scripts and languages. In this work, we study the structure of the internal representations that enable this transfer. We focus on the representation of gender distinctions as a practical case study, and examine the extent to which the gender concept is encoded in shared subspaces across different languages. Our analysis shows that gender representations consist of several prominent components that are shared across languages, alongside language-specific components. The existence of language-independent and language-specific components provides an explanation for an intriguing empirical observation we make: while gender classification transfers well across languages, interventions for gender removal, trained on a single language, do not transfer easily to others.

</details>
  
[Paper](https://arxiv.org/abs/2204.09168) [Code](https://github.com/gonenhila/multilingual_gender)


Identifying Helpful Sentences in Product Reviews
------
Iftah Gamzu, <u>Hila Gonen</u>, Gilad Kutiel, Ran Levy, Eugene Agichtein. NAACL 2021

<details>
<summary>Abstract</summary>
  
In recent years online shopping has gained momentum and became an important venue for customers wishing to save time and simplify their shopping process. A key advantage of shopping online is the ability to read what other customers are saying about products of interest. In this work, we aim to maintain this advantage in situations where extreme brevity is needed, for example, when shopping by voice. We suggest a novel task of extracting a single representative helpful sentence from a set of reviews for a given product. The selected sentence should meet two conditions: first, it should be helpful for a purchase decision and second, the opinion it expresses should be supported by multiple reviewers. This task is closely related to the task of Multi Document Summarization in the product reviews domain but differs in its objective and its level of conciseness. We collect a dataset in English of sentence helpfulness scores via crowd-sourcing and demonstrate its reliability despite the inherent subjectivity involved. Next, we describe a complete model that extracts representative helpful sentences with positive and negative sentiment towards the product and demonstrate that it outperforms several baselines.

</details>
  
[Paper](https://aclanthology.org/2021.naacl-main.55/)


Pick a Fight or Bite your Tongue: Investigation of Gender Differences in Idiomatic Language Usage
------
Ella Rabinovich, <u>Hila Gonen</u>, Suzanne Stevenson. COLING 2020

<details>
<summary>Abstract</summary>
  
A large body of research on gender-linked language has established foundations regarding cross-gender differences in lexical, emotional, and topical preferences, along with their sociological underpinnings. We compile a novel, large and diverse corpus of spontaneous linguistic productions annotated with speakers' gender, and perform a first large-scale empirical study of distinctions in the usage of \textit{figurative language} between male and female authors. Our analyses suggest that (1) idiomatic choices reflect gender-specific lexical and semantic preferences in general language, (2) men's and women's idiomatic usages express higher emotion than their literal language, with detectable, albeit more subtle, differences between male and female authors along the dimension of dominance compared to similar distinctions in their literal utterances, and (3) contextual analysis of idiomatic expressions reveals considerable differences, reflecting subtle divergences in usage environments, shaped by cross-gender communication styles and semantic biases.

</details>
  
[Paper](https://aclanthology.org/2020.coling-main.454/) [Code](https://github.com/ellarabi/gender-idiomatic-language)


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


It's All in the Name: Mitigating Gender Bias with Name-Based Counterfactual Data Substitution
------
Rowan Hall Maudslay, <u>Hila Gonen</u>, Ryan Cotterell, Simone Teufel. EMNLP 2019

  
<details>
<summary>Abstract</summary>
  
This paper treats gender bias latent in word embeddings. Previous mitigation attempts rely on the operationalisation of gender bias as a projection over a linear subspace. An alternative approach is Counterfactual Data Augmentation (CDA), in which a corpus is duplicated and augmented to remove bias, e.g. by swapping all inherently-gendered words in the copy. We perform an empirical comparison of these approaches on the English Gigaword and Wikipedia, and find that whilst both successfully reduce direct bias and perform well in tasks which quantify embedding quality, CDA variants outperform projection-based methods at the task of drawing non-biased gender analogies by an average of 19% across both corpora. We propose two improvements to CDA: Counterfactual Data Substitution (CDS), a variant of CDA in which potentially biased text is randomly substituted to avoid duplication, and the Names Intervention, a novel name-pairing technique that vastly increases the number of words being treated. CDA/S with the Names Intervention is the only approach which is able to mitigate indirect gender bias: following debiasing, previously biased words are significantly less clustered according to gender (cluster purity is reduced by 49%), thus improving on the state-of-the-art for bias mitigation.

</details>
  
[Paper](https://aclanthology.org/D19-1530/) 


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


Inherent Vacuity in Lattice Automata?
------
<u>Hila Gonen</u>, Orna Kupferman. In Fields of Logic and Computation II, volume 9300 of Lecture Notes in Computer Science, pages 174-192. Springer, 2015

<details>
<summary>Abstract</summary>
  
Vacuity checking is traditionally performed after model checking has terminated successfully. It ensures that all the elements of the specification have played a role in its satisfaction by the system. The need to check the quality of specifications is even more acute in property-based design, where the specification is the only input, serving as a basis to the development of the system. Inherent vacuity adapts the theory of vacuity in model checking to the setting of property-based design. Essentially, a specification is inherently vacuous if it can be mutated into a simpler equivalent specification, which is known, in the case of specifications in linear temporal logic, to coincide with the fact the specification is satisfied vacuously in all systems.
A recent development in formal methods is an extension of the Boolean setting to a multi-valued one. In particular, instead of Boolean automata, which either accept or reject their input, there is a growing interest in weighted automata, which map an input word to a value from a semiring over a large domain. A distributive finite lattice is a special case of a semiring, and lattice automata are used in several methods for reasoning about multi-valued objects. We study inherent vacuity in the setting of lattice automata, namely the ability to mutate the value of a transition in the automaton without changing its language. We define the concept of inherent vacuity in lattice automata, study the complexity of deciding different types of vacuity, and relate the setting to the one known for linear temporal logics.

</details>

[Paper](https://www.cs.huji.ac.il/~ornak/publications/yuri15.pdf)


{% include base_path %}


