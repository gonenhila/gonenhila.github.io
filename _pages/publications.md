---
layout: archive
title: "Publications"
permalink: /publications/
author_profile: true
---


Full list of publications can be found in <u><a href="https://scholar.google.com/citations?user=URThmtMAAAAJ&hl=en">my Google Scholar profile</a>.</u>

Analyzing Gender Representation in Multilingual Models
------
<u>Hila Gonen</u>, Shauli Ravfogel, Yoav Goldberg. RepL4NLP workshop at ACL, 2022

Multilingual language models were shown to allow for nontrivial transfer across scripts and languages. In this work, we study the structure of the internal representations that enable this transfer. We focus on the representation of gender distinctions as a practical case study, and examine the extent to which the gender concept is encoded in shared subspaces across different languages. Our analysis shows that gender representations consist of several prominent components that are shared across languages, alongside language-specific components. The existence of language-independent and language-specific components provides an explanation for an intriguing empirical observation we make: while gender classification transfers well across languages, interventions for gender removal, trained on a single language, do not transfer easily to others.

[Paper](https://arxiv.org/abs/2204.09168) [Code](https://github.com/gonenhila/multilingual_gender)


Identifying Helpful Sentences in Product Reviews
------
Iftah Gamzu, <u>Hila Gonen</u>, Gilad Kutiel, Ran Levy, Eugene Agichtein. NAACL 2021

In recent years online shopping has gained momentum and became an important venue for customers wishing to save time and simplify their shopping process. A key advantage of shopping online is the ability to read what other customers are saying about products of interest. In this work, we aim to maintain this advantage in situations where extreme brevity is needed, for example, when shopping by voice. We suggest a novel task of extracting a single representative helpful sentence from a set of reviews for a given product. The selected sentence should meet two conditions: first, it should be helpful for a purchase decision and second, the opinion it expresses should be supported by multiple reviewers. This task is closely related to the task of Multi Document Summarization in the product reviews domain but differs in its objective and its level of conciseness. We collect a dataset in English of sentence helpfulness scores via crowd-sourcing and demonstrate its reliability despite the inherent subjectivity involved. Next, we describe a complete model that extracts representative helpful sentences with positive and negative sentiment towards the product and demonstrate that it outperforms several baselines.

[Paper](https://aclanthology.org/2021.naacl-main.55/)


Pick a Fight or Bite your Tongue: Investigation of Gender Differences in Idiomatic Language Usage
------
Ella Rabinovich, <u>Hila Gonen</u>, Suzanne Stevenson. COLING 2020

A large body of research on gender-linked language has established foundations regarding cross-gender differences in lexical, emotional, and topical preferences, along with their sociological underpinnings. We compile a novel, large and diverse corpus of spontaneous linguistic productions annotated with speakers' gender, and perform a first large-scale empirical study of distinctions in the usage of \textit{figurative language} between male and female authors. Our analyses suggest that (1) idiomatic choices reflect gender-specific lexical and semantic preferences in general language, (2) men's and women's idiomatic usages express higher emotion than their literal language, with detectable, albeit more subtle, differences between male and female authors along the dimension of dominance compared to similar distinctions in their literal utterances, and (3) contextual analysis of idiomatic expressions reveals considerable differences, reflecting subtle divergences in usage environments, shaped by cross-gender communication styles and semantic biases.

[Paper](https://aclanthology.org/2020.coling-main.454/) [Code](https://github.com/ellarabi/gender-idiomatic-language)


It's not Greek to mBERT: Inducing Word-Level Translations from Multilingual BERT
------
<u>Hila Gonen</u>, Shauli Ravfogel, Yanai Elazar, Yoav Goldberg. BlackBoxNLP workshop, 2020

Recent works have demonstrated that multilingual BERT (mBERT) learns rich cross-lingual representations, that allow for transfer across languages. We study the word-level translation information embedded in mBERT and present two simple methods that expose remarkable translation capabilities with no fine-tuning. The results suggest that most of this information is encoded in a non-linear way, while some of it can also be recovered with purely linear tools. As part of our analysis, we test the hypothesis that mBERT learns representations which contain both a language-encoding component and an abstract, cross-lingual component, and explicitly identify an empirical language-identity subspace within mBERT representations.

[Paper](https://aclanthology.org/2020.blackboxnlp-1.5/) [Code](https://github.com/gonenhila/mbert)


Automatically Identifying Gender Issues in Machine Translation using Perturbations
------
<u>Hila Gonen</u>, Kellie Webster. Findings of EMNLP, 2020

The successful application of neural methods to machine translation has realized huge quality advances for the community. With these improvements, many have noted outstanding challenges, including the modeling and treatment of gendered language. While previous studies have identified issues using synthetic examples, we develop a novel technique to mine examples from real world data to explore challenges for deployed systems. We use our method to compile an evaluation benchmark spanning examples for four languages from three language families, which we publicly release to facilitate research. The examples in our benchmark expose where model representations are gendered, and the unintended consequences these gendered representations can have in downstream application.

[Paper](https://aclanthology.org/2020.findings-emnlp.180/) [Dataset](https://github.com/google-research-datasets/NatGenMT)


Simple, Interpretable and Stable Method for Detecting Words with Usage Change across Corpora
------
<u>Hila Gonen</u>*, Ganesh Jawahar*, Djamé Seddah, Yoav Goldberg (* equal contribution). ACL 2020

The problem of comparing two bodies of text and searching for words that differ in their usage between them arises often in digital humanities and computational social science. This is commonly approached by training word embeddings on each corpus, aligning the vector spaces, and looking for words whose cosine distance in the aligned space is large. However, these methods often require extensive filtering of the vocabulary to perform well, and - as we show in this work - result in unstable, and hence less reliable, results. We propose an alternative approach that does not use vector space alignment, and instead considers the neighbors of each word. The method is simple, interpretable and stable. We demonstrate its effectiveness in 9 different setups, considering different corpus splitting criteria (age, gender and profession of tweet authors, time of tweet) and different languages (English, French and Hebrew).

[Paper](https://aclanthology.org/2020.acl-main.51/) [Code](https://github.com/gonenhila/usage_change)


Null It Out: Guarding Protected Attributes by Iterative Nullspace Projection
------

Shauli Ravfogel, Yanai Elazar, <u>Hila Gonen</u>, Michael Twiton, Yoav Goldberg. ACL 2020

The ability to control for the kinds of information encoded in neural representation has a variety of use cases, especially in light of the challenge of interpreting these models. We present Iterative Null-space Projection (INLP), a novel method for removing information from neural representations. Our method is based on repeated training of linear classifiers that predict a certain property we aim to remove, followed by projection of the representations on their null-space. By doing so, the classifiers become oblivious to that target property, making it hard to linearly separate the data according to it. While applicable for multiple uses, we evaluate our method on bias and fairness use-cases, and show that our method is able to mitigate bias in word embeddings, as well as to increase fairness in a setting of multi-class classification.


[Paper](https://aclanthology.org/2020.acl-main.647/) [Code](https://github.com/shauli-ravfogel/nullspace_projection)


How does Grammatical Gender Affect Noun Representations in Gender-Marking Languages?
------
<u>Hila Gonen</u>, Yova Kementchedjhieva, Yoav Goldberg. CoNLL 2019, <span style="color:blue">Best Paper</span> 
Many natural languages assign grammatical gender also to inanimate nouns in the language. In such languages, words that relate to the gender-marked nouns are inflected to agree with the noun's gender. We show that this affects the word representations of inanimate nouns, resulting in nouns with the same gender being closer to each other than nouns with different gender. While "embedding debiasing" methods fail to remove the effect, we demonstrate that a careful application of methods that neutralize grammatical gender signals from the words' context when training word embeddings is effective in removing it. Fixing the grammatical gender bias yields a positive effect on the quality of the resulting word embeddings, both in monolingual and cross-lingual settings. We note that successfully removing gender signals, while achievable, is not trivial to do and that a language-specific morphological analyzer, together with careful usage of it, are essential for achieving good results.

[Paper](https://aclanthology.org/W19-3622/) [Code](https://github.com/gonenhila/grammatical_gender)


It's All in the Name: Mitigating Gender Bias with Name-Based Counterfactual Data Substitution
------
Rowan Hall Maudslay, <u>Hila Gonen</u>, Ryan Cotterell, Simone Teufel. EMNLP, 2019

This paper treats gender bias latent in word embeddings. Previous mitigation attempts rely on the operationalisation of gender bias as a projection over a linear subspace. An alternative approach is Counterfactual Data Augmentation (CDA), in which a corpus is duplicated and augmented to remove bias, e.g. by swapping all inherently-gendered words in the copy. We perform an empirical comparison of these approaches on the English Gigaword and Wikipedia, and find that whilst both successfully reduce direct bias and perform well in tasks which quantify embedding quality, CDA variants outperform projection-based methods at the task of drawing non-biased gender analogies by an average of 19% across both corpora. We propose two improvements to CDA: Counterfactual Data Substitution (CDS), a variant of CDA in which potentially biased text is randomly substituted to avoid duplication, and the Names Intervention, a novel name-pairing technique that vastly increases the number of words being treated. CDA/S with the Names Intervention is the only approach which is able to mitigate indirect gender bias: following debiasing, previously biased words are significantly less clustered according to gender (cluster purity is reduced by 49%), thus improving on the state-of-the-art for bias mitigation.

[Paper](https://aclanthology.org/D19-1530/) 


Lipstick on a Pig: Debiasing Methods Cover up Systematic Gender Biases in Word Embeddings But do not Remove Them
------
<u>Hila Gonen</u>, Yoav Goldberg. NAACL 2019

Word embeddings are widely used in NLP for a vast range of tasks. It was shown that word embeddings derived from text corpora reflect gender biases in society. This phenomenon is pervasive and consistent across different word embedding models, causing serious concern. Several recent works tackle this problem, and propose methods for significantly reducing this gender bias in word embeddings, demonstrating convincing results. However, we argue that this removal is superficial. While the bias is indeed substantially reduced according to the provided bias definition, the actual effect is mostly hiding the bias, not removing it. The gender bias information is still reflected in the distances between "gender-neutralized" words in the debiased embeddings, and can be recovered from them. We present a series of experiments to support this claim, for two debiasing methods. We conclude that existing bias removal techniques are insufficient, and should not be trusted for providing gender-neutral modeling.

[Paper](https://aclanthology.org/N19-1061/) [Code](https://github.com/gonenhila/gender_bias_lipstick)


[Paper]() [Code]()
[Paper]() [Code]()
{% include base_path %}


