---
layout: archive
title: "Publications"
permalink: /publications/
author_profile: true
---


Full list of publications can be found in <u><a href="https://scholar.google.com/citations?user=URThmtMAAAAJ&hl=en">my Google Scholar profile</a>.</u>

Analyzing Gender Representation in Multilingual Models
------
*Hila Gonen*, Shauli Ravfogel, Yoav Goldberg, RepL4NLP workshop at ACL, 2022

Multilingual language models were shown to allow for nontrivial transfer across scripts and languages. In this work, we study the structure of the internal representations that enable this transfer. We focus on the representation of gender distinctions as a practical case study, and examine the extent to which the gender concept is encoded in shared subspaces across different languages. Our analysis shows that gender representations consist of several prominent components that are shared across languages, alongside language-specific components. The existence of language-independent and language-specific components provides an explanation for an intriguing empirical observation we make: while gender classification transfers well across languages, interventions for gender removal, trained on a single language, do not transfer easily to others.

[Paper](https://arxiv.org/abs/2204.09168) [Code](https://github.com/gonenhila/multilingual_gender)


Identifying Helpful Sentences in Product Reviews
------
Iftah Gamzu, *Hila Gonen*, Gilad Kutiel, Ran Levy, Eugene Agichtein, NAACL 2021

In recent years online shopping has gained momentum and became an important venue for customers wishing to save time and simplify their shopping process. A key advantage of shopping online is the ability to read what other customers are saying about products of interest. In this work, we aim to maintain this advantage in situations where extreme brevity is needed, for example, when shopping by voice. We suggest a novel task of extracting a single representative helpful sentence from a set of reviews for a given product. The selected sentence should meet two conditions: first, it should be helpful for a purchase decision and second, the opinion it expresses should be supported by multiple reviewers. This task is closely related to the task of Multi Document Summarization in the product reviews domain but differs in its objective and its level of conciseness. We collect a dataset in English of sentence helpfulness scores via crowd-sourcing and demonstrate its reliability despite the inherent subjectivity involved. Next, we describe a complete model that extracts representative helpful sentences with positive and negative sentiment towards the product and demonstrate that it outperforms several baselines.

[Paper](https://aclanthology.org/2021.naacl-main.55/)


Pick a Fight or Bite your Tongue: Investigation of Gender Differences in Idiomatic Language Usage
------
Ella Rabinovich, *Hila Gonen*, Suzanne Stevenson, COLING 2020

A large body of research on gender-linked language has established foundations regarding cross-gender differences in lexical, emotional, and topical preferences, along with their sociological underpinnings. We compile a novel, large and diverse corpus of spontaneous linguistic productions annotated with speakers' gender, and perform a first large-scale empirical study of distinctions in the usage of \textit{figurative language} between male and female authors. Our analyses suggest that (1) idiomatic choices reflect gender-specific lexical and semantic preferences in general language, (2) men's and women's idiomatic usages express higher emotion than their literal language, with detectable, albeit more subtle, differences between male and female authors along the dimension of dominance compared to similar distinctions in their literal utterances, and (3) contextual analysis of idiomatic expressions reveals considerable differences, reflecting subtle divergences in usage environments, shaped by cross-gender communication styles and semantic biases.

[Paper](https://aclanthology.org/2020.coling-main.454/) [Code](https://github.com/ellarabi/gender-idiomatic-language)

{% include base_path %}


