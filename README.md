# MONETA

This repository contains Code and Data of MONETA: Multimodal Industry Classification through Geographic Information with Multi Agent Systems for ACL 2026.

MONETA is the first multimodal benchmark for industry classification. Code will be released soon.

## Abstract

Industry classification schemes are integral parts of public and corporate databases as they classify businesses based on economic activity. Due to the size of the company registers, manual annotation is costly, and fine-tuning models with every update in industry classification schemes requires significant data collection. We replicate the manual expert verification by using existing or easily retrievable multimodal resources for industry classification. We present MONETA, the first multimodal industry classification benchmark with text (Website, Wikipedia, Wikidata) and geospatial sources (OpenStreetMap and satellite imagery). Our dataset enlists 1,000 businesses in Europe with 20 economic activity labels according to EU guidelines (NACE). Our training-free baseline reaches 62.10% and 74.10% with open and closed-source Multimodal Large Language Models (MLLM). We observe an increase of up to 22.80% with the combination of multi-turn design, context enrichment, and classification explanations. We will release our dataset and the enhanced guidelines.

## Cite
```bibtex
@inproceedings{yuksel-etal-2026-moneta,
    title = "{MONETA}: Multimodal Industry Classification through Geographic Information with Multi Agent Systems",
    author = {Y{\"u}ksel, Arda  and
      Thiem, Gabriel  and
      Walter, Susanne  and
      Felka, Patrick  and
      Werb, Gabriela Alves  and
      Habernal, Ivan},
    editor = "Liakata, Maria  and
      Moreira, Viviane P.  and
      Zhang, Jiajun  and
      Jurgens, David",
    booktitle = "Proceedings of the 64th Annual Meeting of the {A}ssociation for {C}omputational {L}inguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2026",
    address = "San Diego, California, United States",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2026.acl-long.674/",
    pages = "14790--14814",
    ISBN = "979-8-89176-390-6",
    abstract = "Industry classification schemes are integral parts of public and corporate databases as they classify businesses based on economic activity. Due to the size of the company registers, manual annotation is costly, and fine-tuning models with every update in industry classification schemes requires significant data collection. We replicate the manual expert verification by using existing or easily retrievable multimodal resources for industry classification. We present MONETA, the first multimodal industry classification benchmark with text (Website, Wikipedia, Wikidata) and geospatial sources (OpenStreetMap and satellite imagery). Our dataset enlists 1,000 businesses in Europe with 20 economic activity labels according to EU guidelines (NACE). Our training-free baseline reaches 62.10{\%} and 74.10{\%} with open and closed-source Multimodal Large Language Models (MLLM). We observe an increase of up to 22.80{\%} with the combination of multi-turn design, context enrichment, and classification explanations. We will release our dataset and the enhanced guidelines."
}
```
