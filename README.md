# Entity Resolution with Small-Scale LLMs: A Study on Prompting Strategies and Hardware Limitations

A study on using 7B parameter LLMs with 4-bit quantization for entity matching tasks. This work evaluates different prompting strategies while considering hardware limitations.

## Overview

We explore:
* Zero-shot vs few-shot prompting
* Message vs system format
* General matching definitions  
* Position bias
* Model performance under resource constraints

## Models Evaluated

* Orca2
* OpenHermes 2 
* Zephyr
* Mistral-OpenOrca
* Stable-Beluga
* Llama-Pro

## Key Results (F1 Score)

| Model | Abt-Buy | Walmart-Amazon |
|-------|----------|----------------|
| Orca2 | 0.804 | 0.538 |
| OpenHermes | 0.791 | 0.515 |
| Zephyr | 0.789 | 0.531 |

## Implementation Details

* Uses [Ollama](https://github.com/ollama/ollama) for local deployment
* 7B parameter models
* 4-bit quantization
* NVIDIA GeForce GTX 1080 Ti (11GB)

## Key Findings

* Few-shot prompting outperforms zero-shot approaches
* Intersection method yields best results in few-shot prompting
* Example order significantly impacts model performance
* Message format shows advantages over system format
* Orca2 consistently leads across different scenarios
