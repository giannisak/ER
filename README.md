# Entity Resolution with Small-Scale LLMs: A Study on Prompting Strategies and Hardware Limitations

## Description
A study on using 7B parameter LLMs with 4-bit quantization for entity matching tasks. This research evaluates different prompting strategies while considering hardware limitations.

## Overview
We explore:
* Zero-shot vs few-shot prompting
* Message vs system format  
* General matching definitions
* Impact of example ordering
* Model performance under resource constraints

## Models Evaluated
* Orca2
* OpenHermes 2 
* Zephyr
* Mistral-OpenOrca
* Stable-Beluga
* Llama-Pro

## Key Results
| Model | Abt-Buy | Walmart-Amazon |
|-------|----------|----------------|
| Orca2 | 0.804 | 0.534 |
| OpenHermes | 0.771 | 0.515 |
| Zephyr | 0.726 | 0.496 |

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
