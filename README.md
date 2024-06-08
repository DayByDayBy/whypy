
# whypy 

## an iterative LLM-botherer

iteratively invoking Large Language Models (LLM) using Ollama via `langchain_community` package

### Scripts

#### 1. `_loop.py`

iteratively invoke an LLM with different prompts

#### 2. `explainyourself.py`

iteratively prompts an LLM to explain itself. saves the responses in csv and txt file formats.

### Requirements

- Python 3.x
- Packages: `argparse`, `pandas`, `langchain_community`, `langchain`, `langchain_core`

install the required packages using pip:

```
pip install argparse pandas langchain_community
```

### Usage

1. clone the repo or copy the script to your local machine.
2. run the script with the desired number of iterations:



```bash
python loop.py [iterations]
```

(if no iterations are given, the script defaults to 10)

output files will be saved in the `outputs/pitching` directory with a filename format that includes the model name, temperature, and a timestamp.




```bash
python explainyourself.py
```

output files will be saved in the `outputs/explain` directory with filenames that include the model name and a timestamp.


---
