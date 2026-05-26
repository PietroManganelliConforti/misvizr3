



# Fork of the project:

# Is this chart lying to me? Automating the detection of misleading visualizations

[![License](https://img.shields.io/github/license/UKPLab/ukp-project-template)](https://opensource.org/licenses/Apache-2.0)
[![Python Versions](https://img.shields.io/badge/Python-3.10-blue.svg?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![HuggingFace Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-yellow)](https://huggingface.co/datasets/UKPLab/misviz)


This repository contains the datasets and code associated with the ACL 2026 Main conference paper: [Is this chart lying to me? Automating the detection of misleading visualizations](https://arxiv.org/abs/2508.21675). The Misviz and Misviz-synth datasets are released under a **CC-BY-SA 4.0** license. The code is released under an **Apache 2.0** license.

Contact person: [Jonathan Tonglet](mailto:jonathan.tonglet@tu-darmstadt.de) 

[UKP Lab](https://www.ukp.tu-darmstadt.de/) | [TU Darmstadt](https://www.tu-darmstadt.de/)

Don't hesitate to send us an e-mail or report an issue, if something is broken (and it shouldn't be) or if you have further questions. 


## Datasets

We briefly describe the datasets below. More information can be found in the [README](https://github.com/UKPLab/arxiv2025-misviz/tree/main/data) of the data folder.

### Misviz-synth

- *data/misviz_synth/misviz_synth.json* contains the task labels and metadata
- The visualizations, the underlying data tables, the code snippets, and the axis metadata can be downloaded from [TUdatalib](https://tudatalib.ulb.tu-darmstadt.de/handle/tudatalib/5003)

### Misviz 

- *data/misviz/misviz.json* contains the task labels and metadata
- The visualizations can be downloaded from the web using the following script. Please contact the authors if you face any issues downloading the images.

```python
python data/download_misviz_images.py --use_wayback 0
```

- The dataset can also be accessed on [HuggingFace](https://huggingface.co/datasets/UKPLab/misviz). However, please keep in mind that the experiment code is designed for the JSON version of the dataset available in this repo.

### Misviz instance example

<p align="center">
  <img width="70%" src="img/example.png" alt="Example instance of Misviz" />
</p>

```json
  {
      "image_path": "img/68718369730_misrepresentation.png",
      "image_url": "https://64.media.tumblr.com/88844d8c3be687e0549e7b7c0a403293/tumblr_mx1as48rLr1sgh0voo1_1280.jpg",
      "chart_type": [
          "bar chart",
          "pie chart"
      ],
      "misleader": [
          "misrepresentation"
      ],
      "wayback_image_url": "https://web.archive.org/web/20250619095605/https://64.media.tumblr.com/88844d8c3be687e0549e7b7c0a403293/tumblr_mx1as48rLr1sgh0voo1_1280.jpg",
      "split": "test",
      "bbox": []
  }
```


## Environment

Follow these instructions to recreate the environment used for our experiments.

```
$ conda create --name lying_charts python=3.10
$ conda activate lying_charts
$ pip install -r requirements.txt
```

## Experiments


### Evaluate zero-shot MLLMs

```python
python src/mllm_inference/misleader_detection_MLLM.py --datasets misviz_synth-misviz --split test --model internvl3/8B/ --max_tokens 200
```

The ```--model``` argument expects a string in the format ```model_name/model_size/```. By default, the following models are available:

| Name     | Available sizes | 🤗 models   |
| :---: | :---: | :---: |
| internvl3   |  8B, 38B, 78B | [Link](https://huggingface.co/collections/OpenGVLab/internvl3-67f7f690be79c2fe9d74fe9d) |
| qwen2.5-vl      | 7B, 32B, 72B   | [Link](https://huggingface.co/collections/Qwen/qwen25-vl-6795ffac22b334a837c0f9a5)  |

We also provide code to run experiments with GPT-4.1, GPT-o3, and Gemini-2.5-flash-lite using the OpenAI API and Google AI Studio. You will first need to obtain API keys from both providers and store them as environment variables.


### Rule-based linter

The rule-based linter can be evaluated both on ground truth and predicted axis metadata for Misviz-synth, but only on predicted axis metadata for Misviz. 

```python
python src/rule_based_linter/linter.py --datasets misviz_synth --split test --use_predicted_axis 0
```


## Disclaimer

> This repository contains experimental software and is published for the sole purpose of giving additional background details on the respective publication.
