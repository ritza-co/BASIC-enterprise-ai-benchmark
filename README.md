<p align="center">
  <a href="https://enterprisebot.ai/">
    <img alt="Enterprise Bot" title="Enterprise Bot" src="./logo.svg" width="400" style="color: black">
  </a>
</p>


<p align="center">
  <i>Conversational Automation for Enterprises</i><br/> 
  <a href="https://enterprisebot.ai">enterprisebot.ai</a>
</p>

<h1 align="center">
Benchmarking Enterprise AI
</h1>

<br/>



## Getting Started

Follow these instructions to set up the [BASIC benchmarking tool](https://www.enterprisebot.ai/blog/back-to-basics-a-generative-ai-benchmark-for-enterprise) on your local machine to evaluate [LLMs](https://en.wikipedia.org/wiki/Large_language_model) on key metrics like accuracy, contextual understanding, compliance, consistency, and performance.

### Installing the tool

Clone the repository to your local machine. 

Install the required libraries using the following command:

```bash
pip install -r requirements.txt
````

Create a `.env` file to store your API keys. Add the following lines to the `.env` file:

```bash
OPENAI_API_KEY=<your_openai_api_key>
ANTHROPIC_API_KEY=<your_anthropic_api_key>
GOOGLE_API_KEY=<your_google_api_key>
```

### Running the benchmark

Run the project using the following command:

```bash
python basic.py <model>
```

Replace `<model>` with the name of the model you want to evaluate. The available models are:

- claude-3-opus-20240229
- gpt-4-1106-preview
- gpt-3.5-turbo-0125
- gpt-4

To evaluate all available models, run the project using the following command:

```bash
python basic.py
```


### Running using custom datasets

You can run the benchmark using your own datasets by adding the dataset to the `dataset` folder. The dataset should 
be a `.csv` file, with each line containing a `question`, `answer`, and `context`, in that order. 

Run the benchmark with your dataset using the following command:

```bash
python basic.py <dataset_name>
```

To run the benchmark with a specific model and dataset:

```bash
python basic.py <model> <dataset_name>
```


### Adding new models

Add a new model to the `available_models` array in the `basic.py` file. The key should be the model name.

```python
available_models = ["claude-3-opus-20240229", "gpt-4-1106-preview", "gpt-3.5-turbo-0125", "gpt-4"]
```

You also need to add the model to the `calculateModelCost` function. The function should return the cost of the model based on the AI provider's pricing.

```python
def calculateModelCost(model, token_usage):
	if model == "gpt-4-0125-preview" or model == "gpt-4-1106-preview":
		cost = token_usage * 0.00003
	elif model == "gpt-4":
		cost = token_usage * 0.00006
	elif model == "gpt-3.5-turbo-0125":
		cost = token_usage * 0.0000015
	elif model == "claude-3-opus-20240229":
		cost = token_usage * 0.000075
	elif model == "<new_model>":
		cost = token_usage * <new_price>

```



