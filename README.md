# OffsetBias: Leveraging Debiased Data for Tuning Evaluators

Official implementation for paper **OffsetBias: Leveraging Debiased Data for Tuning Evaluators**. In the paper we present:
- **EvalBiasBench**, a meta-evaluation benchmark for testing judge models, 
- **OffsetBias Data**, a training dataset for pairwise preference evaluation,
- **OffsetBias Model**, a judge model trained using Offsetbias.

This repository contains sample code for running **Offsetbias Model** for evaluation, **EvalBiasBench** dataset, and an inference script for various evaluation models on various meta-evaluation benchmarks.

## Requirements

```sh
pip install -r requirements.txt
```

## Evaluation Inference with OffsetBias Model

OffsetBias Model works as a judge model that performs pairwise preference evaluation task, where *Instruction*, *Output (a)*, *Output (b)* are given, and a better output to the instruction needs to be found. You can use modules from this repository for simple and quick inference. Example code is in `offsetbias_inference.py`.
```python
from module import VllmModule

instruction = "explain like im 5"
output_a = "Scientists are studying special cells that could help treat a sickness called prostate cancer. They even tried these cells on mice and it worked!"
output_b = "Sure, I'd be happy to help explain something to you! What would you like me to explain?"

model_name = "NCSOFT/Llama-3-OffsetBias-8B"
module = VllmModule(prompt_name="offsetbias", model_name=model_name)

conversation = module.make_conversation(
  instruction=instruction,
  response1=output_a,
  response2=output_b,
  swap=False)

output = module.generate([conversation])
print(output[0])
# The model should output "Output (b)"
```

## Running EvalBiasBench

**EvalBiasBench**, ia benchmark for testing *judge models* robustness to evaluation scenarios containing biases. You can find the benchmark data under `data/evalbiasbench/`. The following shows instructions for running inference with various *judge models*, including *OffsetBias* model, on several benchmarks, including **EvalBiasBench**.

### Configuration

Prepare model configuration under `config/`. For OpenAI models, api key is required.

A configuration file, `offsetbias-8b.yaml`, looks like the following:
```yaml
prompt: llmbar # name of prompt file under prompt/

vllm_args:
  model_args: # args for vllm.LLM()
    model: NCSOFT/Llama-3-OffsetBias-8B
    dtype: float16
  sampling_params: # args for vllm.SamplingParams()
    temperature: 0
    max_tokens: 20

hf_args:
  model_args: # args for AutoModelForCausalLM.from_pretrained()
    model: NCSOFT/Llama-3-OffsetBias-8B
    dtype: float16
  generate_kwargs: # args for model.generate()
    max_new_tokens: 20
    pad_token_id: 128001
    do_sample: false
    temperature: 0

```


### Run Inference

Running inference will automatically create inference result file and score file under `result/`. Below are various possible commands.
```sh
# run offsetbias inference on BiasBench dataset
python run_bench.py --config config/offsetbias-8b.yaml

# run offsetbias with custom name on all benchmarks
python run_bench.py --name my_inference --config config/offsetbias-8b.yaml --benchmarks llmbar,hhh,mtbench,biasbench

# no inference, redo parsing on existing inference result
python run_bench.py --name my_inference --config config/offsetbias-8b.yaml --benchmarks biasbench --parse

# no inference, recalculate score
python run_bench.py --name my_inference --score
```

# Citation

If you find our work useful, please cite our paper:
```bibtex
@misc{park2024offsetbias,
      title={OffsetBias: Leveraging Debiased Data for Tuning Evaluators},
      author={Junsoo Park and Seungyeon Jwa and Meiying Ren and Daeyoung Kim and Sanghyuk Choi},
      year={2024},
      eprint={2407.06551},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
