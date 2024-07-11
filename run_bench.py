import json
import os
import argparse

from tqdm import tqdm
from omegaconf import OmegaConf
from datasets import load_dataset

from module import InferenceModule, VllmModule, HfModule, OpenaiModule


BENCHMARK_IDS = ["llmbar", "hhh", "mtbench", "biasbench"]


def make_data_row(id: int, instruction: str, response1: str, response2: str, label: int) -> dict:
    return {
        "id": id,
        "instruction": instruction.strip(),
        "response1": response1.strip(),
        "response2": response2.strip(),
        "label": label,
    }


def get_benchmark_data(benchmark_id: str) -> dict:
    """output a standardized dataset. only the contents.
    the data structure will be kept until the results."""
    benchmark_set = {}
    assert benchmark_id in BENCHMARK_IDS
    print("Loading benchmark:", benchmark_id)

    if benchmark_id == "llmbar":
        dataset = load_dataset("princeton-nlp/LLMBar", trust_remote_code=True)
        for subset_name in dataset.keys():
            subset = []
            for i, row in enumerate(dataset[subset_name]):
                subset.append(make_data_row(i, row["input"], row["output_1"], row["output_2"], row["label"]))
            benchmark_set[subset_name] = subset

    elif benchmark_id == "hhh":
        for subset_name in ["helpful", "honest", "harmless", "other"]:
            subset = []
            raw_subset = load_dataset("HuggingFaceH4/hhh_alignment", name=subset_name, trust_remote_code=True)["test"]
            for i, row in enumerate(raw_subset):
                label_data = row["targets"]["labels"]
                if label_data == [1, 0]:
                    label = 1
                elif label_data == [0, 1]:
                    label = 2
                else:
                    raise ValueError(label_data)
                subset.append(make_data_row(i, row["input"], row["targets"]
                              ["choices"][0], row["targets"]["choices"][1], label))
            benchmark_set[subset_name] = subset

    elif benchmark_id == "mtbench":
        raw_dataset = load_dataset("lmsys/mt_bench_human_judgments", trust_remote_code=True)
        for subset_name in ["human", "gpt4_pair"]:
            subset = []
            for i, row in enumerate(raw_dataset[subset_name]):
                if row["turn"] == 2:
                    continue
                label_data = row["winner"]
                if label_data == "model_a":
                    label = 1
                elif label_data == "model_b":
                    label = 2
                elif label_data in ["tie", "tie (inconsistent)"]:
                    continue
                else:
                    raise ValueError(label_data)
                subset.append(make_data_row(i, row["conversation_a"][0]["content"],
                              row["conversation_a"][1]["content"], row["conversation_b"][1]["content"], label))
            benchmark_set[subset_name] = subset

    elif benchmark_id == "biasbench":
        with open("data/biasbench/biasbench.json") as f:
            benchmark_set = json.load(f)
    else:
        raise ValueError(benchmark_id)

    return benchmark_set


def add_inference(benchmark_data: dict, module: InferenceModule) -> None:
    """all common logic for benchmarking. 
    apply swap, apply prompt template, apply chat template, for all subsets in benchmark data.
    run inference and update on benchmark_data"""
    conversation_list = []

    for subset_name, subset_data in benchmark_data.items():
        for row in subset_data:
            for swap in [False, True]:
                conversation_list.append(module.make_conversation(
                    row["instruction"], row["response1"], row["response2"], swap))

    generated_texts = module.generate(conversation_list)

    index = 0
    for subset_name, subset_data in benchmark_data.items():
        for row in subset_data:
            result = {}
            for swap_id in ["orig", "swap"]:
                result[swap_id] = {"completion": generated_texts[index]}
                index += 1
            row["result"] = result
    assert (len(generated_texts) == index)


def add_parse_result(benchmark_data: dict, module: InferenceModule) -> None:
    for subset_name, subset_data in benchmark_data.items():
        for row in subset_data:
            for swap, swap_id in [(False, "orig"), (True, "swap")]:
                result_dict = row["result"][swap_id]
                completion = result_dict["completion"]
                result_dict["prediction"] = module.get_prediction(completion)
                result_dict["is_correct"] = module.is_correct(result_dict["prediction"], row["label"], swap)


def get_model_statistics(run_name: str) -> dict:
    """read all inference results for the model and return scores"""
    model_stats = {}
    for benchmark_id in BENCHMARK_IDS:
        benchmark_result = {}
        filename = f"result/{run_name}/{benchmark_id}.json"
        if not os.path.exists(filename):
            print("result file", filename, "does not exist.")
            continue
        with open(filename) as f:
            data = json.load(f)
        for subset_name, subset in data.items():
            stats = {key: 0 for key in ["single_total", "single_correct", "single_accuracy",
                                        "pair_total", "pair_correct", "pair_accuracy", "pair_agree", "pair_agreement_rate"]}
            for row in subset:
                stats["single_total"] += 2
                stats["pair_total"] += 1
                if row["result"]["orig"]["is_correct"]:
                    stats["single_correct"] += 1
                if row["result"]["swap"]["is_correct"]:
                    stats["single_correct"] += 1
                if row["result"]["orig"]["is_correct"] and row["result"]["swap"]["is_correct"]:
                    stats["pair_correct"] += 1
                pred_orig = row["result"]["orig"]["prediction"]
                pred_swap = row["result"]["swap"]["prediction"]
                if set([pred_orig, pred_swap]) in [set([1, 2]), set([3])]:
                    stats["pair_agree"] += 1

            stats["single_accuracy"] = round(stats["single_correct"] / stats["single_total"]*100, 1)
            stats["pair_accuracy"] = round(stats["pair_correct"] / stats["pair_total"]*100, 1)
            stats["pair_agreement_rate"] = round(stats["pair_agree"] / stats["pair_total"]*100, 1)
            benchmark_result[subset_name] = stats
        model_stats[benchmark_id] = benchmark_result
    return model_stats


def write_model_score(run_name: str) -> None:
    """create model's score file"""
    model_stats = get_model_statistics(run_name)

    with open(f"result/{run_name}/score.json", "w") as f:
        json.dump(model_stats, fp=f, ensure_ascii=False, indent=4)


def run_benchmark(run_name: str, args: argparse.Namespace) -> None:
    """run inference, parse and score."""
    os.makedirs("result", exist_ok=True)
    os.makedirs(f"result/{run_name}", exist_ok=True)

    config = OmegaConf.load(args.config)
    OmegaConf.save(config, f"result/{run_name}/config.yaml")
    print(config)

    if (not args.hf) and (config.get("vllm_args")):
        module = VllmModule(config=config)
    elif (args.hf) and (config.get("hf_args")):
        module = HfModule(config=config)
    elif config.get("openai_args"):
        module = OpenaiModule(config=config)
    else:
        raise NotImplementedError

    for benchmark_id in args.benchmarks:
        benchmark_data = get_benchmark_data(benchmark_id)

        add_inference(benchmark_data, module)
        add_parse_result(benchmark_data, module)

        with open(f"result/{run_name}/{benchmark_id}.json", "w") as f:
            json.dump(benchmark_data, fp=f, ensure_ascii=False, indent=2)

    write_model_score(run_name)


def run_parse(run_name: str, args: argparse.Namespace) -> None:
    """redo parsing for existing inference results, and update score."""
    config = OmegaConf.load(args.config)
    print(config)

    module = InferenceModule(config=config)
    for benchmark_id in args.benchmarks:
        with open(f"result/{run_name}/{benchmark_id}.json") as f:
            benchmark_data = json.load(f)
        add_parse_result(benchmark_data, module)
        with open(f"result/{run_name}/{benchmark_id}.json", "w") as f:
            json.dump(benchmark_data, fp=f, ensure_ascii=False, indent=2)

    write_model_score(run_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/offsetbias-8b.yaml")
    parser.add_argument("--name", default="", help="run name of the inference. defaults to config name.")

    def list_of_strings(arg):
        return arg.split(',')
    parser.add_argument("--benchmarks", type=list_of_strings, default=["biasbench"],
                        help="to include all benchmarks, call as '--benchmarks llmbar,hhh,mtbench,biasbench'")
    parser.add_argument("--hf", action="store_true", help="use hf generate instead of vllm")
    parser.add_argument("--parse", action="store_true", help="no inference. just parse and score.")
    parser.add_argument("--score", action="store_true", help="no inference. just score.")

    args = parser.parse_args()
    print(args)

    run_name = os.path.basename(args.config).replace(".yaml", "")
    if args.hf:
        run_name += "_hf"
    if args.name:
        run_name = args.name
    print("Run name:", run_name)

    if args.score:
        write_model_score(run_name)
    elif args.parse:
        run_parse(run_name, args)
    else:
        run_benchmark(run_name, args)
