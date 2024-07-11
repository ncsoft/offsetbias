import importlib
import re

from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm import LLM, SamplingParams
from openai import OpenAI


class InferenceModule():
    def __init__(self, prompt_name: str = "", config: dict = {}):
        self.config = config

        prompt_name = config.get("prompt", prompt_name)
        prompt_module = importlib.import_module(f"prompt.{prompt_name}")
        self.prompt_name: str = prompt_name
        self.system_message: str = prompt_module.system if "system" in dir(prompt_module) else ""
        self.user_message_template: str = prompt_module.user
        self.output_pattern: dict = prompt_module.output_pattern

    def make_conversation(self, instruction: str, response1: str, response2: str, swap: bool) -> list:
        conversation = []

        if self.system_message:
            conversation.append({"role": "system", "content": self.system_message})

        user_message = self.user_message_template.format(
            input=instruction,
            output_1=response1 if not swap else response2,
            output_2=response2 if not swap else response1,
        )
        conversation.append({"role": "user", "content": user_message})

        return conversation

    def get_prediction(self, output_text: str) -> int:
        """parse output text into prediction label: 1(A), 2(B), 3(TIE), 4(N/A)"""
        for prediction, pattern in self.output_pattern.items():
            if re.search(pattern, output_text):
                return prediction
        return 4

    def is_correct(self, prediction: int, label: int, swap: bool = False) -> bool:
        if not swap:
            return prediction == label and label in [1, 2]
        else:
            return prediction + label == 3 and prediction in [1, 2] and label in [1, 2]


class VllmModule(InferenceModule):
    def __init__(
            self,
            prompt_name: str = "",
            model_name: str = "",
            dtype: str = "float16",
            temperature: float = 0.0,
            max_tokens: int = 20,
            config: dict = {}):

        super().__init__(prompt_name=prompt_name, config=config)

        print("Initializing vllm model...")
        vllm_args = self.config.get("vllm_args", {})

        model_args = dict(model=model_name, dtype=dtype)
        model_args.update(vllm_args.get("model_args", {}))
        print("model args:", model_args)
        self.model_name = model_args["model"]
        tokenizer_name = self.config.get("tokenizer", self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.model = LLM(**model_args)

        sampling_params_args = dict(temperature=temperature, max_tokens=max_tokens)
        sampling_params_args.update(vllm_args.get("sampling_params", {}))
        self.sampling_params = SamplingParams(**sampling_params_args)
        print(self.sampling_params)

    def generate(self, conversation_list: list) -> list:
        if "prometheus" in self.model_name:
            from fastchat.conversation import get_conv_template

            def _get_conversation_prompt(messages):
                """
                From filled prompt, convert it into llama-2 conversation prompt
                """
                conv = get_conv_template("mistral")

                for message in messages:
                    if message["role"] == "system":
                        conv.set_system_message(message["content"])
                    elif message["role"] == "user":
                        conv.append_message(conv.roles[0], message["content"])

                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()
                return prompt
            inputs = [_get_conversation_prompt(conversation).strip() for conversation in conversation_list]
            outputs = self.model.generate(inputs, sampling_params=self.sampling_params)

        elif "PandaLM" in self.model_name:
            inputs = [conversation[0]['content'] for conversation in conversation_list]
            outputs = self.model.generate(inputs, sampling_params=self.sampling_params)

        else:
            # llama3 style
            prompt_token_ids = [self.tokenizer.apply_chat_template(
                conversation, add_generation_prompt=True) for conversation in conversation_list]
            outputs = self.model.generate(prompt_token_ids=prompt_token_ids, sampling_params=self.sampling_params)

        generated_texts = [output.outputs[0].text.strip() for output in outputs]
        return generated_texts


class HfModule(InferenceModule):
    def __init__(
            self,
            model_name: str = "",
            dtype: str = "float16",
            max_new_tokens: int = 20,
            pad_token_id: int = None,
            do_sample: bool = False,
            temperature: float = 0.0,
            config: dict = {}):

        super().__init__(config=config)

        print("Initializing hf model...")
        hf_args = self.config.get("hf_args", {})

        model_name = hf_args.get("model_args", {}).get("model", model_name)
        dtype_name = hf_args.get("model_args", {}).get("dtype", dtype)

        dtype_mapping = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
        torch_dtype = dtype_mapping[dtype_name]

        tokenizer_name = self.config.get("tokenizer", model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch_dtype, device_map="auto").eval()  # use_auth_token = args.hf_use_auth_token
        self.generate_kwargs = dict(max_new_tokens=max_new_tokens, pad_token_id=pad_token_id,
                                    do_sample=do_sample, temperature=temperature)
        self.generate_kwargs.update(hf_args.get("generate_kwargs", {}))
        print("generate_kwargs:", self.generate_kwargs)

    def generate(self, conversation_list: list) -> list:
        generated_texts = []
        for conversation in tqdm(conversation_list):
            # llama3 style
            input_ids = self.tokenizer.apply_chat_template(
                conversation, tokenize=True, add_generation_prompt=True, return_tensors="pt")
            with torch.inference_mode():
                if torch.cuda.is_available():
                    input_ids = input_ids.cuda()
                generation = self.model.generate(input_ids=input_ids, **self.generate_kwargs)
                completion = self.tokenizer.decode(
                    generation[0][len(input_ids[0]):], skip_special_tokens=True, clean_up_tokenization_spaces=True)
            generated_texts.append(completion.strip())
        return generated_texts


class OpenaiModule(InferenceModule):
    def __init__(self, config: dict):
        super().__init__(config=config)
        print("Initializing openai client...")
        openai_args = self.config["openai_args"]
        self.client = OpenAI(**openai_args)
        self.create_args = self.config["create_args"]

    def generate(self, conversation_list: list) -> list:
        generated_texts = []
        for conversation in tqdm(conversation_list):
            response = self.client.chat.completions.create(
                messages=conversation,
                **self.create_args
            )
            generated_text = response.choices[0].message.content
            generated_texts.append(generated_text)
        return generated_texts
