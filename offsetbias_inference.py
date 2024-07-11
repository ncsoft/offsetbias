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
