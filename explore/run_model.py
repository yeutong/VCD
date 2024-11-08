# %%
import requests
from PIL import Image

import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration, AutoTokenizer


device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_id = "llava-hf/llava-1.5-7b-hf"
model = LlavaForConditionalGeneration.from_pretrained(
    model_id, 
    torch_dtype=torch.float16, 
    low_cpu_mem_usage=True, 
).to(device)

processor = AutoProcessor.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)
# context_len = model.config.max_position_embeddings

# %%
class Args:
    def __init__(self):
        self.question_file = "../experiments/data/POPE/coco/coco_pope_popular.json"
        self.answers_file = "../results/answer.jsonl"
        self.conv_mode = "llava_v1"
        self.image_folder = "../experiments/data/coco/val2014"

args = Args()

# %%
import json
import os
from tqdm import tqdm

questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
answers_file = os.path.expanduser(args.answers_file)
os.makedirs(os.path.dirname(answers_file), exist_ok=True)
ans_file = open(answers_file, "w")
for line in tqdm(questions):
    idx = line["question_id"]
    image_file = line["image"]
    # qs = line["text"]
    # cur_prompt = qs
    # # if model.config.mm_use_im_start_end:
    # if getattr(model.config, 'mm_use_im_start_end', False):
    #     qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
    # else:
    #     qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

    # conv = conv_templates[args.conv_mode].copy()
    # conv.append_message(conv.roles[0], qs + " Please answer this question with one word.")
    # conv.append_message(conv.roles[1], None)
    # prompt = conv.get_prompt()

    # input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

    image = Image.open(os.path.join(args.image_folder, image_file))
    # image_tensor = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
    break
# %%
idx
# %%
# Sample question from POPE COCO
question = "Are there any cats in the image?"

# Prepare the prompt
prompt = f"USER: <image>\n{question}? ASSISTANT:"

# Process the inputs
inputs = processor(text=prompt, images=image, return_tensors="pt").to("cuda", torch.float16)

# Generate the response
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
    )

# Decode and print the response
response = processor.decode(outputs[0], skip_special_tokens=True)
print(response)
# %%
conversation = [
    {

      "role": "user",
      "content": [
          {"type": "text", "text": "What are these?"},
          {"type": "image"},
        ],
    },
]
prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

image_file = "http://images.cocodataset.org/val2017/000000039769.jpg"
raw_image = Image.open(requests.get(image_file, stream=True).raw)
inputs = processor(images=raw_image, text=prompt, return_tensors='pt').to(0, torch.float16)

output = model.generate(**inputs, max_new_tokens=200, do_sample=False)
print(processor.decode(output[0][2:], skip_special_tokens=True))
# %%
# %%
processor.tokenizer.padding_side = "left"
# %%
from transformers.processing_utils import ProcessingKwargs

class LlavaProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {
            "padding": False,
        },
        "images_kwargs": {
        },
    }

output_kwargs = processor._merge_kwargs(
    LlavaProcessorKwargs,
    tokenizer_init_kwargs=processor.tokenizer.init_kwargs,
    padding=True,
    # return_tensors="pt",
)
# %%
image_inputs = processor.image_processor(image, **output_kwargs["images_kwargs"])
# %%
# processor(text=prompt, images=image, return_tensors="pt", padding=True)
# %%
text_inputs = processor.tokenizer(prompt, **output_kwargs["text_kwargs"])
# %%
from transformers import BatchFeature
inputs = BatchFeature(data={**text_inputs, **image_inputs}).to(0, torch.float16)
# %%
output = model.generate(**inputs, max_new_tokens=200, do_sample=False)
# %%
import requests
from PIL import Image

import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration

model_id = "llava-hf/llava-1.5-7b-hf"
model = LlavaForConditionalGeneration.from_pretrained(
    model_id, 
    torch_dtype=torch.float16, 
    low_cpu_mem_usage=True, 
    load_in_4bit=True
)

processor = AutoProcessor.from_pretrained(model_id)

# Define a chat histiry and use `apply_chat_template` to get correctly formatted prompt
# Each value in "content" has to be a list of dicts with types ("text", "image") 
conversation = [
    {

      "role": "user",
      "content": [
          {"type": "text", "text": "What are these?"},
          {"type": "image"},
        ],
    },
]
prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
# prompt = 'testing 123'
image_file = "http://images.cocodataset.org/val2017/000000039769.jpg"
raw_image = Image.open(requests.get(image_file, stream=True).raw)
inputs = processor(images=raw_image, text=prompt, return_tensors='pt').to(0, torch.float16)

output = model.generate(**inputs, max_new_tokens=200, do_sample=False)
print(processor.decode(output[0][2:], skip_special_tokens=True))

# %%
