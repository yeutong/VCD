# %%
import json
import os
from tqdm import tqdm
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
        # self.question_file = "../experiments/data/POPE/coco/coco_pope_popular.json"
        self.question_file = "../experiments/data/POPE/coco/coco_pope_adversarial.json"
        self.answers_file = "../results/answer.jsonl"
        self.image_folder = "../experiments/data/coco/val2014"

args = Args()


# %%
def generate_conv(question, processor=processor):
    conversation = [
        {

        "role": "user",
        "content": [
            {"type": "text", "text": f'{question} Please answer this question with one word.'},
            {"type": "image"},
            ],
        },
    ]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    return prompt
    
def convert_label(output):
    output = output.lower()
    if 'yes' in output and 'no' not in output:
        return 'yes'
    elif 'no' in output:
        return 'no'
    else:
        return 'other'
# %%


questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
answers_file = os.path.expanduser(args.answers_file)
os.makedirs(os.path.dirname(answers_file), exist_ok=True)
# ans_file = open(answers_file, "w")
score = 0
count = 0
for line in tqdm(questions):
    idx = line["question_id"]
    image_file = line["image"]
    qs = line["text"]
    label = line["label"]
    prompt = generate_conv(qs)
    image = Image.open(os.path.join(args.image_folder, image_file))

    inputs = processor(text=prompt, images=image, return_tensors="pt").to("cuda", torch.float16)

    # image_tensor = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
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
    # print(response, ' Label: ', label)


    pred = convert_label(response)
    if pred == label:
        score += 1 
    count += 1

# %%
print(score/count)
# %%
