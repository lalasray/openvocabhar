from sentence_transformers import SentenceTransformer
import vec2text
import transformers
import torch 
from transformers import AutoModelForCausalLM, AutoTokenizer
#huggingface-cli login

if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"

device = torch.device(device)
#print(device)

inversion_model = vec2text.models.InversionModel.from_pretrained("ielabgroup/vec2text_gtr-base-st_inversion")
model = vec2text.models.CorrectorEncoderModel.from_pretrained("ielabgroup/vec2text_gtr-base-st_corrector")
inversion_trainer = vec2text.trainers.InversionTrainer(model=inversion_model,train_dataset=None,eval_dataset=None,data_collator=transformers.DataCollatorForSeq2Seq(inversion_model.tokenizer,label_pad_token_id=-100,),)
model.config.dispatch_batches = None
corrector = vec2text.trainers.Corrector(model=model,inversion_trainer=inversion_trainer,args=None,data_collator=vec2text.collator.DataCollatorForCorrection(tokenizer=inversion_trainer.model.tokenizer),)
model = SentenceTransformer('sentence-transformers/gtr-t5-base')
embeddings = model.encode(["A person is running","a person in jumping", "A person is eating", "A person is eating and crying"], convert_to_tensor=True,).to(device)
text = vec2text.invert_embeddings(embeddings=embeddings,corrector=corrector,num_steps=2000,)
print(text)

#llm = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", device_map="auto")
#tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
#sentence = text[0]
#prompt = f"Extract all the main activities from the following sentence and describe them in the format 'A person is <activity>':\n{sentence}\n"
#model_inputs = tokenizer(prompt, return_tensors="pt").to(device)
#generated_ids = llm.generate(**model_inputs, max_new_tokens=10, do_sample=True)
#generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
#activities = generated_text.split("A person is")[-1].strip().split('.')
#activities = [activity.strip() for activity in activities if activity.strip()]
#formatted_outputs = [f"A person is {activity}" for activity in activities]

#for output in formatted_outputs:
#    print(output)