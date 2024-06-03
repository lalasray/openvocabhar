from sentence_transformers import SentenceTransformer
import vec2text
import transformers
import torch 
from transformers import AutoModelForCausalLM, AutoTokenizer

if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"

device = torch.device(device)

inversion_model = vec2text.models.InversionModel.from_pretrained("ielabgroup/vec2text_gtr-base-st_inversion")
model = vec2text.models.CorrectorEncoderModel.from_pretrained("ielabgroup/vec2text_gtr-base-st_corrector")
inversion_trainer = vec2text.trainers.InversionTrainer(model=inversion_model,train_dataset=None,eval_dataset=None,data_collator=transformers.DataCollatorForSeq2Seq(inversion_model.tokenizer,label_pad_token_id=-100,),)
model.config.dispatch_batches = None
corrector = vec2text.trainers.Corrector(model=model,inversion_trainer=inversion_trainer,args=None,data_collator=vec2text.collator.DataCollatorForCorrection(tokenizer=inversion_trainer.model.tokenizer),)
model = SentenceTransformer('sentence-transformers/gtr-t5-base')
embeddings = model.encode(["A person is running","a person in jumping"], convert_to_tensor=True,).to(device)
text = vec2text.invert_embeddings(embeddings=embeddings,corrector=corrector,num_steps=1,)
print(text)
new_embeddig = embeddings.mean(dim=0, keepdim=True).cuda()
text = vec2text.invert_embeddings(embeddings=new_embeddig,corrector=corrector,num_steps=1,)
print(text)
