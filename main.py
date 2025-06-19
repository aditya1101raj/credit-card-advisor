from transformers import pipeline
from transformers.pipelines import task

model = pipieline(task:"summarization",model="facebook/bart-large-cnn")
response=model("text to summarise")
print(response)