from flask import Flask

from youtube_transcript_api import YouTubeTranscriptApi
from transformers import T5ForConditionalGeneration, T5Tokenizer
import truecase

# initialize model pretrained
model = T5ForConditionalGeneration.from_pretrained("t5-base")
# initialize model tokenizer
tokenizer = T5Tokenizer.from_pretrained("t5-base")

app = Flask(__name__)


@app.route('/')
def index():
    return "test"


def format_transcript(summary_json):
    text = ""
    for i in summary_json:
        text += i['text']
    return text


@app.route('/summary/<video_id>/')
def summary_generator(video_id):
    summary = YouTubeTranscriptApi.get_transcript(video_id)
    text = format_transcript(summary)

    inputs = tokenizer.encode(
        "summarize: " + text,
        return_tensors="pt",
        max_length=512,
        truncation=True
    )

    # hypertune parameters later. what features to use for tuning these parameters?
    outputs = model.generate(
        inputs,
        max_length=150,
        min_length=40,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True)

    summary_text = str(tokenizer.decode(outputs[0]))
    return truecase.get_true_case(summary_text)
