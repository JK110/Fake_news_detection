import gradio as gr
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from PIL import Image
import pytesseract
import numpy as np
import re

# Load Fake News Detection model
MODEL_NAME = "jy46604790/Fake-News-Bert-Detect"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

# Load summarization model
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def clean_text(text):
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    return text.strip()

def ocr_image(image):
    if image is None:
        return ""
    if not isinstance(image, Image.Image):
        image = Image.fromarray(np.uint8(image))
    text = pytesseract.image_to_string(image)
    return clean_text(text)

def summarize_text(text):
    if len(text) < 50:
        return text
    summary = summarizer(text, max_length=130, min_length=30, do_sample=False)
    return summary[0]['summary_text']

def predict_text(text):
    if not text.strip():
        return "❌ Please enter some text."
    result = classifier(text)[0]
    label = "🟢 Real News" if result["label"] == "LABEL_1" else "🔴 Fake News"
    confidence = round(result["score"] * 100, 2)
    return f"{label} ({confidence}% confidence)"

# This function combines OCR + summarization + prediction
def ocr_summarize_predict(image):
    raw_text = ocr_image(image)
    if not raw_text:
        return "⚠️ No text found in image.", ""
    summarized_text = summarize_text(raw_text)
    prediction = predict_text(summarized_text)
    return summarized_text, prediction

with gr.Blocks() as app:
    gr.Markdown("📰 **Fake News Detector with OCR + Summarization + Auto Prediction**\n\nUpload an image or enter text.")

    with gr.Tab("Text Input"):
        text_input = gr.Textbox(lines=4, label="Enter News Text")
        text_output = gr.Textbox(label="Prediction")
        text_btn = gr.Button("Detect")
        text_btn.click(predict_text, inputs=text_input, outputs=text_output)

    with gr.Tab("Image Upload"):
        img_input = gr.Image(type="numpy", label="Upload News Image")
        extracted_text = gr.Textbox(label="Extracted & Summarized Text")
        prediction_output = gr.Textbox(label="Prediction")

        # When image uploaded or button clicked, run OCR + summarize + predict
        img_input.change(ocr_summarize_predict, inputs=img_input, outputs=[extracted_text, prediction_output])
        # Optional: also trigger on button click
        # btn = gr.Button("Analyze Image")
        # btn.click(ocr_summarize_predict, inputs=img_input, outputs=[extracted_text, prediction_output])

app.launch()
