# 📰 Fake News Detection with OCR & Summarization

A powerful fake news detection tool that combines **Optical Character Recognition (OCR)**, **Text Summarization**, and **AI-powered fake news classification** into a single intuitive web application.

## ✨ Features

- **Text Input Analysis**: Detect fake news directly from text input
- **Image-to-Text OCR**: Extract text from images using Tesseract OCR
- **Automatic Summarization**: Summarize long articles using BART-Large-CNN
- **Real-time Prediction**: Classify news as real or fake with confidence scores
- **User-Friendly Interface**: Built with Gradio for an interactive web experience
- **Multi-Tab Interface**: Seamlessly switch between text and image analysis modes

## 🚀 Live Demo

**[Try the Live Demo Here](https://huggingface.co/spaces/jk2951/fnd_12)** 
## 🛠️ Technology Stack

| Component | Technology |
|-----------|-----------|
| **Fake News Detection** | BERT-based model (`jy46604790/Fake-News-Bert-Detect`) |
| **Text Summarization** | BART-Large-CNN (Facebook/Meta) |
| **OCR** | Tesseract OCR with Pytesseract |
| **Web Interface** | Gradio |
| **Image Processing** | PIL (Python Imaging Library) |
| **Deep Learning** | Transformers, PyTorch |

## 📋 Requirements

```bash
pip install transformers torch gradio pillow pytesseract numpy
