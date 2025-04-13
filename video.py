import os
import sys
import tempfile
import base64
import torch
from datetime import timedelta
import streamlit as st
from PIL import Image
import cv2
import numpy as np
import whisper
from moviepy import VideoFileClip, AudioFileClip, AudioClip
import google.generativeai as genai
from pytube import YouTube
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoImageProcessor,
    AutoModelForObjectDetection,
    ViTFeatureExtractor,
    ViTForImageClassification
)

# Setting up GEmini
# GEMINI_API_KEY = "AIzaSyCIFYZq6dWZDCy0W_ZsBFooadFy6PgR-HA"
api_key = st.secrets["api"]["key"]
genai.configure(api_key=api_key)

# FFmpeg configuration
if sys.platform == "win32":
    ffmpeg_path = "C:\\ffmpeg\\bin\\ffmpeg.exe"
else:
    ffmpeg_path = "ffmpeg"

os.environ["IMAGEIO_FFMPEG_EXE"] = ffmpeg_path
os.environ["FFMPEG_BINARY"] = ffmpeg_path

# Model loading
@st.cache_resource
def load_models():
    models = {
        "whisper": whisper.load_model("base"),
        "scene_processor": ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k"),
        "scene_model": ViTForImageClassification.from_pretrained("google/vit-base-patch16-224-in21k"),
        "object_processor": AutoImageProcessor.from_pretrained("facebook/detr-resnet-50"),
        "object_model": AutoModelForObjectDetection.from_pretrained("facebook/detr-resnet-50"),
        "summarizer_tokenizer": AutoTokenizer.from_pretrained("facebook/bart-large-cnn"),
        "summarizer_model": AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")
    }
    return models

def get_video_duration(video_path):
    with VideoFileClip(video_path) as clip:
        return clip.duration
    
def extract_frames(video_path, interval=5):
    frames = []
    timestamps = []
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * interval)
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_count % frame_interval == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
            timestamps.append(frame_count / fps)
            
        frame_count += 1
    
    cap.release()
    return frames, timestamps


def detect_scenes(frames, timestamps, models):
    scene_info = []
    
    for idx, frame in enumerate(frames):
        img = Image.fromarray(frame)
        
        # Scene classification
        scene_inputs = models["scene_processor"](images=img, return_tensors="pt")
        scene_outputs = models["scene_model"](**scene_inputs)
        scene_label = scene_outputs.logits.argmax(-1).item()
        scene_desc = models["scene_model"].config.id2label[scene_label]
        
        # Object detection
        obj_inputs = models["object_processor"](images=img, return_tensors="pt")
        obj_outputs = models["object_model"](**obj_inputs)
        
        target_sizes = torch.tensor([img.size[::-1]])
        results = models["object_processor"].post_process_object_detection(
            obj_outputs, target_sizes=target_sizes, threshold=0.7
        )[0]
        
        objects = []
        for label in results["labels"].unique():
            obj_name = models["object_model"].config.id2label[label.item()]
            objects.append(obj_name)
        
        scene_info.append({
            "timestamp": timestamps[idx],
            "scene_description": scene_desc,
            "objects": objects,
            "frame": frame
        })
    
    return scene_info

def transcribe_audio(video_path, model):
    result = model.transcribe(video_path)
    return result

def translate_text(text):
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content(f"Translate to English: {text}")
    return response.text

def summarize_text(text, models):
    inputs = models["summarizer_tokenizer"](text, max_length=1024, truncation=True, return_tensors="pt")
    summary_ids = models["summarizer_model"].generate(
        inputs["input_ids"],
        max_length=150,
        min_length=50,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True
    )
    return models["summarizer_tokenizer"].decode(summary_ids[0], skip_special_tokens=True)

def generate_theme_highlights(video_path, theme, models, duration=30):
    try:
        # Process transcription
        transcription = transcribe_audio(video_path, models["whisper"])
        translated = translate_text(transcription['text'])
        
        # Get timestamp suggestion
        model = genai.GenerativeModel('gemini-pro')
        prompt = f"""Analyze this transcript for {theme} highlights:
        {translated}
        Suggest START and END timestamps in seconds (format: 123.4-456.7)"""
        response = model.generate_content(prompt)
        
        # Parse response
        times = response.text.split('\n')[0].split('-')
        start, end = float(times[0]), float(times[1])
        end = min(start + duration, end)
        
        # Generate clip
        with VideoFileClip(video_path) as clip:
            subclip = clip.subclip(start, end)
            
            # Handle missing audio
            if not subclip.audio:
                audio = AudioClip(lambda t: [0,0], duration=subclip.duration)
                subclip = subclip.set_audio(audio)
            
            highlight_path = os.path.join(tempfile.gettempdir(), "highlight.mp4")
            subclip.write_videofile(
                highlight_path,
                codec="libx264",
                audio_codec="aac",
                ffmpeg_params=["-ar", "16000"]
            )
            
        return highlight_path, response.text
    
    except Exception as e:
        return None, f"Highlight generation failed: {str(e)}"

def generate_banner(frames, summary_text):
    try:
        middle_frame = frames[len(frames)//2]
        img = Image.fromarray(middle_frame)
        
        # Add text overlay
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(img)
        
        try:
            font = ImageFont.truetype("arial.ttf", 30)
        except:
            font = ImageFont.load_default()
        
        text = "\n".join([summary_text[i:i+40] for i in range(0, len(summary_text), 40)])
        draw.text((10, 10), text, fill="white", font=font, stroke_width=2, stroke_fill="black")
        
        banner_path = os.path.join(tempfile.gettempdir(), "banner.jpg")
        img.save(banner_path)
        return banner_path
    
    except Exception as e:
        st.error(f"Banner generation failed: {str(e)}")
        return None

def format_time(seconds):
    return str(timedelta(seconds=seconds)).split(".")[0]

def main():
    st.set_page_config(layout="wide", page_title="Arabic Video Analyzer")
    st.title("📹 Arabic Video Analysis Platform")
    
    # Load models
    models = load_models()
    
    # File upload
    video_file = st.file_uploader("Upload Arabic Video", type=["mp4", "mov"])
    youtube_url = st.text_input("Or Enter YouTube URL")
    
    video_path = None
    if youtube_url:
        try:
            yt = YouTube(youtube_url)
            stream = yt.streams.filter(progressive=True, file_extension='mp4').first()
            video_path = stream.download(output_path=tempfile.gettempdir())
            st.success("YouTube video downloaded!")
        except Exception as e:
            st.error(f"YouTube Error: {str(e)}")
    
    if video_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
            tmp_file.write(video_file.read())
            video_path = tmp_file.name
    
    if video_path:
        st.video(video_path)
        duration = get_video_duration(video_path)
        st.caption(f"Duration: {format_time(duration)}")
        
        if st.button("Analyze Video"):
            with st.spinner("Processing..."):
                try:
                    # Core processing
                    frames, timestamps = extract_frames(video_path)
                    scenes = detect_scenes(frames, timestamps, models)
                    transcription = transcribe_audio(video_path, models["whisper"])
                    translated = translate_text(transcription['text'])
                    summary = summarize_text(translated, models)
                    
                    # UI Tabs
                    tab1, tab2, tab3, tab4 = st.tabs([
                        "Transcript", "Scenes", "Highlights", "Report"
                    ])
                    
                    with tab1:
                        st.subheader("Arabic Transcription")
                        st.write(transcription['text'])
                        st.subheader("English Translation")
                        st.write(translated)
                        st.subheader("Summary")
                        st.write(summary)
                    
                    with tab2:
                        for scene in scenes:
                            # with st.expander(f"{format_time(scene['timestamp']} - {scene['scene_description']}"):
                            with st.expander(f"{format_time(scene['timestamp'])} - {scene['scene_description']}"):

                                col1, col2 = st.columns([1, 2])
                                with col1:
                                    st.image(scene['frame'])
                                with col2:
                                    st.write("**Objects Detected:**")
                                    st.write(", ".join(scene['objects']))
                    
                    with tab3:
                        theme = st.text_input("Enter highlight theme")
                        if theme:
                            highlight_path, explanation = generate_theme_highlights(
                                video_path, theme, models
                            )
                            if highlight_path:
                                st.write(explanation)
                                st.video(highlight_path)
                                with open(highlight_path, "rb") as f:
                                    st.download_button(
                                        "Download Highlight",
                                        f.read(),
                                        file_name="highlight.mp4"
                                    )
                    
                    with tab4:
                        banner_path = generate_banner(frames, summary)
                        if banner_path:
                            st.image(banner_path)
                            with open(banner_path, "rb") as f:
                                st.download_button(
                                    "Download Banner",
                                    f.read(),
                                    file_name="banner.jpg"
                                )
                        
                        st.subheader("Technical Report")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Total Scenes", len(scenes))
                            st.write("Common Objects:")
                            objects = [obj for scene in scenes for obj in scene['objects']]
                            st.write(", ".join(sorted(set(objects))[:500] + "..."))
                        
                        with col2:
                            st.metric("Audio Length", f"{transcription['segments'][-1]['end']:.1f}s")
                            st.write("Key Frames:")
                            st.image([scene['frame'] for scene in scenes[:3]], width=200)
                
                except Exception as e:
                    st.error(f"Processing Error: {str(e)}")

if __name__ == "__main__":
    main()