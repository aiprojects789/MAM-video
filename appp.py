import streamlit as st
import tempfile, os, shutil, time
import cv2
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from moviepy import AudioFileClip  # Correct import for AudioFileClip
from google.cloud import speech_v1p1beta1 as speech
from google.cloud import videointelligence_v1 as vi, vision
import boto3
from PIL import Image, ImageDraw, ImageFont
from openai import OpenAI
import google.generativeai as genai
import json


# # --- Configuration & Clients ---
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "project-alpha-456519-c49943658d77.json"

google_credentials = st.secrets["google_cloud"]

credentials_dict = {
    "type": "service_account",
    "project_id": google_credentials["project_id"],
    "private_key_id": google_credentials["private_key_id"],
    # "private_key": google_credentials["private_key"].strip(),
    "private_key": google_credentials["private_key"].replace('\\n', '\n'),
    "client_email": google_credentials["client_email"],
    "client_id": google_credentials["client_id"],
    "auth_uri": google_credentials["auth_uri"],
    "token_uri": google_credentials["token_uri"],
    "auth_provider_x509_cert_url": google_credentials["auth_provider_x509_cert_url"],
    "client_x509_cert_url": google_credentials["client_x509_cert_url"]
}


# Write credentials to a temporary file
with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".json") as temp_file:
    json.dump(credentials_dict, temp_file)
    temp_file_path = temp_file.name

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = temp_file_path




# OpenAI v1 client
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])


# Google Speech‑to‑Text client
speech_client = speech.SpeechClient()

# GCP & AWS clients
vi_client = vi.VideoIntelligenceServiceClient()
vision_client = vision.ImageAnnotatorClient()
rekognition = boto3.client(
    'rekognition',
    aws_access_key_id=st.secrets["AWS_ACCESS_KEY"],
    aws_secret_access_key=st.secrets["AWS_SECRET_KEY"],
    region_name="us-east-1"
)

st.title("🎬 AI-Powered Video Analyzer")

# --- Helpers ---
def save_temp_video(uploaded):
    temp_dir = tempfile.mkdtemp()
    path = os.path.join(temp_dir, uploaded.name)
    with open(path, "wb") as f: f.write(uploaded.read())
    return path, temp_dir


def transcribe(path):
    st.info("Transcribing with Google Cloud Speech‑to‑Text…")
    
    # build output path
    base, _ = os.path.splitext(path)
    audio_path = base + "_audio.wav"
    
    clip = AudioFileClip(path)
    
    # always write as mono at 16 kHz
    clip.write_audiofile(
        audio_path,
        fps=16000,
        ffmpeg_params=["-ac", "1"],    # <— force 1 audio channel
        logger=None
    )
    
    # load bytes and delete temp file
    with open(audio_path, "rb") as f:
        audio_content = f.read()
    os.remove(audio_path)

    # configure Google Speech API
    audio = speech.RecognitionAudio(content=audio_content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code="en-US",
        enable_automatic_punctuation=True
    )
    response = speech.SpeechClient().recognize(config=config, audio=audio)
    
    # return the joined transcript
    time.sleep(5)
    return " ".join(r.alternatives[0].transcript for r in response.results)





def detect_scenes(path):
    st.info("Detecting shots…")
    with open(path, "rb") as f: content = f.read()
    req = vi.AnnotateVideoRequest(
        input_content=content,
        features=[vi.Feature.SHOT_CHANGE_DETECTION]
    )
    op = vi_client.annotate_video(request=req)
    time.sleep(5)  # Added sleep after scene detection
    return op.result(timeout=300).annotation_results[0].shot_annotations

def save_frame(path, t, outp):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
    ret, frame = cap.read()
    if ret: cv2.imwrite(outp, frame)
    cap.release()
    return outp


def analyze_frame(image_bytes):
    labels = []

    # Amazon Rekognition: Detect objects
    try:
        aws = rekognition.detect_labels(Image={"Bytes": image_bytes}, MaxLabels=10)
        labels += [l["Name"] for l in aws.get("Labels", [])]
    except Exception as e:
        st.error(f"AWS Rekognition (labels) error: {e}")

    # 🔍 Amazon Rekognition: Recognize celebrities
    try:
        celebs = rekognition.recognize_celebrities(Image={"Bytes": image_bytes})
        labels += [c["Name"] for c in celebs.get("CelebrityFaces", [])]
    except Exception as e:
        st.error(f"AWS Rekognition (celebrities) error: {e}")

    # Google Vision: Detect objects
    try:
        gimg = vision.Image(content=image_bytes)
        gv = vision_client.label_detection(image=gimg)
        labels += [l.description for l in gv.label_annotations]
    except Exception as e:
        st.error(f"Google Vision error: {e}")

    time.sleep(5)  # Pause to avoid API overload
    return list(dict.fromkeys(labels))[:15]



# def summarize_scene(transcript, labels):
#     model = genai.GenerativeModel('gemini-1.5-flash')
#     prompt = (
#         f"Create a concise scene summary (2–3 sentences) capturing:\n"
#         f"- Visual elements: {', '.join(labels)}\n"
#         f"- Transcript context: {transcript}\n"
#         f"- Emotional tone"
#     )
#     return model.generate_content(prompt).text.strip()



def summarize_scene(transcript, labels, temperature=0.7):
    """
    Generates a 2–3 sentence summary of a video scene using OpenAI's GPT-4 Turbo.

    Args:
        transcript (str): Transcript text from the video scene.
        labels (list): List of visual elements detected from the scene.
        temperature (float): Sampling temperature (0.0 = deterministic, 1.0 = creative).

    Returns:
        str: Natural-language scene summary.
    """
    
    prompt = (
        "You are an assistant that summarizes video scenes for storytelling purposes.\n"
        "Given the transcript and visual elements below, create a vivid 2–3 sentence summary. "
        "Describe the scene's main actions, setting, atmosphere, and emotional tone naturally.\n\n"
        f"Transcript:\n{transcript}\n\n"
        f"Visual Elements:\n{', '.join(labels)}"
    )

    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": "You summarize video scenes in a descriptive, cinematic style."},
            {"role": "user", "content": prompt}
        ],
        temperature=temperature,
        top_p=0.9,
        max_tokens=200
    )

    return response.choices[0].message.content.strip()



def generate_video_title(transcript):
    resp = client.chat.completions.create(
        model="gpt-4",
        messages=[{
            "role": "system", "content": "You are a creative video marketer."
        }, {
            "role": "user", "content":
                f"Generate a short (3–5 word), catchy title for this video:\n\n{transcript[:1000]}"
        }],
        temperature=0.7,
    )
    time.sleep(5)  # Added sleep after generating video title
    return resp.choices[0].message.content.strip()




def extract_clip(path, start, end, outp):
    return ffmpeg_extract_subclip(path, start, end, outp)


# def _make_mask(frame_path):
#     # create a mask: bottom third white, rest transparent
#     img = Image.open(frame_path)
#     mask = Image.new("RGBA", img.size, (0,0,0,0))
#     draw = ImageDraw.Draw(mask)
#     h_third = img.height // 3
#     draw.rectangle([(0, img.height - h_third), (img.width, img.height)], fill="white")
#     mask_path = frame_path.replace(".jpg", "_mask.png")
#     mask.save(mask_path)
#     return mask_path



def create_dynamic_banner(frame_path, video_title, scene_summary, style):
    img = Image.open(frame_path).convert("RGBA")
    width, height = img.size

    # Create overlay
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 120))
    img = Image.alpha_composite(img, overlay)

    draw = ImageDraw.Draw(img)

    try:
        font_title = ImageFont.truetype("arial.ttf", size=48)
        font_sub = ImageFont.truetype("arial.ttf", size=32)
    except:
        font_title = ImageFont.load_default()
        font_sub = ImageFont.load_default()

    # Draw the title and summary
    padding = 40
    draw.text((padding, padding), video_title, font=font_title, fill="white")
    draw.text((padding, padding + 70), scene_summary, font=font_sub, fill="lightgray")

    out_path = frame_path.replace(".jpg", "_banner.jpg")
    img.convert("RGB").save(out_path)
    return out_path







# --- Session-State Init ---
if 'uploaded' not in st.session_state:
    st.session_state.update({
        'uploaded': None,
        'transcript': "",
        'scene_data': [],
        'video_title': ""
    })

# --- UI: Upload & Buttons ---
uploaded = st.file_uploader("Upload a video (30s–4min)", type=["mp4","mov","avi"])
if uploaded:
    st.session_state.uploaded, tmp = save_temp_video(uploaded)

    if st.button("1️⃣ Transcribe Video"):
        st.session_state.transcript = transcribe(st.session_state.uploaded)

    if st.button("2️⃣ Detect & Analyze Scenes"):
        shots = detect_scenes(st.session_state.uploaded)
        data = []
        for i, s in enumerate(shots):
            stime, etime = s.start_time_offset.total_seconds(), s.end_time_offset.total_seconds()
            fp = os.path.join(tmp, f"frame_{i}.jpg")
            save_frame(st.session_state.uploaded, (stime+etime)/2, fp)
            with open(fp, "rb") as f:
                lbls = analyze_frame(f.read())
            words = st.session_state.transcript.split()
            sub = " ".join(words[i*50:(i+1)*50])
            sm = summarize_scene(sub, lbls)
            data.append({"start": stime, "end": etime, "frame": fp, "labels": lbls, "summary": sm})
            time.sleep(1)  # Optional sleep for scene processing, added here for additional pacing
        st.session_state.scene_data = data

    # --- Tabs: Transcript / Analysis / Promo / Banner ---
    tab1, tab2, tab3, tab4 = st.tabs([
        "📝 Transcript", "🎞 Scene Analysis",
        "🎥 Promo Clip", "🖼 Banner Generator"
    ])

    with tab1:
        st.header("Full Transcript")
        if st.session_state.transcript:
            st.write(st.session_state.transcript)
        else:
            st.info("Click **Transcribe Video** first.")

    with tab2:
        st.header("Scene Breakdowns")
        for i, sc in enumerate(st.session_state.scene_data):
            with st.expander(f"Scene {i+1} ({sc['start']:.1f}s–{sc['end']:.1f}s)"):
                c1, c2 = st.columns([1, 3])
                with c1:
                    st.image(sc["frame"], width=200)
                    st.write(f"**Labels:** {', '.join(sc['labels'])}")
                with c2:
                    st.write(f"**Summary:** {sc['summary']}")

    with tab3:
        st.header("Promo Clip Creator")
        query = st.text_input("Search scenes for:")
        if query:
            matches = [
                s for s in st.session_state.scene_data
                if query.lower() in " ".join(s["labels"]).lower()
                or query.lower() in s["summary"].lower()
            ]
            sel = st.selectbox("Matches", matches,
                               format_func=lambda x: f"{x['start']:.1f}s–{x['end']:.1f}s")
            length = st.slider("Clip length", 5, 60, 30)
            if st.button("Extract Clip"):
                cs = max(sel["start"] - 5, 0)
                ce = min(cs + length, sel["end"])
                clip = os.path.join(tmp, "promo.mp4")
                extract_clip(st.session_state.uploaded, cs, ce, clip)
                st.video(clip)
                st.download_button("Download Clip", open(clip, "rb"), "promo.mp4")

    
    with tab4:
        st.header("🖼 Dynamic Banner Generator")
        keyword = st.text_input("Enter a label/keyword (e.g., 'mountain', 'goal'):")
        style = st.selectbox("Banner Style", ["Hero Banner","Promo Banner","Social Media"])
        if st.button("Generate Banner"):
            # find matching scene
            match = next(
                (s for s in st.session_state.scene_data
                if any(keyword.lower() in lbl.lower() for lbl in s["labels"])
                or keyword.lower() in s["summary"].lower()),
                None
            )
            if not match:
                st.error(f"No scene found containing '{keyword}'.")
            else:
                # extract the midpoint frame if not already saved:
                frame_path = match["frame"]
                # ensure we have a title (or regenerate)
                if not st.session_state.video_title:
                    st.session_state.video_title = generate_video_title(
                        st.session_state.transcript
                    )
                banner_url = create_dynamic_banner(
                    frame_path,
                    st.session_state.video_title,
                    match["summary"],
                    style
                )
                # st.image(banner_url, caption="Your AI‑designed banner")
                st.image(banner_url, caption="Banner from Real Frame")
                st.download_button("Download Banner", open(banner_url, "rb"), "banner.jpg")



    # Cleanup
    if st.button("🗑️ Clear All"):
        shutil.rmtree(tmp, ignore_errors=True)
        for k in ["uploaded", "transcript", "scene_data", "video_title"]:
            st.session_state.pop(k, None)
