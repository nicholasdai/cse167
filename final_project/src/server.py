import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import openai
import moviepy.editor as mp
from pydub import AudioSegment
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
PORT = 3001
CHUNK_DURATION_VIDEO = 60 # adjust off API, this just makes the most sense for now
CHUNK_DURATION_AUDIO = 60000 # adjust off API, this just makes the most sense for now
NUM_QUESTIONS = 15 # max 15 questions seem to fit

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ChatGPT was utilized to write the code for splitting into chunks
def split_video_into_chunks(video_path, chunk_duration):
    chunk_paths = []
    video = mp.VideoFileClip(video_path)
    video_duration = video.duration

    for start_time in range(0, int(video_duration), chunk_duration):
        end_time = min(start_time + chunk_duration, video_duration)
        chunk_filename = os.path.join(UPLOAD_FOLDER, f'chunk_{start_time}_{end_time}.mp4')
        chunk = video.subclip(start_time, end_time)
        chunk.write_videofile(chunk_filename, codec="libx264", audio_codec="aac")
        chunk_paths.append(chunk_filename)

    return chunk_paths

def split_audio_into_chunks(audio_path, chunk_duration):
    chunk_paths = []
    audio = AudioSegment.from_mp3(audio_path)
    audio_duration = len(audio)

    for start_time in range(0, int(audio_duration), chunk_duration):
        end_time = min(start_time + chunk_duration, audio_duration)
        chunk_filename = os.path.join(UPLOAD_FOLDER, f'chunk_{start_time}_{end_time}.mp3')
        chunk = audio[start_time:end_time]
        chunk.export(chunk_filename, format="mp3")
        chunk_paths.append(chunk_filename)

    return chunk_paths

@app.route('/transcribe', methods=['POST'])
def transcribe():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded.'}), 400

    file = request.files['file']

    if file.filename == '' or all(s not in (file.filename) for s in ['mp3', 'mp4']):
        return jsonify({'error': 'Invalid file type. Only mp4 and mp3 files are allowed.'}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)

    full_transcription = ""
    chunk_paths = split_video_into_chunks(file_path, CHUNK_DURATION_VIDEO) if file.filename.endswith('.mp4') else split_audio_into_chunks(file_path, CHUNK_DURATION_AUDIO)
            
    try:
        for chunk in chunk_paths:
            with open(chunk, 'rb') as f:
                transcription = openai.audio.transcriptions.create(
                    file=f,
                    model='whisper-1'
                )
                full_transcription += transcription.text + '\n'
            
            os.remove(chunk)

        completion = openai.chat.completions.create(
            model='gpt-3.5-turbo',
            messages=[{
                'role': 'user',
                'content': f"Create quiz questions based on the following audio transcription. The questions should test whether the person answering listened to the audio. Write {NUM_QUESTIONS} questions. Write each answer immediately after each question, but with the number of the question before each answer as well so that both questions and answers are numbered. it should be in the form of \" 1. question1\n 1. answer1\n 2. question2\n and so on \" Do not mess up the formatting:\n\n{full_transcription}"
            }]
        )

        content = completion.choices[0].message.content
        questions = '\n'.join(content.split('\n')[i] for i in range(0, len(content.split('\n')), 2))
        answers = '\n'.join(content.split('\n')[i] for i in range(1, len(content.split('\n')), 2))

        os.remove(file_path)

        return jsonify({
            'transcription': full_transcription,
            'questions': questions,
            'answers': answers
        })

    except Exception as e:
        return jsonify({'error': f'Error transcribing video/audio: {str(e)}'}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=PORT)