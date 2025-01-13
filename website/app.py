from flask import Flask, request, jsonify, Response 
from flask_cors import CORS
from model import synthesize_speech
import soundfile as sf
import os
import io

app = Flask(__name__)
CORS(app)

@app.route('/synthesize', methods=['POST'])
def predict():
    try:
        data =  request.get_json()
        text = data.get('text', "")

        if not text:
            return jsonify({'error': 'No text is provided'}), 400

        waveform = synthesize_speech(text)

        def generate():
            buffer = io.BytesIO()
            sf.write(buffer, waveform, samplerate=22050, format='wav')
            buffer.seek(0)
            data = buffer.read(1024)
            while data:
                yield data
                data = buffer.read(1024)

        return Response(generate(), mimetype="audio/wav")

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)