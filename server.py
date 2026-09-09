import logging
import os
import tempfile

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from werkzeug.exceptions import RequestEntityTooLarge
import worker

app = Flask(__name__)
CORS(app, resources={r'/*': {'origins': '*'}})
app.config['MAX_CONTENT_LENGTH'] = 20 * 1024 * 1024
app.logger.setLevel(logging.ERROR)


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/process-message', methods=['POST'])
def process_message_route():
    payload = request.get_json(silent=True)
    message = payload.get('userMessage') if isinstance(payload, dict) else None
    if not isinstance(message, str) or not message.strip():
        return jsonify(botResponse='Please enter a non-empty question.'), 400
    try:
        answer = worker.process_prompt(message.strip())
        return jsonify(botResponse=answer), 200
    except ValueError as exc:
        return jsonify(botResponse=str(exc)), 400
    except RuntimeError as exc:
        return jsonify(botResponse=str(exc)), 503
    except Exception as exc:
        app.logger.error('Answer generation failed (%s)', type(exc).__name__)
        return jsonify(botResponse='Unable to generate an answer. Check the Groq API key, service availability, and network connection.'), 502


@app.route('/process-document', methods=['POST'])
def process_document_route():
    file = request.files.get('file')
    if file is None or not file.filename:
        return jsonify(botResponse='Please select a PDF file to upload.'), 400
    if not file.filename.lower().endswith('.pdf'):
        return jsonify(botResponse='Only PDF files are supported.'), 400
    file_path = None
    try:
        # Never use an uploaded filename as a filesystem destination.
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp:
            file_path = temp.name
        file.save(file_path)
        with open(file_path, 'rb') as uploaded:
            if b'%PDF-' not in uploaded.read(1024):
                return jsonify(botResponse='This file is not a valid PDF.'), 400
        worker.process_document(file_path)
        return jsonify(botResponse='Your PDF is ready. You can now ask questions about it!'), 200
    except ValueError as exc:
        return jsonify(botResponse=str(exc)), 400
    except Exception as exc:
        app.logger.error('Document processing failed (%s)', type(exc).__name__)
        return jsonify(botResponse='Unable to process this PDF. Check that it is readable and not password-protected, and that the embedding model can be loaded.'), 422
    finally:
        if file_path and os.path.exists(file_path):
            os.remove(file_path)


@app.route('/reset', methods=['POST'])
def reset_route():
    worker.reset_document()
    return jsonify(botResponse='Chat reset. Please upload a PDF.'), 200


@app.errorhandler(RequestEntityTooLarge)
def upload_too_large(error):
    return jsonify(botResponse='The upload exceeds the 20 MB limit.'), 413


if __name__ == '__main__':
    port = int(os.environ.get('PORT', '8000'))
    app.run(debug=False, port=port, host='0.0.0.0')
