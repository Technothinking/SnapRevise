from flask import Flask, request, render_template, redirect, url_for
import os
from ocr_nlp.ocr_engine import process_image
from ocr_nlp.nlp_generator import generate_questions, summarize_text  

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        image = request.files['image']
        if image:
            path = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
            image.save(path)

            # OCR step
            extracted_text = process_image(path)

            # Summary step
            summary = summarize_text(extracted_text)  # ⬅️ summarize text here

            # Flashcard generation step
            flashcards = generate_questions(extracted_text)

            # Pass both summary and flashcards to the template
            return render_template('flashcards.html', flashcards=flashcards, summary=summary)

    return render_template('uploads.html')

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
