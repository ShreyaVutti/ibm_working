from flask import Flask, render_template, request, redirect, url_for
from resume1 import extract_resume_info_from_pdf
from rag_backend import generate_rag_response
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'pdf', 'doc', 'docx'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def landing_page():
    return render_template('landing_page.html')

@app.route('/upload')
def upload_page():
    return render_template('upload.html')  # ✅ Correct name

@app.route('/upload', methods=['POST'])
def upload():
    if 'resume' not in request.files:
        return redirect(request.url)

    file = request.files['resume']
    if file.filename == '':
        return redirect(request.url)

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        parsed_output = extract_resume_info_from_pdf(file_path)
        rag_response = generate_rag_response(parsed_output)

        return render_template(
            'rag_output.html',
            parsed=parsed_output,
            rag_response=rag_response
        )

    return redirect(request.url)

if __name__ == '__main__':
    app.run(debug=True)
