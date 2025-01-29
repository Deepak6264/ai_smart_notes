import fitz  # PyMuPDF
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from dotenv import load_dotenv
import os
import ollama  # Ollama for local LLM processing
from fpdf import FPDF

# Load environment variables
load_dotenv()

# Initialize Flask
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend requests

# A global variable to store the PDF text
pdf_text = ""


def chunk_text(text, chunk_size=500):
    """
    Splits text into smaller chunks of a specified size.
    """
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]


def send_to_ollama(text):
    """
    Sends text chunk to Ollama for summarization.
    """
    try:
        response = ollama.chat(
            model="mistral",  # Change this to your preferred model (e.g., "llama3", "gemma", etc.)
            messages=[{"role": "user", "content": f"Summarize this text:\n\n{text}"}]
        )
        return response["message"]["content"] if "message" in response else "Error: No response"
    except Exception as e:
        return f"Error: {str(e)}"


@app.route('/api/upload', methods=['POST', 'GET'])
def upload_file():
    global pdf_text

    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files['file']

        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        if file and file.filename.endswith('.pdf'):
            try:
                # ✅ Load PDF and extract full text
                pdf_data = file.read()
                doc = fitz.open(stream=pdf_data, filetype="pdf")

                # Extract text from all pages
                pdf_text = "\n".join([page.get_text("text") for page in doc])

                return jsonify({"message": "PDF loaded and text extracted successfully"}), 200

            except Exception as e:
                return jsonify({"error": str(e)}), 500

        return jsonify({"error": "Invalid file format"}), 400

    elif request.method == 'GET':
        if not pdf_text:
            return jsonify({"error": "No PDF uploaded or processed"}), 400

        # ✅ Chunking the text
        text_chunks = chunk_text(pdf_text, chunk_size=500)

        # ✅ Summarizing each chunk using Ollama
        summaries = [send_to_ollama(chunk) for chunk in text_chunks]

        # ✅ Save summarized text into a new PDF
        summarized_pdf_path = "summarized_output.pdf"
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        pdf.set_font("Arial", size=12)

        for summary in summaries:
            pdf.multi_cell(0, 10, summary)
            pdf.ln(5)

        pdf.output(summarized_pdf_path)

        # ✅ Return JSON response with download link
        return jsonify({
            "message": "Summary generated successfully",
            "download_url": f"http://127.0.0.1:5000/api/download"
        }), 200


@app.route('/api/download', methods=['GET'])
def download_pdf():
    """Endpoint to download the summarized PDF."""
    summarized_pdf_path = "summarized_output.pdf"
    if not os.path.exists(summarized_pdf_path):
        return jsonify({"error": "No summarized PDF available"}), 400

    return send_file(summarized_pdf_path, as_attachment=True)


@app.route('/', methods=['GET'])
def home():
    return "Hello world!"


if __name__ == '__main__':
    app.run(debug=True, port=5000)
