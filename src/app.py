# waitress-serve --listen=0.0.0.0:8000 src.app:app

import logging
from flask import Flask, jsonify, request
from flask_cors import CORS
import time
import logging

from langchain_ollama import ChatOllama


LOG_LEVEL = "INFO"
MODEL_GENERATION = "deepseek-r1"
MODEL_EMBEDDING = ""
OLLAMA_BASE_URL = "http://localhost:11434"


logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)s %(message)s")

app = Flask(__name__)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


llm = ChatOllama(model=MODEL_GENERATION, base_url=OLLAMA_BASE_URL, temperature=0.8)


CORS(
    app, 
    resources={r"/*": {"origins": "*"}},
    supports_credentials=False,
    allow_headers=["Content-Type", "Authorization"],
    methods=["GET", "POST", "OPTIONS"],
    max_age=86400)

@app.route("/health", methods=["GET"])
def health_check():
    logger.info("health_check method=%s path=%s remote=%s", request.method, request.path, request.remote_addr)
    return jsonify({"status":"ok"}), 200

@app.route("/chat", methods=["POST"])
def chat():

    start_time = time.perf_counter()

    payload = request.get_json(force=True, silent=False) or {}

    question = payload.get("question")

    if not question or not isinstance(question, str):
        return jsonify({"error": "question is required and must be a string"}), 400

    try:
        response = llm.invoke(question)
        answer = getattr(response, "content", None)
        if answer is None:
            answer = str(response)
    except Exception:
        logger.exception("llm_invoke_failed")
        return jsonify({"error": "failed to generate response"}), 500



    latency = time.perf_counter() - start_time

    logger.info("chat method=%s path=%s remote=%s latency=%.6fs",request.method,request.path,request.remote_addr,latency)
    
    return jsonify({
        "answer": answer,
        "latency":latency
    }), 200
