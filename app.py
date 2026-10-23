
import os

from flask import Flask, jsonify, render_template, request
from openai import OpenAI

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024  # Limit request size

# Configure an API key using the OPENAI_API_KEY environment variable.
api_key = os.getenv("OPENAI_API_KEY", "").strip()
client = OpenAI(api_key=api_key) if api_key else None

MEDICAL_INSTRUCTIONS = """
You are a health information chatbot for educational purposes.

Rules:
- Provide clear, empathetic, general health information.
- Never claim to diagnose a patient or replace a clinician.
- Do not prescribe medicines or recommend personalized dosages.
- Explain when symptoms warrant professional medical evaluation.
- Ask relevant follow-up questions when useful.
- If symptoms could indicate an emergency, tell the user to contact
  local emergency services immediately.
- Do not claim certainty when the information is insufficient.
- Keep answers concise and easy to understand.
- Do not request names, addresses, or other unnecessary personal data.
"""

EMERGENCY_TERMS = [
    "can't breathe",
    "cannot breathe",
    "difficulty breathing",
    "chest pain",
    "heart attack",
    "stroke symptoms",
    "face drooping",
    "suicidal",
    "kill myself",
    "overdose",
    "unconscious",
    "severe bleeding",
]

FALLBACK_RESPONSES = {
    "headache": (
        "Headaches can have many causes, including dehydration, "
        "stress, lack of sleep, or illness. Rest and drink water "
        "if appropriate. Seek urgent medical help for a sudden, "
        "extremely severe headache or one with weakness, confusion, "
        "fever with a stiff neck, or vision changes."
    ),
    "fever": (
        "Fever can occur with many infections. Rest, drink fluids, "
        "and monitor your symptoms. Seek medical advice for a "
        "persistent or worsening fever. Fever in a baby under "
        "3 months (38°C / 100.4°F or higher) requires urgent "
        "medical assessment."
    ),
    "cough": (
        "Coughs can result from infections, allergies, or irritation. "
        "Rest and fluids may help. Seek medical care for a persistent "
        "or worsening cough. Severe breathing difficulty, blue lips, "
        "or significant chest pain require emergency care."
    ),
    "water": (
        "Fluid needs vary with activity, climate, age, and health. "
        "Drink regularly and follow any fluid restrictions prescribed "
        "by your healthcare professional."
    ),
    "sleep": (
        "A regular sleep schedule, a relaxing bedtime routine, "
        "limiting late caffeine, and a comfortable dark bedroom "
        "may improve sleep. Persistent sleep problems are worth "
        "discussing with a healthcare professional."
    ),
    "stress": (
        "Slow breathing, gentle movement, regular sleep, and talking "
        "with someone you trust may help manage stress. If stress "
        "is overwhelming or persistent, consider speaking with "
        "a qualified mental health professional."
    ),
    "diet": (
        "A balanced diet generally includes vegetables, fruit, "
        "whole grains, protein sources, and healthy fats. Individual "
        "needs vary, especially with medical conditions or allergies."
    ),
}


def fallback_answer(question):
    text = question.lower()

    if any(term in text for term in EMERGENCY_TERMS):
        return (
            "Your message may describe a medical emergency. Please "
            "contact your local emergency number or go to the nearest "
            "emergency department now. Do not wait for an online "
            "chatbot to assess these symptoms."
        )

    for keyword, answer in FALLBACK_RESPONSES.items():
        if keyword in text:
            return (
                answer
                + "\n\nThis is general information, not a diagnosis. "
                "Consult a qualified healthcare professional for "
                "advice about your individual situation."
            )

    return (
        "I am currently running in offline demo mode, so I cannot "
        "generate an AI answer to that question. Try asking about "
        "headaches, fever, cough, sleep, stress, diet, or hydration. "
        "For a broader range of questions, configure an OpenAI API key."
    )


def get_answer(question):
    # Emergency keyword check before calling the AI service.
    text = question.lower()

    if any(term in text for term in EMERGENCY_TERMS):
        return (
            "Your message may describe a medical emergency. Please "
            "contact your local emergency number or go to the nearest "
            "emergency department now. Do not wait for an online "
            "chatbot to assess these symptoms."
        )

    if client is None:
        return fallback_answer(question)

    try:
        response = client.responses.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
            instructions=MEDICAL_INSTRUCTIONS,
            input=question,
            max_output_tokens=500,
        )

        answer = response.output_text.strip()
        if answer:
            return answer

        return (
            "I could not generate a response. Please try again "
            "or contact a healthcare professional."
        )

    except Exception:
        app.logger.exception("AI service request failed")
        return (
            "Sorry, the AI service is temporarily unavailable. "
            "Check your API key and internet connection, then try again. "
            "For medical concerns, contact a qualified healthcare "
            "professional."
        )


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({"error": "Please send a valid chat message."}), 400

    question = data.get("message", "")

    if not isinstance(question, str):
        return jsonify({"error": "Message must be text."}), 400

    question = question.strip()

    if not question:
        return jsonify({"error": "Please enter a question."}), 400

    if len(question) > 2000:
        return jsonify({
            "error": "Your question is too long. Please use 2000 characters or fewer."
        }), 400

    answer = get_answer(question)
    return jsonify({"answer": answer})


@app.errorhandler(413)
def request_too_large(_error):
    return jsonify({"error": "Request is too large."}), 413


if __name__ == "__main__":
    # Debug mode is disabled for safer default operation.
    app.run(host="127.0.0.1", port=5000, debug=False)
