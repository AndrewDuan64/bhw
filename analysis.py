from typing import List, Dict, Tuple
import re
import json
import spacy
from transformers import pipeline

MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment"

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading spaCy model 'en_core_web_sm'...")
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model=MODEL_NAME,
    return_all_scores=True,
    device=-1,  # CPU only
)

LABEL_MAP = {
    "LABEL_0": "NEGATIVE",
    "LABEL_1": "NEUTRAL",
    "LABEL_2": "POSITIVE",
}

DEFAULT_FILTER={"um", "like", "you know"}

def load_transcript(filepath: str = "transcript.txt") -> List[Dict]:
    """
    Load and parse transcript into a list of dialogue turns.
    Each turn is a dictionary: {'speaker': 'A' or 'B', 'text': ...}
    Ensures alternating speakers and proper formatting.
    """
    dialogue = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except UnicodeDecodeError:
        with open(filepath, 'r', encoding='latin-1') as f:
            lines = f.readlines()

    if not lines:
        raise ValueError("Transcript file is empty.")

    previous_speaker = None

    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue # Skip blank lines

        match = re.match(r"Speaker\s([AB]):\s(.+)", line)
        if match:
            speaker = match.group(1)
            text = match.group(2).strip()

            if previous_speaker is not None and previous_speaker == speaker:
                raise ValueError(f"Line {i+1}: Speakers should alternate. Found two consecutive lines from '{speaker}'.")

            if not text:
                raise ValueError(f"Line {i+1}: Dialogue text cannot be empty.")

            dialogue.append({
                "speaker": speaker,
                "text": text
            })
            previous_speaker = speaker
        else:
            raise ValueError(f"Line {i+1}: Format error. Line does not match 'Speaker A/B: text' format. Line content: '{line}'")

    if len(dialogue) < 12:
        raise ValueError("Transcript must contain at least 12 lines of dialogue.")
    elif len(dialogue) > 16:
        raise ValueError("Transcript must contain no more than 16 lines of dialogue.")

    return dialogue


def compute_sentiment(text: str) -> Tuple[str, float]:
    """
    Returns (readable_label, sentiment_score)

    readable_label  : 'NEGATIVE' / 'NEUTRAL' / 'POSITIVE'
    sentiment_score : P(positive) - P(negative) ∈ [-1, 1]
    On failure, returns ("UNKNOWN", 0.0)
    """
    
    def _analyze_single_sentence(sentence: str) -> Tuple[str, float, float]:
        """Analyze single sentence -> (readable_label, score, top_class_confidence)"""
        outputs = sentiment_pipeline(sentence[:512])[0]
        scores = {LABEL_MAP[d["label"]]: d["score"] for d in outputs}
        label = max(scores, key=scores.get)
        confidence = scores[label]
        score = scores.get("POSITIVE", 0.0) - scores.get("NEGATIVE", 0.0)
        return label, score, confidence

    try:
        token_threshold = 512
        if len(text) <= token_threshold:
            label, score, _ = _analyze_single_sentence(text)
            return label, score

        sentences = re.split(r'(?<=[.!?。！？])\s+', text.strip())
        sentences = [s for s in sentences if s]

        if not sentences:
            return _analyze_single_sentence(text)[:2]

        total_score = 0.0
        best_label, best_conf = "UNKNOWN", -1.0

        for sentence in sentences:
            label, score, confidence = _analyze_single_sentence(sentence)
            total_score += score
            if confidence > best_conf:
                best_conf = confidence
                best_label = label

        avg_score = total_score / len(sentences)
        return best_label, avg_score

    except Exception as e:
        print(f"[ERROR] Sentiment analysis failed: {e}")
        return "UNKNOWN", 0.0

def compute_filler_ratio(text: str, fillers=None) -> Tuple[float, str]:
    """
    Computes:
    - filler ratio (float)
    - Markdown-highlighted version of the input text (str)
    
    Filler words are wrapped in **bold**.
    """
    try:
        if fillers is None:
            fillers = DEFAULT_FILTER

        text_original = text.strip()
        text = text_original.lower()
        doc = nlp(text)
        words = [token.text for token in doc if token.is_alpha]

        if not words:
            return 0.0, text_original

        total_words = len(words)
        filler_count = 0
        i = 0
        tokens = [token.text for token in doc]

        # Build token-by-token for reconstruction
        output_tokens = []
        text_words = [token.text for token in doc]
        fillers = sorted(fillers, key=lambda x: -len(x.split()))

        while i < len(text_words):
            matched = False
            for filler in fillers:
                filler_tokens = filler.split()
                candidate = text_words[i:i + len(filler_tokens)]

                if [w.lower() for w in candidate] == filler_tokens:
                    filler_count += 1
                    bolded = f"**{' '.join(candidate)}**"
                    output_tokens.append(bolded)
                    i += len(filler_tokens)
                    matched = True
                    break
            if not matched:
                output_tokens.append(text_words[i])
                i += 1

        result_md = ' '.join(output_tokens)
        return round(filler_count / total_words, 4), result_md

    except Exception as e:
        print(f"[ERROR] spaCy filler computation failed: {e}")
        return 0.0, text

if __name__ == "__main__":
    transcript = load_transcript()
    print(json.dumps(transcript, indent=2, ensure_ascii=False))

    example_sentiment = compute_sentiment("It's just a normal day.")
    print(example_sentiment)

    filler_ratio = compute_filler_ratio("Um, I mean, it's like, you know, actually kind of weird.")
    print(filler_ratio)
