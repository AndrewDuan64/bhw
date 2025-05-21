import json
from typing import Tuple
import re
import spacy
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment"

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

def load_transcript(filepath: str = "transcript.txt"):
    dialogue = []
    try:
        # Try reading with UTF-8 encoding first, fallback to latin-1 if needed
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
                continue

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

    except FileNotFoundError:
        print(f"[ERROR] File '{filepath}' not found.")
        return []
    except ValueError as ve:
        print(f"[ERROR] {ve}")
        return []
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        return []

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


# import json
# from typing import Tuple
# import re
# import spacy

# from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# MODEL = "cardiffnlp/twitter-roberta-base-sentiment"

# nlp = spacy.load("en_core_web_sm")

# sentiment_pipeline = pipeline(
#     "sentiment-analysis",
#     model=MODEL,
#     return_all_scores=True,
#     device=-1,        # CPU only
# )

# label_map = {
#     "LABEL_0": "NEGATIVE",
#     "LABEL_1": "NEUTRAL",
#     "LABEL_2": "POSITIVE",
# }

# def load_transcript(filepath="transcript.txt"):
#     dialogue = []
#     try:
#         # Try to read the file with UTF-8 encoding first, then fallback to latin-1 encoding if necessary
#         try:
#             with open(filepath, 'r', encoding='utf-8') as f:
#                 lines = f.readlines()
#         except UnicodeDecodeError:
#             with open(filepath, 'r', encoding='latin-1') as f:
#                 lines = f.readlines()

#         # If the file is empty, raise an error
#         if not lines:
#             raise ValueError("Transcript file is empty.")

#         # Variable to keep track of the previous speaker (to ensure alternating speakers)
#         previous_speaker = None

#         # Iterate through each line in the file
#         for i, line in enumerate(lines):
#             line = line.strip()  # Remove leading and trailing whitespace

#             if not line:
#                 continue  # Skip empty lines

#             # Match format: Speaker A: text or Speaker B: text
#             match = re.match(r"Speaker\s([AB]):\s(.+)", line)
#             if match:
#                 speaker = match.group(1)  # Extract the speaker (A or B)
#                 text = match.group(2).strip()  # Extract the dialogue text

#                 # Ensure that speakers alternate (i.e., no consecutive turns from the same speaker)
#                 if previous_speaker is not None and previous_speaker == speaker:
#                     raise ValueError(f"Line {i+1}: Speakers should alternate. Found two consecutive lines from '{speaker}'.")

#                 # Ensure the dialogue text is not empty
#                 if not text:
#                     raise ValueError(f"Line {i+1}: Dialogue text cannot be empty.")

#                 # Add the speaker and their text to the dialogue list
#                 dialogue.append({
#                     "speaker": speaker,
#                     "text": text
#                 })
#                 previous_speaker = speaker  # Update the previous speaker
#             else:
#                 raise ValueError(f"Line {i+1}: Format error. Line does not match 'Speaker A/B: text' format. Line content: '{line}'")

#         # Ensure the transcript contains between 12 and 16 lines of dialogue
#         if len(dialogue) < 12:
#             raise ValueError("Transcript must contain at least 12 lines of dialogue.")
#         elif len(dialogue) > 16:
#             raise ValueError("Transcript must contain no more than 16 lines of dialogue.")

#         return dialogue

#     except FileNotFoundError:
#         print(f"[ERROR] File '{filepath}' not found.")
#         return []
#     except ValueError as ve:
#         print(f"[ERROR] {ve}")
#         return []
#     except Exception as e:
#         print(f"[ERROR] Unexpected error: {e}")
#         return []



# def compute_sentiment(text: str) -> Tuple[str, float]:
#     """
#     返回 (readable_label, sentiment_score)

#     readable_label  : 'NEGATIVE' / 'NEUTRAL' / 'POSITIVE'
#     sentiment_score : P(positive) - P(negative) ∈ [-1, 1]
#     如失败，返回 ("UNKNOWN", 0.0)
#     """
#     def _sentiment_once(sentence: str) -> Tuple[str, float, float]:
#         """单句情感分析 -> (可读标签, 分值, 最高类别概率)"""
#         outputs = sentiment_pipeline(sentence[:512])[0]
#         scores = {label_map[d["label"]]: d["score"] for d in outputs}

#         # 本句标签与置信度
#         label = max(scores, key=scores.get)
#         confidence = scores[label]

#         # 本句情感分值
#         score = scores.get("POSITIVE", 0.0) - scores.get("NEGATIVE", 0.0)
#         return label, score, confidence

#     try:
#         # 1) 判断是否需要分句
#         token_threshold = 512
#         if len(text) <= token_threshold:            # 简单长度近似；严格处理可接 tokenizer
#             label, score, _ = _sentiment_once(text)
#             return label, score

#         # 2) 超长文本 -> 分句
#         sentences = re.split(r'(?<=[.!?。！？])\s+', text.strip())
#         sentences = [s for s in sentences if s]     # 去空串

#         if not sentences:                           # 极端 fallback
#             return _sentiment_once(text)[:2]

#         total_score = 0.0
#         best_label, best_conf = "UNKNOWN", -1.0

#         for s in sentences:
#             label, score, conf = _sentiment_once(s)
#             total_score += score
#             if conf > best_conf:
#                 best_conf = conf
#                 best_label = label

#         avg_score = total_score / len(sentences)
#         return best_label, avg_score

#     except Exception as e:
#         print(f"[ERROR] Sentiment analysis failed: {e}")
#         return "UNKNOWN", 0.0



# def compute_filler_ratio(text: str, fillers=None) -> float:
#     try:
#         # 如果没有传入FILLERS，使用默认值
#         if fillers is None:
#             fillers = {"um", "like", "you know"}
        
#         # 文本预处理：将文本转换为小写并去除额外空格
#         text = text.strip().lower()
#         doc = nlp(text)

#         # 筛选出有效的单词
#         words = [token.text for token in doc if token.is_alpha]

#         if not words:
#             return 0.0

#         total_words = len(words)
#         filler_count = 0
#         i = 0

#         # 预先按长度排序填充词
#         fillers = sorted(fillers, key=lambda x: -len(x.split()))

#         while i < len(words):
#             matched = False
#             for filler in fillers:
#                 filler_tokens = filler.split()
#                 # 如果当前片段与填充词匹配
#                 if words[i:i+len(filler_tokens)] == filler_tokens:
#                     filler_count += 1
#                     i += len(filler_tokens)
#                     matched = True
#                     break
#             if not matched:
#                 i += 1

#         print(filler_count, total_words)
#         return round(filler_count / total_words, 4)
#     except Exception as e:
#         print(f"[ERROR] spaCy filler computation failed: {e}")
#         return 0.0



# if __name__ == "__main__":
#     transcript_data = load_transcript()
#     print(json.dumps(transcript_data, indent=2, ensure_ascii=False))

#     text_sentiment = compute_sentiment("It's just a normal day.")
#     print(text_sentiment)

#     ratio = compute_filler_ratio("Um, I mean, it's like, you know, actually kind of weird.")
#     print(ratio)