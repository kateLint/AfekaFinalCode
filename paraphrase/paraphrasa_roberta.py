from __future__ import annotations

import os
import json
import textwrap
import warnings
import threading
import logging
import time
from dataclasses import dataclass
from typing import Tuple, List, Dict, Any

import numpy as np
import pandas as pd
import joblib

import nltk
from nltk.tokenize import sent_tokenize

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import optuna
from keybert import KeyBERT

# =========================
# Logging Setup
# =========================
def setup_logging(level: str = "INFO") -> logging.Logger:
    """Setup comprehensive logging for the application."""
    logger = logging.getLogger("kickstarter_ai")
    logger.setLevel(getattr(logging, level.upper()))
    
    # Create handlers
    console_handler = logging.StreamHandler()
    file_handler = logging.FileHandler("kickstarter_ai.log")
    
    # Create formatters and add it to handlers
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
    formatter = logging.Formatter(log_format)
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    # Add handlers to the logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger

# Initialize logger
LOGGER = setup_logging()

# Hide these specific warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn.utils.extmath")

# Also suppress numpy-level warnings
np.seterr(all="ignore")

# =========================
# Config & Constants
# =========================
@dataclass
class Config:
    """Configuration class for all settings."""
    # Model paths
    model_path: str = os.environ.get(
        "KS_MODEL_PATH",
        "/Users/kerenlint/Projects/cursor/runs/xgb_roberta_20250810_095255/xgboost_kickstarter_success_model.pkl",
    )
    features_path: str = os.environ.get(
        "KS_FEATURES_PATH",
        "/Users/kerenlint/Projects/cursor/runs/xgb_roberta_20250810_095255/xgboost_feature_columns.json",
    )
    
    # RoBERTa settings
    roberta_model_name: str = os.environ.get("KS_ROBERTA_NAME", "sentence-transformers/roberta-base-nli-mean-tokens")
    embed_dim: int = 768
    story_prefix: str = "story_roberta_embedding_"
    risk_prefix: str = "risk_roberta_embedding_"
    
    # Text processing settings
    max_tokens: int = 512
    stride: int = 64
    batch_size: int = 16
    coherence_threshold: float = 0.60
    
    # Optimization settings
    optuna_trials: int = 10
    paraphrase_return_count: int = 4
    
    # Performance settings
    enable_logging: bool = True
    log_level: str = "INFO"
    
    def validate(self) -> None:
        """Validate configuration settings."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model path not found: {self.model_path}")
        if not os.path.exists(self.features_path):
            raise FileNotFoundError(f"Features path not found: {self.features_path}")
        if self.embed_dim <= 0:
            raise ValueError(f"Invalid embed_dim: {self.embed_dim}")
        if self.coherence_threshold < 0 or self.coherence_threshold > 1:
            raise ValueError(f"Invalid coherence_threshold: {self.coherence_threshold}")

# Initialize configuration
CONFIG = Config()
CONFIG.validate()

# Backward compatibility
MODEL_PATH = CONFIG.model_path
FEATURES_PATH = CONFIG.features_path
ROBERTA_MODEL_NAME = CONFIG.roberta_model_name
EMBED_DIM = CONFIG.embed_dim
STORY_PREFIX = CONFIG.story_prefix
RISK_PREFIX = CONFIG.risk_prefix

# Minimal numeric inputs expected from the user
USER_NUMERIC_KEYS = [
    "goal",
    "rewardscount",
    "projectFAQsCount",
    "project_length_days",
    "preparation_days",
]

# A minimal sample category key (extend to your full set as needed)
USER_CATEGORY_KEYS = [
    "category_Web_Development",
]

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def check_dependencies() -> bool:
    """Check if all required dependencies are available."""
    missing_deps = []
    
    try:
        import psutil
    except ImportError:
        missing_deps.append("psutil")
    
    try:
        import nltk
        nltk.data.find('tokenizers/punkt')
    except (ImportError, LookupError):
        missing_deps.append("nltk punkt tokenizer")
        try:
            nltk.download('punkt')
        except Exception as e:
            LOGGER.warning(f"Failed to download punkt: {e}")
    
    if missing_deps:
        LOGGER.warning(f"Missing optional dependencies: {missing_deps}")
        return False
    
    return True

# Check dependencies
check_dependencies()


def set_seeds(seed: int = 1337) -> None:
    np.random.seed(seed)
    try:
        import random
        random.seed(seed)
    except Exception:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seeds()


# =========================
# Loading models
# =========================
@dataclass
class Models:
    paraphraser_model: Any
    paraphraser_tokenizer: Any
    embedder: SentenceTransformer
    clf_model: Any
    clf_features: List[str]
    kw_model: KeyBERT
    device: torch.device
    roberta_tokenizer: Any 



def load_models() -> Models:
    """Load all required models with comprehensive error handling and logging."""
    start_time = time.time()
    LOGGER.info("Starting model loading process...")
    
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        LOGGER.info(f"Using device: {device}")

        # Paraphraser (T5 base)
        LOGGER.info("Loading paraphraser model...")
        paraphraser_model = AutoModelForSeq2SeqLM.from_pretrained(
            "humarin/chatgpt_paraphraser_on_T5_base"
        ).to(device)
        paraphraser_tokenizer = AutoTokenizer.from_pretrained(
            "humarin/chatgpt_paraphraser_on_T5_base"
        )
        LOGGER.info("✅ Paraphraser model loaded successfully")

        # RoBERTa embedder (768-dim)
        LOGGER.info(f"Loading RoBERTa embedder: {ROBERTA_MODEL_NAME}")
        embedder = SentenceTransformer(ROBERTA_MODEL_NAME, device="cuda" if device.type == "cuda" else None)
        
        # Validate embedding size early
        _probe = np.asarray(embedder.encode("probe"), dtype=float)
        if _probe.ndim != 1 or _probe.shape[0] != EMBED_DIM:
            raise RuntimeError(
                f"Embedding dim mismatch. Expected {EMBED_DIM}, got {_probe.shape[0]} from {ROBERTA_MODEL_NAME}."
            )
        LOGGER.info(f"✅ RoBERTa embedder loaded successfully (dim: {_probe.shape[0]})")

        # Classifier
        LOGGER.info(f"Loading classifier model from: {MODEL_PATH}")
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Classifier model not found at: {MODEL_PATH}")
        clf_model = joblib.load(MODEL_PATH)
        LOGGER.info("✅ Classifier model loaded successfully")

        # Feature list (column order)
        LOGGER.info(f"Loading feature columns from: {FEATURES_PATH}")
        if not os.path.exists(FEATURES_PATH):
            raise FileNotFoundError(f"Features JSON not found at: {FEATURES_PATH}")
        with open(FEATURES_PATH, "r") as f:
            clf_features = json.load(f)
        LOGGER.info(f"✅ Feature columns loaded successfully ({len(clf_features)} features)")

        # Keyword extraction (KeyBERT using same family space is OK; can be any)
        LOGGER.info("Loading KeyBERT model...")
        kw_model = KeyBERT(ROBERTA_MODEL_NAME)
        LOGGER.info("✅ KeyBERT model loaded successfully")

        # RoBERTa tokenizer
        LOGGER.info("Loading RoBERTa tokenizer...")
        roberta_tokenizer = AutoTokenizer.from_pretrained(ROBERTA_MODEL_NAME, use_fast=True)
        LOGGER.info("✅ RoBERTa tokenizer loaded successfully")

        models = Models(
            paraphraser_model=paraphraser_model,
            paraphraser_tokenizer=paraphraser_tokenizer,
            embedder=embedder,
            clf_model=clf_model,
            clf_features=clf_features,
            kw_model=kw_model,
            device=device,
            roberta_tokenizer=roberta_tokenizer,  
        )
        
        load_time = time.time() - start_time
        LOGGER.info(f"🎉 All models loaded successfully in {load_time:.2f} seconds")
        return models
        
    except Exception as e:
        LOGGER.error(f"❌ Failed to load models: {str(e)}")
        raise

# =========================
# Long-text embedding utilities (Chunk & Pool)
# =========================
def _tokenize_to_ids(text: str, max_tokens: int) -> List[int]:
    """
    Tokenize to input_ids (includes special tokens). Returns list of token IDs.
    """
    if not text:
        return []
    tok = MODELS.roberta_tokenizer(
        text,
        return_attention_mask=False,
        return_token_type_ids=False,
        truncation=False,
        add_special_tokens=True,
    )
    return tok["input_ids"]

def _chunk_token_ids(
    token_ids: List[int],
    max_tokens: int = 512,
    stride: int = 64
) -> List[List[int]]:
    """
    Split token ids into overlapping chunks of size <= max_tokens.
    Keeps special tokens handling per chunk: ensures each chunk starts with BOS and ends with EOS if available.
    """
    if not token_ids:
        return []

    # Try to detect special tokens (RoBERTa uses <s>=0, </s>=2 usually)
    bos_id = MODELS.roberta_tokenizer.bos_token_id
    eos_id = MODELS.roberta_tokenizer.eos_token_id

    # Remove existing BOS/EOS to avoid duplication, we'll re-add per chunk
    core = [tid for tid in token_ids if tid not in (bos_id, eos_id) and tid is not None]

    chunks = []
    i = 0
    step = max_tokens - stride if max_tokens > stride else max_tokens
    if step <= 0:
        step = max_tokens

    while i < len(core):
        piece = core[i : i + (max_tokens - 2 if bos_id is not None and eos_id is not None else max_tokens)]
        if not piece:
            break
        if bos_id is not None:
            piece = [bos_id] + piece
        if eos_id is not None:
            piece = piece + [eos_id]
        chunks.append(piece)
        i += step
    return chunks

def _decode_ids_to_text(ids_chunk: List[int]) -> str:
    return MODELS.roberta_tokenizer.decode(ids_chunk, skip_special_tokens=True, clean_up_tokenization_spaces=True)

def validate_text_input(text: str, field_name: str = "text") -> str:
    """Validate and clean text input with comprehensive checks."""
    if text is None:
        LOGGER.warning(f"{field_name}: Received None, converting to empty string")
        return ""
    
    if not isinstance(text, str):
        LOGGER.warning(f"{field_name}: Expected string, got {type(text)}, converting")
        text = str(text)
    
    # Remove excessive whitespace
    text = " ".join(text.split())
    
    # Check for minimum meaningful content
    if len(text.strip()) < 10:
        LOGGER.warning(f"{field_name}: Text too short ({len(text)} chars), may affect quality")
    
    # Check for maximum reasonable length
    if len(text) > 10000:
        LOGGER.warning(f"{field_name}: Text very long ({len(text)} chars), may slow processing")
    
    return text

def _embed_text_chunks(
    text: str,
    max_tokens: int = 512,
    stride: int = 64,
    batch_size: int = 16,
    normalize: bool = True,
    weight_by_len: bool = True,
) -> np.ndarray:
    """
    Produce a single 768-d vector for long text by:
      1) tokenizing
      2) chunking with overlap
      3) encoding each chunk
      4) pooling (weighted/mean) and optional L2-normalization
    """
    # Validate input
    text = validate_text_input(text, "embedding_text")
    
    if not text or not text.strip():
        LOGGER.info("Empty text provided, returning zero vector")
        return np.zeros(EMBED_DIM, dtype=float)

    start_time = time.time()
    LOGGER.debug(f"Starting embedding for text of length {len(text)}")

    token_ids = _tokenize_to_ids(text, max_tokens=max_tokens)
    if not token_ids:
        LOGGER.warning("No tokens generated, returning zero vector")
        return np.zeros(EMBED_DIM, dtype=float)

    ids_chunks = _chunk_token_ids(token_ids, max_tokens=max_tokens, stride=stride)
    if not ids_chunks:
        LOGGER.warning("No chunks generated, returning zero vector")
        return np.zeros(EMBED_DIM, dtype=float)

    LOGGER.debug(f"Generated {len(ids_chunks)} chunks for embedding")

    # Decode chunks back to text for SentenceTransformer
    txt_chunks = [_decode_ids_to_text(c) for c in ids_chunks if c]

    try:
        embs = MODELS.embedder.encode(
            txt_chunks,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=False,  # we normalize after pooling
            show_progress_bar=False,
        )
        LOGGER.debug(f"Successfully encoded {len(embs)} chunks")
    except Exception as e:
        LOGGER.error(f"Embedding failed for chunks: {e}")
        return np.zeros(EMBED_DIM, dtype=float)

    if embs is None or len(embs) == 0:
        LOGGER.warning("No embeddings generated, returning zero vector")
        return np.zeros(EMBED_DIM, dtype=float)

    embs = np.asarray(embs, dtype=float)

    # Pooling
    if weight_by_len:
        # weight by (approx) chunk token length to reduce bias from very short chunks
        lengths = np.array([len(c) for c in ids_chunks], dtype=float)
        weights = lengths / (lengths.sum() if lengths.sum() > 0 else 1.0)
        pooled = (embs * weights[:, None]).sum(axis=0)
        LOGGER.debug(f"Applied weighted pooling with {len(weights)} weights")
    else:
        pooled = embs.mean(axis=0)
        LOGGER.debug("Applied mean pooling")

    # Normalize final vector (optional but usually helpful)
    if normalize:
        norm = np.linalg.norm(pooled)
        if norm > 0:
            pooled = pooled / norm
            LOGGER.debug(f"Applied L2 normalization (norm: {norm:.4f})")

    # Ensure correct dim
    if pooled.shape[0] != EMBED_DIM:
        LOGGER.error(f"Unexpected pooled dim {pooled.shape[0]} != {EMBED_DIM}")
        raise RuntimeError(f"Unexpected pooled dim {pooled.shape[0]} != {EMBED_DIM}")

    embed_time = time.time() - start_time
    LOGGER.debug(f"Embedding completed in {embed_time:.2f} seconds")
    
    return pooled

def health_check() -> Dict[str, Any]:
    """Perform comprehensive health check of all system components."""
    health_status = {
        "status": "healthy",
        "timestamp": time.time(),
        "components": {},
        "warnings": []
    }
    
    try:
        # Check models
        health_status["components"]["models"] = {
            "paraphraser": "loaded" if MODELS.paraphraser_model else "missing",
            "embedder": "loaded" if MODELS.embedder else "missing", 
            "classifier": "loaded" if MODELS.clf_model else "missing",
            "tokenizer": "loaded" if MODELS.roberta_tokenizer else "missing"
        }
        
        # Check embeddings
        test_text = "This is a test sentence for health check."
        test_embedding = _embed_text_chunks(test_text)
        health_status["components"]["embeddings"] = {
            "dimension": test_embedding.shape[0],
            "expected_dim": EMBED_DIM,
            "status": "working" if test_embedding.shape[0] == EMBED_DIM else "dimension_mismatch"
        }
        
        # Check prediction pipeline
        test_df = pd.DataFrame([[0] * len(MODELS.clf_features)], columns=MODELS.clf_features)
        test_prob = predict_success_probability(test_df)
        health_status["components"]["prediction"] = {
            "status": "working",
            "test_probability": test_prob
        }
        
        # Check memory usage
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        health_status["components"]["memory"] = {
            "rss_mb": memory_info.rss / 1024 / 1024,
            "vms_mb": memory_info.vms / 1024 / 1024
        }
        
        # Check GPU if available
        if torch.cuda.is_available():
            health_status["components"]["gpu"] = {
                "available": True,
                "device_count": torch.cuda.device_count(),
                "current_device": torch.cuda.current_device(),
                "memory_allocated_mb": torch.cuda.memory_allocated() / 1024 / 1024
            }
        else:
            health_status["components"]["gpu"] = {"available": False}
            
    except Exception as e:
        health_status["status"] = "unhealthy"
        health_status["error"] = str(e)
        LOGGER.error(f"Health check failed: {e}")
    
    return health_status

MODELS = load_models()

# Print model loading confirmation and health status
print("🚀 Models loaded successfully!")
print(f"📊 Using: {ROBERTA_MODEL_NAME}")
print(f"🔢 Embedding dimensions: {EMBED_DIM}")
print(f"📈 Total features expected: {len(MODELS.clf_features)}")

# Run health check
health = health_check()
if health["status"] == "healthy":
    print("✅ System health check passed")
else:
    print(f"⚠️ System health check failed: {health.get('error', 'Unknown error')}")

print("=" * 80)


# =========================
# Utilities
# =========================

def render_score_bar(probability: float, width: int = 20) -> str:
    probability = max(0.0, min(1.0, float(probability)))
    filled = int(round(probability * width))
    return "🟩" * filled + "⬜" * (width - filled)


def clean_paraphrase(text: str) -> str:
    return (text or "").strip().replace(" .", ".").replace("..", ".").replace("\n", " ")


def extract_keyphrases(text: str, top_n: int = 3) -> str:
    try:
        pairs = MODELS.kw_model.extract_keywords(
            text or "",
            keyphrase_ngram_range=(1, 3),
            stop_words="english",
            top_n=top_n,
        )
        return " / ".join([k for k, _ in pairs]) if pairs else "N/A"
    except Exception as e:
        print(f"⚠️ Keyphrase extraction failed: {e}")
        return "N/A"


# =========================
# Feature prep
# =========================

def build_base_dataframe(project_input: Dict[str, Any], clf_features: List[str]) -> pd.DataFrame:
    """Create a single-row DataFrame aligned with the classifier's expected columns."""
    row: Dict[str, Any] = {}

    # Numeric inputs
    for k in USER_NUMERIC_KEYS:
        if k in project_input:
            row[k] = project_input[k]

    # Categories (accept any that exist in clf_features)
    for k in USER_CATEGORY_KEYS:
        if k in clf_features:
            row[k] = int(bool(project_input.get(k, 0)))

    df = pd.DataFrame([row])

    # Add missing columns as zeros
    missing = [c for c in clf_features if c not in df.columns]
    if missing:
        df = pd.concat([df, pd.DataFrame([[0]*len(missing)], columns=missing)], axis=1)

    # Reorder to match classifier
    df = df[clf_features]
    return df


def fill_roberta_embeddings(df: pd.DataFrame, story: str, risks: str) -> None:
    """
    Fill embedding columns in-place for story and risks using RoBERTa with long-text support.
    Uses chunking+pooling so texts longer than 512 tokens are fully represented.
    """
    try:
        story_vec = _embed_text_chunks(story or "", max_tokens=512, stride=64, batch_size=16)
    except Exception as e:
        print(f"⚠️ Story embedding failed (fallback to single-encode): {e}")
        story_vec = np.asarray(MODELS.embedder.encode(story or ""), dtype=float)

    try:
        risk_vec = _embed_text_chunks(risks or "", max_tokens=512, stride=64, batch_size=16)
    except Exception as e:
        print(f"⚠️ Risks embedding failed (fallback to single-encode): {e}")
        risk_vec = np.asarray(MODELS.embedder.encode(risks or ""), dtype=float)

    # Safety: if dimension mismatch (shouldn't happen), fallback to zeros
    if story_vec.shape[0] != EMBED_DIM:
        story_vec = np.zeros(EMBED_DIM, dtype=float)
    if risk_vec.shape[0] != EMBED_DIM:
        risk_vec = np.zeros(EMBED_DIM, dtype=float)

    for i in range(EMBED_DIM):
        sc = f"{STORY_PREFIX}{i}"
        rc = f"{RISK_PREFIX}{i}"
        if sc in df.columns:
            df.at[df.index[0], sc] = story_vec[i]
        if rc in df.columns:
            df.at[df.index[0], rc] = risk_vec[i]



# =========================
# Prediction
# =========================

def predict_success_probability(df_aligned: pd.DataFrame) -> float:
    """Predict success probability with performance monitoring."""
    start_time = time.time()
    
    try:
        # Validate input
        if df_aligned.empty:
            LOGGER.warning("Empty DataFrame provided for prediction")
            return 0.0
        
        if len(df_aligned) != 1:
            LOGGER.warning(f"Expected single row, got {len(df_aligned)} rows")
        
        # Check for missing values
        missing_cols = df_aligned.columns[df_aligned.isnull().any()].tolist()
        if missing_cols:
            LOGGER.warning(f"Missing values in columns: {missing_cols}")
            df_aligned = df_aligned.fillna(0)
        
        proba = MODELS.clf_model.predict_proba(df_aligned)[0, 1]
        prediction_time = time.time() - start_time
        
        LOGGER.debug(f"Prediction completed in {prediction_time:.3f} seconds")
        LOGGER.info(f"Success probability: {proba:.2%}")
        
        return float(proba)
        
    except Exception as e:
        LOGGER.error(f"Prediction failed: {e}")
        return 0.0


# =========================
# Paraphrasing & Scoring
# =========================

def generate_paraphrases(text: str, params: dict, num_return: int = 4) -> List[str]:
    input_text = f"paraphrase: {text} </s>"
    inputs = MODELS.paraphraser_tokenizer.encode(
        input_text, return_tensors="pt", max_length=512, truncation=True
    ).to(MODELS.device)
    try:
        outputs = MODELS.paraphraser_model.generate(
            inputs,
            num_return_sequences=num_return,
            do_sample=True,
            num_beams=max(1, min(num_return, 4)),
            **params,
        )
        return [clean_paraphrase(MODELS.paraphraser_tokenizer.decode(o, skip_special_tokens=True)) for o in outputs]
    except Exception as e:
        print(f"Error generating paraphrases: {e}")
        return []


def multi_sentence_coherence_score(orig: str, para: str) -> float:
    s1 = sent_tokenize(orig or "")
    s2 = sent_tokenize(para or "")
    if not s1 or not s2:
        return 0.0
    n = min(len(s1), len(s2))
    sims = []
    for i in range(n):
        v1 = MODELS.embedder.encode(s1[i])
        v2 = MODELS.embedder.encode(s2[i])
        sims.append(float(cosine_similarity([v1], [v2])[0][0]))
    return float(np.mean(sims)) if sims else 0.0


def get_quick_suggestions(
    story: str,
    risks: str,
    df_template: pd.DataFrame,
    num_return: int = 1,
    coherence_threshold: float = 0.60,
) -> List[Dict[str, Any]]:
    presets = [
        {"top_k": 40, "top_p": 0.92, "temperature": 1.0, "max_length": 512},
        {"top_k": 80, "top_p": 0.95, "temperature": 1.2, "max_length": 512},
        {"top_k": 60, "top_p": 0.88, "temperature": 0.9, "max_length": 512},
    ]
    results: List[Dict[str, Any]] = []

    for preset in presets:
        for txt in generate_paraphrases(story, preset, num_return=num_return):
            df = df_template.copy()
            fill_roberta_embeddings(df, txt, risks)
            prob = predict_success_probability(df)
            coh = multi_sentence_coherence_score(story, txt)
            results.append(
                {
                    "text": txt,
                    "prob": prob,
                    "coherence": coh,
                    "is_strong": coh >= coherence_threshold,
                    "params": preset,
                    "theme": extract_keyphrases(txt),
                }
            )

    results.sort(key=lambda r: r["prob"], reverse=True)
    return results


def explain_params(params: Dict[str, float]) -> List[str]:
    out = []
    tk = int(params.get("top_k", 50))
    tp = float(params.get("top_p", 0.9))
    temp = float(params.get("temperature", 1.0))

    out.append(
        "Low top-k => conservative wording; high top-k => more novelty (risking drift)."
        if tk < 50
        else ("Very high top-k => creative but unstable." if tk > 100 else "Medium top-k balances diversity & coherence.")
    )
    out.append(
        "Low top-p narrows choice distribution (stable)." if tp <= 0.88 else (
            "High top-p allows flexible sampling (creative)." if tp >= 0.95 else "Medium top-p is balanced."
        )
    )
    out.append(
        "Low temperature => predictable phrasing; high => surprising phrasing." if not (0.9 <= temp <= 1.3) else
        "Medium temperature balances consistency & diversity."
    )
    return out


def optimize_paraphrase_optuna(
    original_story: str,
    risks: str,
    df_template: pd.DataFrame,
    coherence_threshold: float = 0.60,
    n_trials: int = 10,
) -> Tuple[str, float, dict, List[str]]:
    def objective(trial: optuna.trial.Trial) -> float:
        # Show progress
        current_trial = trial.number + 1
        progress = (current_trial / n_trials) * 100
        print(f"\r🔄 Trial {current_trial}/{n_trials} ({progress:.1f}%) - Testing parameters...", end="", flush=True)
        
        params = {
            "top_k": trial.suggest_int("top_k", 20, 150),
            "top_p": trial.suggest_float("top_p", 0.85, 0.98),
            "temperature": trial.suggest_float("temperature", 0.8, 1.5),
            "max_length": 512,
        }
        cands = generate_paraphrases(original_story, params, num_return=2)
        best = 0.0
        best_txt = ""
        for txt in cands:
            coh = multi_sentence_coherence_score(original_story, txt)
            if coh < coherence_threshold:
                continue
            df = df_template.copy()
            fill_roberta_embeddings(df, txt, risks)
            prob = predict_success_probability(df)
            if prob > best:
                best = prob
                best_txt = txt
        
        # Show current best result
        print(f"\r🔄 Trial {current_trial}/{n_trials} ({progress:.1f}%) - Best so far: {best:.2%}", end="", flush=True)
        
        trial.set_user_attr("paraphrase", best_txt)
        return best

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=1337, n_startup_trials=5),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1),
    )
    study.optimize(objective, n_trials=n_trials)
    
    # Clear the progress line and show completion
    print(f"\r✅ Optimization completed! ({n_trials}/{n_trials} trials - 100%)")

    best_txt = study.best_trial.user_attrs.get("paraphrase", "")
    best_prob = float(study.best_value)
    best_params = study.best_params
    return best_txt, best_prob, best_params, explain_params(best_params)


def run_optuna_async(original_story: str, risks: str, df_template: pd.DataFrame) -> None:
    def task():
        print("\n🔍 Optuna searching for improved STORY paraphrase... This may take a few minutes.")
        
        # Show original story and risks analysis first
        print("\n📖 ORIGINAL STORY:")
        print(textwrap.fill(original_story, width=100))
        
        print("\n⚠️ ORIGINAL RISKS:")
        print(textwrap.fill(risks, width=100))
        
        # Get original probability
        original_prob = predict_success_probability(df_template)
        print(f"\n✅ Original Success Probability: {original_prob:.2%}")
        print(f"📈 Original Visual Score: {render_score_bar(original_prob)} ({original_prob:.2%})")
        
        # Run optimization
        best_txt, best_prob, best_params, explanations = optimize_paraphrase_optuna(
            original_story, risks, df_template.copy()
        )
        
        # Show optimized result in the same format as quick suggestions
        print(f"\n🔹 OPTUNA OPTIMIZED SUGGESTION:")
        print(f"🧠 Theme: {extract_keyphrases(best_txt, top_n=3)}")
        print(textwrap.fill(best_txt, width=100))
        print(f"✅ Success Probability: {best_prob:.2%}")
        print(f"📈 Visual Score: {render_score_bar(best_prob)} ({best_prob:.2%})")
        print(f"🧪 Params: top_k={best_params.get('top_k', 'N/A')}, top_p={best_params.get('top_p', 'N/A')}, temperature={best_params.get('temperature', 'N/A')}")
        
        # Calculate coherence score
        coherence = multi_sentence_coherence_score(original_story, best_txt)
        is_strong = coherence >= 0.60
        print(f"🧠 Coherence Score: {coherence:.2f} {'✅ Strong' if is_strong else '⚠️ Weak'}")
        
        print("\n🧠 Optimization Explanations:")
        for e in explanations:
            print("•", e)

    threading.Thread(target=task, daemon=True).start()


# =========================
# Example Project Input (edit as needed)
# =========================
project_input: Dict[str, Any] = {
    "goal": 131421,
    "rewardscount": 6,
    "projectFAQsCount": 8,
    "project_length_days": 30,
    "preparation_days": 5,
    "category_Web_Development": 1,
    "story": (
        "Innovative Device is an ambitious project aimed at revolutionizing the Gadgets industry. "
        "While the concept behind Innovative Device was met with enthusiasm, we faced significant "
        "challenges in securing the necessary funding and resources. The journey has been filled with "
        "obstacles, ranging from supply chain issues to unexpected technical difficulties. Despite our "
        "best efforts, these setbacks delayed our timeline and affected our ability to bring Innovative "
        "Device to market at the scale we envisioned. Our hope was to introduce Innovative Device to the "
        "Gadgets market and make a meaningful impact. Although this campaign did not reach its goal, the "
        "feedback and support from our community have been invaluable. We will continue exploring "
        "alternative funding options and look forward to relaunching Innovative Device in the future with "
        "a stronger foundation."
    ),
    "risks": (
        "Launching Innovative Device in the field of Gadgets comes with its own set of challenges. One of "
        "the biggest concerns is ensuring that Innovative Device integrates seamlessly with existing "
        "Gadgets solutions. Compatibility issues may arise, requiring extensive testing and refinements "
        "before mass production. Additionally, sourcing high-quality components for Gadgets-specific "
        "hardware can be time-consuming and costly. Security is another major factor. Innovative Device "
        "will need to maintain strict data protection standards to ensure privacy and prevent cyber threats. "
        "The regulatory landscape for Gadgets is evolving, and ensuring compliance with industry standards is "
        "crucial for Innovative Device to be legally distributed in multiple markets. Our team is prepared to "
        "address these risks by implementing a robust quality control process, working closely with industry "
        "experts, and securing partnerships with reliable manufacturers to ensure a smooth launch."
    ),
}


# =========================
# Main
# =========================
if __name__ == "__main__":
    # 1) Build classifier-aligned DataFrame and fill embeddings from current texts
    df_base = build_base_dataframe(project_input, MODELS.clf_features)
    fill_roberta_embeddings(df_base, project_input.get("story", ""), project_input.get("risks", ""))

    # 2) Evaluate base probability
    combined_prob = predict_success_probability(df_base)

    print("\n🎯 ORIGINAL STORY:")
    print(textwrap.fill(project_input.get("story", ""), width=100))
    print(f"✅ Success Probability (combined): {combined_prob:.2%}")
    print(f"📈 Visual Score: {render_score_bar(combined_prob)} ({combined_prob:.2%})")

    # 3) Quick suggestions for STORY (change num_return to >1 for more)
    print("\n⚡ Fast Paraphrase Suggestions for STORY:")
    suggestions = get_quick_suggestions(
        story=project_input.get("story", ""),
        risks=project_input.get("risks", ""),
        df_template=df_base,
        num_return=1,
    )
    for i, s in enumerate(suggestions[:5]):
        print(f"\n🔹 Suggestion #{i+1}")
        print(f"🧠 Theme: {s['theme']}")
        print(textwrap.fill(s["text"], width=100))
        print(f"✅ Success Probability: {s['prob']:.2%}")
        print(f"📈 Visual Score: {render_score_bar(s['prob'])} ({s['prob']:.2%})")
        p = s["params"]
        print(f"🧪 Params: top_k={p['top_k']}, top_p={p['top_p']}, temperature={p['temperature']}")
        print(f"🧠 Coherence Score: {s['coherence']:.2f} {'✅ Strong' if s['is_strong'] else '⚠️ Weak'}")

    # 4) Ask to run full Optuna search
    try:
        choice = input("\n🔧 Run full optimization with Optuna for STORY? (y/n): ").strip().lower()
    except EOFError:
        choice = "n"
    if choice.startswith("y"):
        ##run_optuna_async(project_input.get("story", ""), project_input.get("risks", ""), df_base.copy())
        best_txt, best_prob, best_params, explanations = optimize_paraphrase_optuna(
            project_input.get("story", ""),
            project_input.get("risks", ""),
            df_base.copy()
        )

        print("\n🔹 OPTUNA OPTIMIZED SUGGESTION:")
        print(f"🧠 Theme: {extract_keyphrases(best_txt, top_n=3)}")
        print(textwrap.fill(best_txt, width=100))
        print(f"✅ Success Probability: {best_prob:.2%}")
        print(f"📈 Visual Score: {render_score_bar(best_prob)} ({best_prob:.2%})")
        print(f"🧪 Params: {best_params}")
        print(f"🧠 Optimization Explanations: {explanations}")



        
        # Keep the program alive long enough for background thread to print some results
        try:
            import time
            time.sleep(2)
        except KeyboardInterrupt:
            pass
    else:
        print("\n✅ Finished with quick suggestions only. Good luck!")
