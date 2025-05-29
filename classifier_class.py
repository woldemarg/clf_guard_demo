import re
import warnings
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords as nltk_stopwords
from nltk.stem.snowball import SnowballStemmer
from stopwordsiso import stopwords
from langdetect import detect, DetectorFactory

import sklearn
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score, classification_report

from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from tree_stem import stem_word

# %%


# Set seed for consistent language detection
DetectorFactory.seed = 42


@dataclass
class ClassifierMetrics:
    """Container for cross-validation metrics"""
    f1_scores: List[float]
    mean_f1: float
    std_f1: float
    classification_reports: List[str]

    def __repr__(self) -> str:
        return f"ClassifierMetrics(mean_f1={self.mean_f1:.4f}±{self.std_f1:.4f})"


class LanguageProcessor:
    """Language-specific text processing utilities"""

    # Language configuration
    LANG_CONFIG = {
        'uk': {
            'stemmer': None,  # Use tree_stem
            'stopwords_source': 'iso',
            'nltk_lang': None
        },
        'ru': {
            'stemmer': 'russian',
            'stopwords_source': 'nltk',
            'nltk_lang': 'russian'
        },
        'en': {
            'stemmer': 'english',
            'stopwords_source': 'nltk',
            'nltk_lang': 'english'
        }
    }

    # Text cleaning patterns
    RE_EMOJI = re.compile(r'[\U00010000-\U0010ffff]', flags=re.UNICODE)
    RE_SYMBOLS = re.compile(r'[^\w\s]', flags=re.UNICODE)
    RE_DIGITS = re.compile(r'\d+')

    def __init__(self, language: str):
        """Initialize language processor"""
        if language not in self.LANG_CONFIG:
            raise ValueError(f"Unsupported language: {language}")

        self.language = language
        self.config = self.LANG_CONFIG[language]

        # Initialize stemmer
        if self.config['stemmer']:
            self.stemmer = SnowballStemmer(self.config['stemmer'])
        else:
            self.stemmer = None

        # Initialize stopwords
        self._init_stopwords()

    def _init_stopwords(self):
        """Initialize stopwords for the language"""
        try:
            if self.config['stopwords_source'] == 'iso':
                self.stop_words = set(stopwords(self.language))
            else:  # nltk
                self.stop_words = set(
                    nltk_stopwords.words(self.config['nltk_lang']))
        except Exception as e:
            warnings.warn(f"Failed to load stopwords for {self.language}: {e}")
            self.stop_words = set()

    def clean_text(self, text: str) -> str:
        """Clean text by removing emojis, digits, and symbols"""
        if not isinstance(text, str):
            return ""

        text = self.RE_EMOJI.sub(' ', text)
        text = self.RE_DIGITS.sub(' ', text)
        text = self.RE_SYMBOLS.sub(' ', text)
        return text.strip().lower()

    def stem_token(self, token: str) -> str:
        """Stem a token using language-specific stemmer"""
        if self.language == 'uk':
            return stem_word(token)
        elif self.stemmer:
            return self.stemmer.stem(token)
        return token

    def tokenize_and_stem(self, text: str) -> List[str]:
        """Tokenize and stem text"""
        try:
            text = self.clean_text(text)
            tokens = word_tokenize(text)
            return " ".join([
                self.stem_token(token)
                for token in tokens
                if token not in self.stop_words and len(token) > 1
            ])
        except Exception:
            # Fallback to simple splitting if tokenization fails
            words = text.split()
            return " ".join([
                word for word in words
                if word not in self.stop_words and len(word) > 1
            ])


class MultilingualTextClassifier(BaseEstimator, ClassifierMixin):
    """
    Multilingual text classifier supporting Ukrainian, Russian, and English.

    This classifier trains separate models for each language and automatically
    detects the language of input text for prediction.
    """

    SUPPORTED_LANGUAGES = ['uk', 'ru', 'en']
    # DEFAULT_LANGUAGE = 'en'  # Fallback language
    DEFAULT_LANGUAGE = 'ru'  # Fallback language

    def __init__(
        self,
        max_features: int = 10000,
        ngram_range: Tuple[int, int] = (1, 3),
        C: float = 1.0,
        cv_folds: int = 5,
        calibration_cv: int = 10,
        random_state: int = 42,
        n_jobs: int = -1
    ):
        """
        Initialize the multilingual classifier.

        Parameters:
        -----------
        max_features : int
            Maximum number of features for TF-IDF vectorizer
        ngram_range : tuple
            N-gram range for TF-IDF vectorizer
        C : float
            Regularization parameter for LogisticRegression
        cv_folds : int
            Number of cross-validation folds
        calibration_cv : int
            Number of CV folds for probability calibration
        random_state : int
            Random state for reproducibility
        n_jobs : int
            Number of parallel jobs
        """
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.C = C
        self.cv_folds = cv_folds
        self.calibration_cv = calibration_cv
        self.random_state = random_state
        self.n_jobs = n_jobs

        # Initialize containers
        self.classifiers_ = {}
        self.label_encoders_ = {}
        self.processors_ = {}
        self.metrics_ = {}
        self.is_fitted_ = False

    def _create_pipeline(self, language: str) -> Pipeline:
        """Create a pipeline for a specific language"""
        processor = LanguageProcessor(language)
        self.processors_[language] = processor

        # TF-IDF Vectorizer with language-specific tokenizer
        vectorizer = TfidfVectorizer(
            # analyzer='word',
            analyzer='char_wb',
            # token_pattern=None,
            # tokenizer=processor.tokenize_and_stem,
            preprocessor=processor.tokenize_and_stem,
            ngram_range=self.ngram_range,
            max_features=self.max_features,
            sublinear_tf=True,
        )

        # Base classifier
        base_clf = LogisticRegression(
            C=self.C,
            class_weight='balanced',
            max_iter=1000,
            n_jobs=self.n_jobs,
            random_state=self.random_state
        )
        base_clf.set_fit_request(sample_weight=True)

        # Calibrated classifier with adaptive CV
        calibrated = CalibratedClassifierCV(
            estimator=base_clf,
            method='sigmoid',
            # Use min 3 folds to avoid small sample issues
            cv=min(self.calibration_cv, 3),
            n_jobs=self.n_jobs
        )
        calibrated.set_fit_request(sample_weight=True)

        # Multi-label classifier
        multilabel_clf = OneVsRestClassifier(calibrated)

        # Complete pipeline
        pipeline = Pipeline([
            ('tfidf', vectorizer),
            ('clf', multilabel_clf)
        ])

        return pipeline

    def _prepare_data(self, df: pd.DataFrame, language: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prepare data for a specific language"""
        lang_data = df[df['lang'] == language].copy()

        if len(lang_data) == 0:
            raise ValueError(f"No data found for language: {language}")

        # Check if we have enough samples for cross-validation
        # At least 10 samples or cv_folds
        min_samples_needed = max(self.cv_folds, 10)
        if len(lang_data) < min_samples_needed:
            raise ValueError(
                f"Insufficient data for {language}: {len(lang_data)} samples (need at least {min_samples_needed})")

        # Features
        X = lang_data['text'].values

        # Labels
        mlb = MultiLabelBinarizer()
        y = mlb.fit_transform(lang_data['predicted_label_upd'])

        # Check label distribution for cross-validation
        min_label_count = np.min(np.sum(y, axis=0))
        if min_label_count < self.cv_folds:
            raise ValueError(
                f"Insufficient samples for some labels in {language}: min {min_label_count} samples per label (need at least {self.cv_folds})")

        self.label_encoders_[language] = mlb

        # Sample weights (use max score across all score columns)
        score_cols = [
            col for col in lang_data.columns if col.startswith('score_')]
        if score_cols:
            sample_weights = lang_data[score_cols].max(axis=1).values
        else:
            sample_weights = np.ones(len(lang_data))

        return X, y, sample_weights

    def _perform_cross_validation(
        self,
        pipeline: Pipeline,
        X: np.ndarray,
        y: np.ndarray,
        sample_weights: np.ndarray,
        language: str,
        cv_folds: Optional[int] = None
    ) -> ClassifierMetrics:
        """Perform cross-validation and collect metrics"""
        f1_scores = []
        classification_reports = []

        # Use adaptive CV folds
        effective_folds = cv_folds or self.cv_folds

        # Use stratified k-fold for multilabel data
        mskf = MultilabelStratifiedKFold(
            n_splits=effective_folds,
            shuffle=True,
            random_state=self.random_state
        )

        mlb = self.label_encoders_[language]

        for fold, (train_idx, val_idx) in enumerate(mskf.split(X, y)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            w_train = sample_weights[train_idx]

            # Fit pipeline
            pipeline.fit(X_train, y_train, sample_weight=w_train)

            # Predict
            y_pred = pipeline.predict(X_val)

            # Calculate F1 score
            f1 = f1_score(y_val, y_pred, average='weighted', zero_division=0)
            f1_scores.append(f1)

            # Generate classification report
            report = classification_report(
                y_val, y_pred,
                target_names=mlb.classes_,
                zero_division=0,
                output_dict=False
            )
            classification_reports.append(f"Fold {fold + 1}:\n{report}")

        return ClassifierMetrics(
            f1_scores=f1_scores,
            mean_f1=np.mean(f1_scores),
            std_f1=np.std(f1_scores),
            classification_reports=classification_reports
        )

    def fit(self, df: pd.DataFrame, verbose: bool = True) -> 'MultilingualTextClassifier':
        """
        Fit the multilingual classifier.

        Parameters:
        -----------
        df : pd.DataFrame
            Training dataframe with columns: 'text', 'lang', 'predicted_label_upd', 'score_*'
        verbose : bool
            Whether to print training progress

        Returns:
        --------
        self : MultilingualTextClassifier
            Fitted classifier
        """
        # Download required NLTK data
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
        except Exception as e:
            warnings.warn(f"Failed to download NLTK data: {e}")

        # Enable sklearn metadata routing
        sklearn.set_config(enable_metadata_routing=True)

        # Check required columns
        required_cols = ['text', 'lang', 'predicted_label_upd']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Filter for supported languages
        available_languages = df['lang'].unique()
        supported_available = [
            lang for lang in available_languages if lang in self.SUPPORTED_LANGUAGES]

        if not supported_available:
            raise ValueError(
                f"No supported languages found. Available: {available_languages}")

        if verbose:
            print(f"Training classifiers for languages: {supported_available}")

        # Train classifier for each language
        for language in supported_available:
            if verbose:
                print(f"\n--- Training {language.upper()} classifier ---")

            try:
                # Prepare data
                X, y, sample_weights = self._prepare_data(df, language)

                if verbose:
                    print(f"Data shape: {X.shape}, Labels: {y.shape[1]}")

                # Create pipeline
                pipeline = self._create_pipeline(language)

                # Cross-validation
                if verbose:
                    print("Performing cross-validation...")

                # Adaptive CV folds based on data size and label distribution
                effective_cv_folds = min(
                    self.cv_folds,
                    len(X) // 10,  # At least 10 samples per fold
                    np.min(np.sum(y, axis=0))  # Min samples per label
                )

                if effective_cv_folds < 2:
                    if verbose:
                        print(
                            f"Skipping CV for {language} due to insufficient data")
                    # Create dummy metrics for consistency
                    self.metrics_[language] = ClassifierMetrics(
                        f1_scores=[0.0],
                        mean_f1=0.0,
                        std_f1=0.0,
                        classification_reports=[
                            "Insufficient data for cross-validation"]
                    )
                else:
                    metrics = self._perform_cross_validation(
                        pipeline, X, y, sample_weights, language, effective_cv_folds
                    )
                    self.metrics_[language] = metrics

                    if verbose:
                        print(
                            f"CV F1 Score: {metrics.mean_f1:.4f} ± {metrics.std_f1:.4f}")

                # Final fit on all data
                pipeline.fit(X, y, sample_weight=sample_weights)
                self.classifiers_[language] = pipeline

            except Exception as e:
                warnings.warn(
                    f"Failed to train classifier for {language}: {e}")
                continue

        if not self.classifiers_:
            raise RuntimeError("Failed to train any classifiers")

        self.is_fitted_ = True

        if verbose:
            print(
                f"\n✓ Training completed for {len(self.classifiers_)} languages")

        return self

    def _detect_language(self, text: str) -> str:
        """Detect language of input text"""
        if not text or not isinstance(text, str):
            return self.DEFAULT_LANGUAGE

        try:
            detected = detect(text)
            # Map some common language codes
            lang_mapping = {'uk': 'uk', 'ru': 'ru', 'en': 'en'}
            detected_lang = lang_mapping.get(detected, detected)

            if detected_lang in self.classifiers_:
                return detected_lang
        except Exception:
            pass

        return self.DEFAULT_LANGUAGE

    def predict_text(self, text: str, return_probabilities: bool = True) -> Dict[str, Union[List[str], Dict[str, float]]]:
        """
        Predict labels for input text.

        Parameters:
        -----------
        text : str
            Input text to classify
        return_probabilities : bool
            Whether to return prediction probabilities

        Returns:
        --------
        dict : Dictionary containing predictions and metadata
        """
        if not self.is_fitted_:
            raise ValueError("Classifier is not fitted. Call fit() first.")

        # Default fallback response
        fallback_response = {
            'predicted_labels': [],
            'probabilities': {},
            'detected_language': 'unknown',
            'confidence': 0.0,
            'error': None
        }

        if not text or not isinstance(text, str):
            fallback_response['error'] = 'Invalid input text'
            return fallback_response

        try:
            # Detect language
            detected_lang = self._detect_language(text)

            # Use fallback language if detected language is not available
            if detected_lang not in self.classifiers_:
                detected_lang = self.DEFAULT_LANGUAGE
                if detected_lang not in self.classifiers_:
                    # Use any available classifier as last resort
                    detected_lang = list(self.classifiers_.keys())[0]

            # Get classifier and label encoder
            classifier = self.classifiers_[detected_lang]
            label_encoder = self.label_encoders_[detected_lang]

            # Predict
            if return_probabilities:
                probs = classifier.predict_proba([text])[0]
                probabilities = dict(
                    zip(label_encoder.classes_, probs.round(3)))

            #     # Get predicted labels (threshold at 0.5)
            #     predictions = classifier.predict([text])[0]
            #     predicted_labels = label_encoder.inverse_transform([predictions])[
            #         0]
            # else:
            #     predictions = classifier.predict([text])[0]
            #     predicted_labels = label_encoder.inverse_transform([predictions])[
            #         0]
            #     probabilities = {}

            return {
                # 'predicted_labels': list(predicted_labels),
                'probabilities': probabilities,
                'detected_language': detected_lang,
                # 'confidence': max(probabilities.values()) if probabilities else 0.0,
                'error': None
            }

        except Exception as e:
            fallback_response['error'] = str(e)
            return fallback_response

    def get_metrics(self, language: Optional[str] = None) -> Union[Dict[str, ClassifierMetrics], ClassifierMetrics]:
        """Get cross-validation metrics"""
        if not self.is_fitted_:
            raise ValueError("Classifier is not fitted. Call fit() first.")

        if language:
            if language not in self.metrics_:
                raise ValueError(
                    f"No metrics available for language: {language}")
            return self.metrics_[language]

        return self.metrics_

    def get_supported_languages(self) -> List[str]:
        """Get list of languages with trained classifiers"""
        return list(self.classifiers_.keys()) if self.is_fitted_ else []

    # def __repr__(self) -> str:
    #     if self.is_fitted_:
    #         langs = list(self.classifiers_.keys())
    #         return f"MultilingualTextClassifier(fitted_languages={langs})"
    #     return "MultilingualTextClassifier(not_fitted)"

