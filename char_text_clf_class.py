from typing import Dict
import re
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV


# %%

class CharTextClassifier:
    def __init__(self, max_features=10000):

        # Compile regexes once
        self.RE_EMOJI = re.compile(
            r'[\U00010000-\U0010ffff]', flags=re.UNICODE)
        self.RE_SYMBOLS = re.compile(r'[^\w\s]', flags=re.UNICODE)
        self.RE_DIGITS = re.compile(r'\d+', flags=re.UNICODE)

        self.vectorizer = TfidfVectorizer(
            decode_error='replace',
            lowercase=True,
            strip_accents='unicode',
            analyzer='char_wb',
            ngram_range=(3, 5),
            max_features=max_features,
            sublinear_tf=True,
            preprocessor=self.clean_text
        )

        base_clf = LogisticRegression(
            C=0.9,
            class_weight='balanced',
            max_iter=1000,
            n_jobs=-1
        )

        self.clf = CalibratedClassifierCV(
            estimator=base_clf,
            method='sigmoid',
            cv=10,
            n_jobs=-1
        )

        self.pipe = Pipeline([
            ('tfidf', self.vectorizer),
            ('clf', self.clf)
        ])

        self.label_encoder = LabelEncoder()

    def clean_text(self, text: str) -> str:
        text = self.RE_EMOJI.sub(' ', text)
        text = self.RE_DIGITS.sub(' ', text)
        text = self.RE_SYMBOLS.sub(' ', text)
        return text.strip().lower()

    def fit(self,
            df: pd.DataFrame,
            text_col: str,
            label_col: str,
            weight_col: str = None):

        y = self.label_encoder.fit_transform(df[label_col])
        X = df[text_col].values
        weights = df[weight_col].values if weight_col else None
        self.pipe.fit(X, y, clf__sample_weight=weights)

    def predict(self, text: str) -> Dict[str, float]:
        probs = self.pipe.predict_proba([text])[0]
        return dict(zip(self.label_encoder.classes_, np.round(probs, 4)))
