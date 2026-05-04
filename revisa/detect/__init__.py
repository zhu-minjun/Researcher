from copy import deepcopy
from typing import Optional, List, Dict

import numpy as np
import torch
from revisa.detect.detect_data import DATA_
from revisa.detect.fast_detect_gpt import get_sampling_discrepancy_analytic
from revisa.detect.model import load_tokenizer, load_model
from tqdm import tqdm


class FastDetectGPT:
    def __init__(
            self,
            model_name: str,
            device: str = "cuda",
            cache_dir: str = "../cache",
            reference_model_name: Optional[str] = None,
            dataset: str = "xsum"
    ):
        """
        Initialize FastDetectGPT detector

        Args:
            model_name: Name or path of the scoring model
            device: Device to run the model on ('cuda' or 'cpu')
            cache_dir: Directory for caching models
            reference_model_name: Optional different model for reference (defaults to scoring model)
            dataset: Dataset name for tokenizer configuration
        """
        self.device = device

        # Load scoring model and tokenizer
        self.scoring_tokenizer = load_tokenizer(model_name, dataset, cache_dir)
        self.scoring_model = load_model(model_name, device, cache_dir)
        self.scoring_model.eval()

        # Load reference model if different
        if reference_model_name and reference_model_name != model_name:
            self.reference_tokenizer = load_tokenizer(reference_model_name, dataset, cache_dir)
            self.reference_model = load_model(reference_model_name, device, cache_dir)
            self.reference_model.eval()
        else:
            self.reference_tokenizer = self.scoring_tokenizer
            self.reference_model = self.scoring_model

        # Load probability estimator reference data
        self._load_reference_data()

    def _load_reference_data(self):
        """Load reference data for probability estimation"""
        res = deepcopy(DATA_)
        self.real_crits = res['predictions']['real']
        self.fake_crits = res['predictions']['samples']
        print(f'Loaded reference data: {len(self.real_crits) * 2} samples')

    def _estimate_probability(self, criterion: float) -> float:
        """Estimate probability of text being machine-generated based on criterion"""
        offset = np.sort(np.abs(np.array(self.real_crits + self.fake_crits) - criterion))[100]
        cnt_real = np.sum((np.array(self.real_crits) > criterion - offset) &
                          (np.array(self.real_crits) < criterion + offset))
        cnt_fake = np.sum((np.array(self.fake_crits) > criterion - offset) &
                          (np.array(self.fake_crits) < criterion + offset))
        return cnt_fake / (cnt_real + cnt_fake)

    def detect(self, text: str, max_length: int = 2048) -> Dict[str, float]:
        """
        Detect if text is machine-generated

        Args:
            text: Input text to analyze
            max_length: Maximum token length to process

        Returns:
            Dictionary containing:
                - criterion: Raw detection criterion score
                - probability: Estimated probability of being machine-generated
        """
        # Tokenize input text
        tokenized = self.scoring_tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            return_token_type_ids=False,
            max_length=max_length,
            truncation=True
        ).to(self.device)

        labels = tokenized.input_ids[:, 1:]

        with torch.no_grad():
            # Get logits from scoring model
            logits_score = self.scoring_model(**tokenized).logits[:, :-1]

            # Get logits from reference model if different
            if self.reference_model is self.scoring_model:
                logits_ref = logits_score
            else:
                tokenized = self.reference_tokenizer(
                    text,
                    return_tensors="pt",
                    padding=True,
                    return_token_type_ids=False,
                    max_length=max_length,
                    truncation=True
                ).to(self.device)
                assert torch.all(tokenized.input_ids[:, 1:] == labels), "Tokenizer mismatch"
                logits_ref = self.reference_model(**tokenized).logits[:, :-1]

            # Calculate criterion using the imported function
            criterion = get_sampling_discrepancy_analytic(logits_ref, logits_score, labels)

            # Estimate probability
            probability = self._estimate_probability(criterion)

            return {
                "criterion": criterion,
                "probability": probability
            }

    def detect_batch(self, texts: List[str], max_length: int = 2048) -> List[Dict[str, float]]:
        """
        Detect machine-generated text for a batch of inputs

        Args:
            texts: List of input texts to analyze
            max_length: Maximum token length to process

        Returns:
            List of detection results for each input text
        """
        results = []
        for text in tqdm(texts, desc="Analyzing texts"):
            results.append(self.detect(text, max_length))
        return results