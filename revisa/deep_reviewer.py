import re
import requests
import json
from copy import deepcopy

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def _auto_device():
    """Pick the best available device: CUDA > MPS (Apple Silicon) > CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _auto_dtype(device):
    """Pick a safe dtype for the detected device.

    - CUDA with compute capability >= 8.0 (Ampere/Hopper): bfloat16
    - CUDA older than Ampere (T4, V100, P100): float16
    - MPS: float16 (bf16 support is inconsistent)
    - CPU: float32 (bf16/fp16 are slow or unsupported on most CPUs)
    """
    if device == "cuda":
        major, _ = torch.cuda.get_device_capability()
        return torch.bfloat16 if major >= 8 else torch.float16
    if device == "mps":
        return torch.float16
    return torch.float32


def _auto_attn_implementation(device):
    """Pick an attention kernel that works on the device.

    SDPA is built into torch >= 2.1 and runs everywhere (CUDA / MPS / CPU).
    flash_attention_2 is faster on Ampere+ but is opt-in (requires the
    flash-attn package), so we never auto-select it.
    """
    return "sdpa"

# Helper Functions for Best Mode
# Adapted from the provided Flask app (main.py)

def extract_questions_from_content(content: str) -> list[str]:
    """Extract questions from the questions block (e.g., \boxed_questions{...})."""
    questions = []
    # Attempt to find the content within oxed_questions{}
    # This regex is a common way to find such blocks if they exist.
    # If the questions are simply listed after a header like "❓ Questions", 
    # this part might need adjustment based on actual LLM output format.
    
    # First, try to find a specific block like oxed_questions{}
    boxed_questions_match = re.search(r'\boxed_questions\{(.*?)\}', content, re.DOTALL)
    lines = [] # Initialize lines to an empty list
    if boxed_questions_match:
        questions_block = boxed_questions_match.group(1)
        # Assuming questions within the block are separated by newlines
        lines = [line.strip() for line in questions_block.split('\n') if line.strip()]
    else:
        # Fallback or alternative: if questions are under a "## Questions" or "❓ Questions" header
        # This part might need refinement based on the actual output format from the LLM.
        # For now, let's assume questions are separated by newlines after such a header.
        if "❓ Questions" in content: # Or a similar marker
            potential_questions_section = content.split("❓ Questions", 1)[-1]
            lines = [line.strip() for line in potential_questions_section.split('\n') if line.strip()]
        elif "## Questions" in content: # Handle markdown style headers
            potential_questions_section = content.split("## Questions", 1)[-1]
            lines = [line.strip() for line in potential_questions_section.split('\n') if line.strip()]
        else: # if no specific block found, assume content itself might be questions or needs different parsing.
            # This part needs to be robust. For now, using the provided logic from main.py's extract_questions_from_content
            # This assumes questions are separated by newlines.
            lines = [line.strip() for line in content.split('\n') if line.strip()]

    # Process lines to extract actual questions
    for line in lines:
        # Skip lines that are not questions (headers, etc.)
        # The flask example had:
        # if line.startswith('#') or not line:
        #    continue
        # This might need to be adapted if the LLM output for questions is different.
        # For now, let's assume any non-empty line in this block is a question.
        # A more robust solution might look for lines ending with '?' or starting with a number/bullet.
        cleaned_line = line.lstrip("0123456789. ").strip() # Remove leading numbers/bullets
        if cleaned_line and cleaned_line != "}": # Ensure it's not just the closing brace of a block
            questions.append(cleaned_line)
    
    # Deduplicate questions
    return list(dict.fromkeys(questions))


def retrieve_information(questions: list[str]) -> list[dict]:
    """Retrieve information for questions using the OpenScholar external API."""
    if not questions:
        return []
    try:
        # The URL for the OpenScholar API
        openscholar_api_url = 'http://127.0.0.1:38015/batch_ask'
        response = requests.post(
            openscholar_api_url,
            json={"questions": questions},
            timeout=600  # Set a reasonable timeout (in seconds)
        )

        if response.status_code == 200:
            # Assuming the API returns a JSON with a 'results' key
            # which is a list of dictionaries, one for each question.
            return response.json().get('results', [])
        else:
            # Log error or handle appropriately
            print(f"Error retrieving information from OpenScholar API: {response.status_code} - {response.text}")
            return [{"error": f"API Error {response.status_code}", "output": "", "final_passages": ""} for _ in questions]
    except requests.exceptions.RequestException as e:
        # Handle network errors, timeouts, etc.
        print(f"Exception during information retrieval: {str(e)}")
        return [{"error": f"RequestException: {str(e)}", "output": "", "final_passages": ""} for _ in questions]


def get_question_and_answer_text(questions: list[str], results: list[dict]) -> str:
    """Format questions and answers for the second model call."""
    qa_text_parts = []
    for i, question in enumerate(questions):
        qa_text_parts.append(f"## Question {i + 1}:\n{question}")
        if i < len(results) and results[i]:
            result = results[i]
            passages = result.get("final_passages", "N/A")
            answer = result.get("output", "N/A")
            # Sanitize content slightly for inclusion in a prompt if necessary, though LLMs are usually robust.
            # The flask app used .replace('"', "'").replace('\\', '') which might be too aggressive.
            # Keeping it simple here.
            qa_text_parts.append(f"### Retrieved Passages:\n{passages}")
            qa_text_parts.append(f"### Answer from OpenScholar:\n{answer}")
        else:
            qa_text_parts.append("### Retrieved Passages:\nNo information retrieved.")
            qa_text_parts.append("### Answer from OpenScholar:\nNo answer retrieved.")
        qa_text_parts.append("**********") # Separator
    
    return "\n\n".join(qa_text_parts)


class DeepReviewer:
    """
    A class for generating automated academic peer reviews using DeepReviewer models.
    """

    def __init__(self,
                 model_size="14B",
                 custom_model_name=None,
                 device=None,
                 torch_dtype=None,
                 attn_implementation=None,
                 max_input_length=8192,
                 device_map="auto",
                 load_in_4bit=False,
                 load_in_8bit=False,
                 verbose=True):
        """
        Initialize the DeepReviewer (transformers backend, portable defaults).

        All hardware-sensitive parameters are auto-detected when left as None,
        so the same call works on an A100, a laptop RTX, a Mac M-series, or CPU.

        Args:
            model_size (str): "7B" or "14B" (default). On <24 GB VRAM prefer "7B"
                or pass load_in_4bit=True.
            custom_model_name (str, optional): HF model id to override the mapping.
            device (str, optional): "cuda" | "mps" | "cpu". Auto-detected if None.
            torch_dtype (torch.dtype, optional): Auto-selected per device if None
                (bf16 on Ampere+, fp16 on older CUDA/MPS, fp32 on CPU).
            attn_implementation (str, optional): "sdpa" (default, portable),
                "flash_attention_2" (faster but requires flash-attn), or "eager".
                Falls back to "sdpa" if the chosen kernel is unavailable.
            max_input_length (int): Prompt truncation length. 8192 is a safe
                portable default. Raise it on big GPUs if your papers need it.
            device_map (str | dict | None): HF accelerate device map. "auto"
                offloads to CPU/disk if the GPU is too small.
            load_in_4bit (bool): 4-bit NF4 quantization via bitsandbytes.
                Lets 14B run on ~10 GB VRAM. Ignored on non-CUDA devices.
            load_in_8bit (bool): 8-bit quantization via bitsandbytes.
            verbose (bool): Print the auto-detected configuration on load.
        """
        model_mapping = {
            "14B": "WestlakeNLP/DeepReviewer-14B",
            "7B": "WestlakeNLP/DeepReviewer-7B",
        }

        if custom_model_name:
            model_name = custom_model_name
        else:
            if model_size not in model_mapping:
                raise ValueError(f"Invalid model size. Choose from {list(model_mapping.keys())}")
            model_name = model_mapping[model_size]

        # --- Auto-detect hardware ---
        if device is None:
            device = _auto_device()
        if torch_dtype is None:
            torch_dtype = _auto_dtype(device)
        if attn_implementation is None:
            attn_implementation = _auto_attn_implementation(device)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        load_kwargs = {
            "torch_dtype": torch_dtype,
            "low_cpu_mem_usage": True,
        }

        # Quantization is CUDA-only (bitsandbytes)
        quantization = None
        if (load_in_4bit or load_in_8bit) and device == "cuda":
            try:
                from transformers import BitsAndBytesConfig
                quantization = BitsAndBytesConfig(
                    load_in_4bit=load_in_4bit,
                    load_in_8bit=load_in_8bit,
                    bnb_4bit_compute_dtype=torch_dtype if load_in_4bit else None,
                    bnb_4bit_quant_type="nf4" if load_in_4bit else None,
                    bnb_4bit_use_double_quant=bool(load_in_4bit),
                )
                load_kwargs["quantization_config"] = quantization
            except ImportError:
                print("[DeepReviewer] bitsandbytes not installed; ignoring quantization flags.")

        # device_map only makes sense with accelerate; fall back to explicit .to(device)
        if device_map is not None:
            load_kwargs["device_map"] = device_map

        def _load(attn):
            return AutoModelForCausalLM.from_pretrained(
                model_name, attn_implementation=attn, **load_kwargs
            )

        try:
            self.model = _load(attn_implementation)
        except (ImportError, ValueError, RuntimeError) as e:
            if attn_implementation != "sdpa":
                print(f"[DeepReviewer] attn='{attn_implementation}' unavailable ({type(e).__name__}); "
                      "falling back to 'sdpa'.")
                self.model = _load("sdpa")
                attn_implementation = "sdpa"
            else:
                raise

        if device_map is None and quantization is None:
            self.model = self.model.to(device)
        self.model.eval()

        self.device = self.model.device
        self.max_input_length = max_input_length
        self.model_name = model_name
        self.model_config = {
            "device": str(self.device),
            "torch_dtype": str(torch_dtype),
            "attn_implementation": getattr(self.model.config, "_attn_implementation", attn_implementation),
            "max_input_length": max_input_length,
            "device_map": device_map,
            "quantization": "4bit" if load_in_4bit else ("8bit" if load_in_8bit else "none"),
        }

        if verbose:
            print(f"[DeepReviewer] Loaded {model_name}")
            for k, v in self.model_config.items():
                print(f"  - {k}: {v}")

    @torch.inference_mode()
    def _generate(self, prompts, max_new_tokens=8192, temperature=0.4, top_p=0.95):
        """
        Batched text generation using transformers (drop-in replacement for vLLM).

        Args:
            prompts (list[str]): List of already chat-templated prompts.
            max_new_tokens (int): Max tokens to generate per prompt.
            temperature (float): Sampling temperature.
            top_p (float): Nucleus sampling top-p.

        Returns:
            list[str]: Decoded completions (input stripped).
        """
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_input_length,
        ).to(self.device)

        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            use_cache=True,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        outputs = self.model.generate(**inputs, **gen_kwargs)
        # Strip the echoed input tokens
        input_len = inputs["input_ids"].shape[1]
        generated = outputs[:, input_len:]
        return self.tokenizer.batch_decode(generated, skip_special_tokens=True)

    def _generate_system_prompt(self, mode="Standard Mode", reviewer_num=4):
        """
        Generate the system prompt based on the review mode and number of reviewers.

        Args:
            mode (str): Review mode. Options: "Fast Mode", "Standard Mode", "Best Mode"
            reviewer_num (int): Number of reviewers to simulate

        Returns:
            str: System prompt for the specified mode
        """
        simreviewer_prompt = "When you simulate different reviewers, write the sections in this order: Summary, Soundness, Presentation, Contribution, Strengths, Weaknesses, Suggestions, Questions, Rating and Confidence."

        if mode == "Best Mode":
            prompt = f"""You are an expert academic reviewer tasked with providing a thorough and balanced evaluation of research papers. Your thinking mode is Best Mode. In this mode, you should aim to provide the most reliable review results by conducting a thorough analysis of the paper. I allow you to use search tools to obtain background knowledge about the paper - please provide three different questions. I will help you with the search. After you complete your thinking, you should review by simulating {reviewer_num} different reviewers, and use self-verification to double-check any paper deficiencies identified. Finally, provide complete review results."""
            return prompt + simreviewer_prompt
        elif mode == "Standard Mode":
            prompt = f"""You are an expert academic reviewer tasked with providing a thorough and balanced evaluation of research papers. Your thinking mode is Standard Mode. In this mode, you should review by simulating {reviewer_num} different reviewers, and use self-verification to double-check any paper deficiencies identified. Finally, provide complete review results."""
            return prompt + simreviewer_prompt
        elif mode == "Fast Mode":
            return "You are an expert academic reviewer tasked with providing a thorough and balanced evaluation of research papers. Your thinking mode is Fast Mode. In this mode, you should quickly provide the review results."
        else:
            return "You are an expert academic reviewer tasked with providing a thorough and balanced evaluation of research papers."

    def evaluate(self, paper_context, mode="Standard Mode", reviewer_num=4,
                 max_tokens=8192, batch_size=1, temperature=0.4, top_p=0.95):
        """
        Generate a peer review for the given academic paper.

        Args:
            paper_context (str): The paper content to review. Can be a single string or a list of strings for batch processing.
            mode (str): Review mode. Options: "Fast Mode", "Standard Mode", "Best Mode"
            reviewer_num (int): Number of reviewers to simulate
            max_tokens (int): Maximum number of tokens to generate for each LLM call.

        Returns:
            list: A list of structured reviews (dictionaries). Each dictionary corresponds to one input paper_context.
        """
        system_prompt = self._generate_system_prompt(mode, reviewer_num)

        if isinstance(paper_context, str):
            paper_contexts = [paper_context]
        elif isinstance(paper_context, list):
            paper_contexts = paper_context
        else:
            raise TypeError("paper_context must be a string or a list of strings.")

        generated_reviews_batch = []

        for i in range(0, len(paper_contexts), batch_size):
            current_batch_contexts = paper_contexts[i:i + batch_size]

            if mode != "Best Mode":
                prompts = []
                for single_paper_context in current_batch_contexts:
                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": single_paper_context}
                    ]
                    input_text = self.tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    prompts.append(input_text)

                generated_texts = self._generate(
                    prompts, max_new_tokens=max_tokens,
                    temperature=temperature, top_p=top_p,
                )
                for generated_text in generated_texts:
                    generated_reviews_batch.append(self._parse_review(generated_text))
            else:  # Best Mode - sequential per paper (two LLM calls + external retrieval)
                for single_paper_context in current_batch_contexts:
                    # --- First LLM Call ---
                    messages_step1 = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": single_paper_context}
                    ]
                    input_text_step1 = self.tokenizer.apply_chat_template(
                        messages_step1, tokenize=False, add_generation_prompt=True
                    )
                    generated_text_step1 = self._generate(
                        [input_text_step1], max_new_tokens=max_tokens,
                        temperature=temperature, top_p=top_p,
                    )[0]

                    questions = extract_questions_from_content(generated_text_step1)
                    if not questions:
                        generated_reviews_batch.append(self._parse_review(generated_text_step1))
                        continue

                    retrieved_data = retrieve_information(questions)
                    qa_text = get_question_and_answer_text(questions, retrieved_data)

                    # --- Second LLM Call ---
                    messages_step2 = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": single_paper_context},
                        {"role": "assistant", "content": generated_text_step1},
                        {"role": "user", "content": qa_text}
                    ]
                    input_text_step2 = self.tokenizer.apply_chat_template(
                        messages_step2, tokenize=False, add_generation_prompt=True
                    )
                    generated_text_step2 = self._generate(
                        [input_text_step2], max_new_tokens=max_tokens,
                        temperature=temperature, top_p=top_p,
                    )[0]

                    generated_reviews_batch.append(self._parse_review(generated_text_step2))

        return generated_reviews_batch

    def _parse_review(self, generated_text):
        """
        Parse the generated review text into structured format.

        Args:
            generated_text (str): Raw generated review text

        Returns:
            dict: Structured review with metadata and reviews
        """
        result = {
            "raw_text": generated_text,
            "reviews": [],
            "meta_review": {},
            "decision": ""
        }

        # Extract meta review if present
        meta_review_match = re.search(r'\\boxed_review\{(.*?)\n}', generated_text, re.DOTALL)
        if meta_review_match:
            result["meta_review"]['content'] = meta_review_match.group(1).strip()
            section = meta_review_match.group(1).strip()
            # Extract summary
            summary_match = re.search(r'## Summary:\s+(.*?)(?=##|\Z)', section, re.DOTALL)
            if summary_match:
                result["meta_review"]["summary"] = summary_match.group(1).strip()

            # Extract rating
            rating_match = re.search(r'## Rating:\s+(.*?)(?=##|\Z)', section, re.DOTALL)
            if rating_match:
                rating_text = rating_match.group(1).strip()
                # Try to extract a numerical rating (1-10)
                number_match = re.search(r'(\d+(?:\.\d+)?)', rating_text)
                if number_match:
                    result["meta_review"]["rating"] = float(number_match.group(1))
                else:
                    result["meta_review"]["rating"] = rating_text

            # Extract other sections as needed
            for section_name in ["Soundness", "Presentation", "Contribution",
                                 "Strengths", "Weaknesses", "Suggestions", "Questions"]:
                section_match = re.search(f'## {section_name}:\s+(.*?)(?=##|\Z)', section, re.DOTALL)
                if section_match:
                    result["meta_review"][section_name.lower()] = section_match.group(1).strip()

        # Extract simulated reviewers' feedback
        simreviewer_match = re.search(r'\\boxed_simreviewers\{(.*?)\n}', generated_text, re.DOTALL)
        if simreviewer_match:
            simreviewer_text = simreviewer_match.group(1).strip()
            # Split into individual reviewer sections
            reviewer_sections = re.split(r'## Reviewer \d+', simreviewer_text)
            # Skip the first empty section if it exists
            if reviewer_sections and not reviewer_sections[0].strip():
                reviewer_sections = reviewer_sections[1:]

            for i, section in enumerate(reviewer_sections):
                review = {
                    "reviewer_id": i + 1,
                    "text": section.strip()
                }

                # Extract summary
                summary_match = re.search(r'## Summary:\s+(.*?)(?=##|\Z)', section, re.DOTALL)
                if summary_match:
                    review["summary"] = summary_match.group(1).strip()

                # Extract rating
                rating_match = re.search(r'## Rating:\s+(.*?)(?=##|\Z)', section, re.DOTALL)
                if rating_match:
                    rating_text = rating_match.group(1).strip()
                    # Try to extract a numerical rating (1-10)
                    number_match = re.search(r'(\d+(?:\.\d+)?)', rating_text)
                    if number_match:
                        review["rating"] = float(number_match.group(1))
                    else:
                        review["rating"] = rating_text

                # Extract other sections as needed
                for section_name in ["Soundness", "Presentation", "Contribution",
                                     "Strengths", "Weaknesses", "Suggestions", "Questions"]:
                    section_match = re.search(f'## {section_name}:\s+(.*?)(?=##|\Z)', section, re.DOTALL)
                    if section_match:
                        review[section_name.lower()] = section_match.group(1).strip()

                result["reviews"].append(review)

        # Extract decision if present
        decision_match = re.search(r'## Decision:\s*\n\s*(\w+)', generated_text)
        if decision_match:
            result["decision"] = decision_match.group(1).strip()

        return result
