from revisa.utils import get_paper_from_generated_text
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


class CycleResearcher:
    """
    A class for generating research papers using CycleResearcher models.
    """

    def __init__(self,
                 model_size="12B",
                 custom_model_name=None,
                 device="cuda",
                 tensor_parallel_size=1,
                 gpu_memory_utilization=0.95,
                 max_model_len=60000):
        """
        Initialize the CycleResearcher.

        Args:
            model_size (str): Size of the default model to use. Options: "12B", "72B", "123B"
            custom_model_name (str, optional): Custom model name to override default mapping
            device (str): Device to run the model on. Default is "cuda"
            tensor_parallel_size (int): Number of GPUs to use for tensor parallelism
            gpu_memory_utilization (float): Fraction of GPU memory to use
        """
        model_mapping = {
            "12B": "WestlakeNLP/CycleResearcher-ML-12B",
            "72B": "WestlakeNLP/CycleResearcher-ML-72B",
            "123B": "WestlakeNLP/CycleResearcher-ML-123B"
        }

        # Determine model name
        if custom_model_name:
            model_name = custom_model_name
        else:
            if model_size not in model_mapping:
                raise ValueError(f"Invalid model size. Choose from {list(model_mapping.keys())}")
            model_name = model_mapping[model_size]

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Load model using vLLM
        self.model = LLM(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization
        )

        # Store model configuration for reference
        self.model_name = model_name
        self.model_config = {
            "tensor_parallel_size": tensor_parallel_size,
            "gpu_memory_utilization": gpu_memory_utilization
        }


    def generate_paper(self, topic=None, references=None, max_tokens=19000, n=1):
        """
        Generate a research paper on the given topic.

        Args:
            topic (str): Research paper topic
            references (str, optional): BibTeX context
            max_tokens (int, optional): Maximum number of tokens to generate

        Returns:
            dict: Generated paper with various components
        """
        # Prepare system prompt
        system_prompt = "You are a research assistant AI tasked with generating a scientific paper based on provided literature. Follow these steps:\n\n1. Analyze the given References. \n2. Identify gaps in existing research to establish the motivation for a new study.\n3. Propose a main idea for a new research work.\n4. Write the paper's main content in LaTeX format, including:\n   - Title\n   - Abstract\n   - Introduction\n   - Related Work\n   - Methods/\n5. Generate experimental setup details in JSON format to guide researchers.\n6. After receiving experimental results in JSON format, analyze them.\n7. Complete the paper by writing:\n   - Results\n   - Discussion\n   - Conclusion\n   - Contributions\n\nEnsure all content is original, academically rigorous, and follows standard scientific writing conventions."""

        # Prepare user prompt
        user_prompt = ''
        if topic != None:
            user_prompt = f"Generate a research paper on the topic: {topic}\n\n"
        if references != None:
            user_prompt += references
            user_prompt += '\n\n'
        user_prompt += 'The above content represents the relevant literature in this field. Please analyze it and provide the motivation and main idea. Then, provide the Title, Abstract, Introduction, Related Work, and Methods sections in LaTeX format.'
        # Prepare messages for generation
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        # Prepare sampling parameters
        sampling_params = SamplingParams(
            temperature=0.4,
            top_p=0.95,
            max_tokens=max_tokens,
            ignore_eos=True,
            stop=['\clearpage','\clear','clearpage']
        )
        # Apply chat template
        prompts = []
        papers = []
        batch_size = 10
        for p in range(0, n, batch_size):
            for _ in range(min(batch_size, n - p)):
                input_text = self.tokenizer.apply_chat_template(messages, tokenize=False,
                                                  add_generation_prompt=True)
                prompts.append(input_text)
            # Generate paper
            outputs = self.model.generate(
                prompts,
                sampling_params
            )
            for output_num in range(len(outputs)):
                # Process generated text
                generated_text = outputs[output_num].outputs[0].text
                # Use existing CycleResearcher utility to parse generated text

                paper = get_paper_from_generated_text(generated_text)
                papers.append(paper)
        return papers