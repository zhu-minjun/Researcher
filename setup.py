from setuptools import setup, find_packages

setup(
    name='revisa',
    version='0.1.0',
    description='RevISA - Agentes de IA na Avaliacao Cientifica de Publicacoes em Periodicos Brasileiros',
    author='Felipe Amaro',
    url='https://github.com/ResearAI/Researcher',
    packages=find_packages(),
    install_requires=[
        'torch>=2.1',
        'transformers>=4.48.2',
        'accelerate>=0.30',
        'bibtexparser',
        'requests',
    ],
    extras_require={
        'vllm': ['vllm>=0.7.2'],
        'flash-attn': ['flash-attn>=2.5'],
        'quantization': ['bitsandbytes>=0.43'],
    },
    classifiers=[
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
)
