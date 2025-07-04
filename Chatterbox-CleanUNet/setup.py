from setuptools import setup, find_packages

setup(
    name="chatterbox-cleanunet",
    version="1.0.0",
    description="CleanUNet-based TTS artifact removal system",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.12.0",
        "torchaudio>=0.12.0",
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "librosa>=0.9.0",
        "pesq>=0.0.3",
        "pystoi>=0.3.3",
        "PyYAML>=6.0",
        "tqdm>=4.64.0",
        "matplotlib>=3.5.0",
        "tensorboard>=2.8.0",
        "soundfile>=0.10.0",
    ],
    entry_points={
        "console_scripts": [
            "cleanunet-train=scripts.train:main",
            "cleanunet-enhance=scripts.enhance_audio:main",
            "cleanunet-evaluate=scripts.evaluate_model:main",
        ],
    },
) 