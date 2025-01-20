from setuptools import setup, find_packages

setup(
    name="DeepCardiac",
    version="0.1.0",
    author="Ghasem Hajainfar",
    author_email="Ghasem.Hajianfar@unige.ch",
    description="A tool to generate saliency maps for cardiac analysis using a pre-trained deep learning model",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/saliency-map-generator",
    packages=find_packages(),
    py_modules=["inference_script", "utilit"],
    install_requires=[
        "tensorflow==2.14",
        "pandas",
        "h5py==3.9.0",
        "matplotlib",
        "skimage==0.21.0",
        "opencv-python-headless",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.9',
)
```