from setuptools import find_packages, setup

with open('README.md', 'r') as f:
    long_description = f.read()

setup(
    name='bharat_chatbot_groq',
    version='0.0.1',
    author='Mohit Kumar',
    author_email='mohitpanghal12345@gmail.com',
    description='An AI-powered chatbot using multiple open source models using groq',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/itsmohitkumar/Bharat_Chatbot_Groq',
    install_requires=[
        'langchain',
        'streamlit',
        'python-dotenv',
        'langchain-core',
        'langchain_community',
        'pypdf',
        'langchain_groq',
        'sentence_transformers',
        'duckduckgo-search',
        'arxiv',
        'wikipedia',
        'faiss-cpu'
    ],
    packages=find_packages(),
    include_package_data=True,
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    python_requires='>=3.9',
)
