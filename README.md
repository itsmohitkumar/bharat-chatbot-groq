## README.md

# Bharat ChatAI

Bharat ChatAI is an AI-powered chatbot application that integrates various AI models and document processing functionalities. This application allows users to chat with the AI using different models, upload and process documents, and retrieve information from URLs.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [File Structure](#file-structure)
- [Code Overview](#code-overview)
  - [Config](#config)
  - [DocumentProcessor](#documentprocessor)
  - [ChatHandler](#chathandler)
  - [ToolsAndAgentsInitializer](#toolsandagentsinitializer)
  - [BharatChatAI](#bharatchatai)
  - [StreamlitInterface](#streamlitinterface)
- [License](#license)

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/bharat-chatbot-groq.git
    cd bharat-chatbot-groq
    ```

2. Create a virtual environment and activate it:
    ```bash
    python -m venv env
    source env/bin/activate  # On Windows, use `env\Scripts\activate`
    ```

3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

4. Create a `.env` file in the project root directory and add your API key:
    ```
    GROQ_API_KEY=your_api_key_here
    ```

## Usage

Run the Streamlit application:
```bash
streamlit run app.py
```

## File Structure

The file structure of the project is as follows:

```
bharat-chatai/
├── src/
│   ├── __init__.py
│   ├── logger.py
│   ├── prompt.py
│   ├── bharatchat/
│   │   ├── __init__.py
│   │   └── chatbot.py
├── setup.py
├── app.py
```

- `src/__init__.py`: Initialization file for the `src` package.
- `src/logger.py`: Module for logging configuration.
- `src/prompt.py`: Module for defining prompt templates.
- `src/bharatchat/__init__.py`: Initialization file for the `bharatchat` package.
- `src/bharatchat/chatbot.py`: Main module for the chatbot logic.
- `setup.py`: Setup script for the package.
- `app.py`: Main application file for running the Streamlit interface.

## Code Overview

### Config

The `Config` class handles the configuration of the application, including retrieving the API key and fetching available model options from the Groq API.

### DocumentProcessor

The `DocumentProcessor` class processes documents from uploaded files or URLs, splits them into chunks, and stores them in a FAISS vector store. It also generates summaries of the processed documents.

### ChatHandler

The `ChatHandler` class handles chat queries, displays chat history, and retrieves responses using the document vectors.

### ToolsAndAgentsInitializer

The `ToolsAndAgentsInitializer` class initializes the tools and agents for the chat interface, including setting up the model and creating combined chains for document and query processing.

### BharatChatAI

The `BharatChatAI` class initializes the application, including embeddings, document processing, and chat handling. It also runs the Streamlit app interface.

### StreamlitInterface

The `StreamlitInterface` class renders the Streamlit app interface, including initializing the sidebar and handling user selections.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
