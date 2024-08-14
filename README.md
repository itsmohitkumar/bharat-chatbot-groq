# Bharat ChatAI

Bharat ChatAI is an AI-powered chatbot application that integrates various AI models and document processing functionalities. This application allows users to chat with the AI using different models, upload and process documents, and retrieve information from URLs.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Docker Setup](#docker-setup)
- [AWS EC2 Deployment](#aws-ec2-deployment)
- [File Structure](#file-structure)
- [Code Overview](#code-overview)
  - [Config](#config)
  - [DocumentProcessor](#documentprocessor)
  - [ChatHandler](#chathandler)
  - [ToolsAndAgentsInitializer](#toolsandagentsinitializer)
  - [BharatChatAI](#bharatchatai)
  - [StreamlitInterface](#streamlitinterface)
- [License](#license)
- [Contact](#contact)

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/itsmohitkumar/bharat-chatbot-groq.git
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

## Docker Setup

To containerize the Bharat ChatAI application using Docker, follow these steps:

1. **Create a Dockerfile:**
   In the root directory of your project, create a `Dockerfile` with the following content:
    ```Dockerfile
    # Use an official Python runtime as a parent image
    FROM python:3.9-slim

    # Set the working directory in the container
    WORKDIR /app

    # Copy the current directory contents into the container at /app
    COPY . /app

    # Install any needed packages specified in requirements.txt
    RUN pip install --no-cache-dir -r requirements.txt

    # Make port 8501 available to the world outside this container
    EXPOSE 8501

    # Define environment variable
    ENV GROQ_API_KEY=your_api_key_here

    # Run the application
    CMD ["streamlit", "run", "app.py"]
    ```

2. **Build the Docker image:**
   Run the following command in the terminal to build your Docker image:
    ```bash
    docker build -t bharat-chatai .
    ```

3. **Run the Docker container:**
   After the image is built, you can run the application in a container with:
    ```bash
    docker run -p 8501:8501 bharat-chatai
    ```

   The application will be accessible at `http://localhost:8501`.

## AWS EC2 Deployment

To deploy the Bharat ChatAI application on AWS EC2, follow these steps:

1. **Launch an EC2 instance:**
   - Go to the [AWS EC2 Dashboard](https://aws.amazon.com/ec2/) and launch a new instance.
   - Choose an Amazon Machine Image (AMI), such as Ubuntu Server 20.04 LTS.
   - Select an instance type, for example, `t2.micro` (free-tier eligible).
   - Configure the instance details, storage, and add tags if necessary.
   - Under "Security Group," configure a rule to allow HTTP traffic (port 80) and port 8501 for Streamlit.

2. **Connect to the EC2 instance:**
   - Use SSH to connect to your instance:
    ```bash
    ssh -i "your-key.pem" ubuntu@ec2-xx-xx-xx-xx.compute-1.amazonaws.com
    ```

3. **Install Docker on the EC2 instance:**
    ```bash
    sudo apt update
    sudo apt install docker.io
    sudo systemctl start docker
    sudo systemctl enable docker
    ```

4. **Clone the Bharat ChatAI repository:**
    ```bash
    git clone https://github.com/itsmohitkumar/bharat-chatbot-groq.git
    cd bharat-chatbot-groq
    ```

5. **Build and run the Docker container:**
   - Follow the Docker setup instructions above to build and run the container on the EC2 instance:
    ```bash
    sudo docker build -t bharat-chatai .
    sudo docker run -p 80:8501 bharat-chatai
    ```

6. **Access the application:**
   - Once the container is running, you can access the application by navigating to the EC2 instance's public IP in your browser (`http://ec2-xx-xx-xx-xx.compute-1.amazonaws.com`).

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

## Contact

For any questions or support, please contact:

Author: Mohit Kumar  
Email: [mohitpanghal12345@gmail.com](mailto:mohitpanghal12345@gmail.com)
