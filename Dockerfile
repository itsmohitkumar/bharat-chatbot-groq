# Use a base Python image
FROM python:3.9-slim

# Set environment variables
ENV LANGCHAIN_TRACING_V2=true
ENV LANGCHAIN_PROJECT=MCQ_Generator
ENV LANGCHAIN_API_KEY=your_langchain_api_key

# Set working directory
WORKDIR /app

# Copy the requirements file
COPY requirements.txt requirements.txt

# Install dependencies with verbose output
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir --upgrade setuptools
RUN pip install --no-cache-dir -r requirements.txt --verbose

# Copy the rest of the application code
COPY . .

# Expose the port that the Streamlit app will run on
EXPOSE 8501

# Command to run the Streamlit app
CMD ["streamlit", "run", "app.py"]
