import os
import time
import requests
import streamlit as st
from datetime import datetime
from dotenv import load_dotenv
from src.prompt import PROMPTS
from langchain_groq import ChatGroq
from src.bharatchat.config import Config
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.utilities import (
    ArxivAPIWrapper,
    WikipediaAPIWrapper
)
from langchain_community.tools import (
    Tool,
    ArxivQueryRun,
    WikipediaQueryRun,
    DuckDuckGoSearchRun
)
from langchain_community.document_loaders import (
    PyPDFLoader,
    WebBaseLoader,
    TextLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.agents import initialize_agent, AgentType
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain

# Disable tokenizers parallelism to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class DocumentProcessor:
    def __init__(self, embeddings):
        self.embeddings = embeddings

    def process_documents(self, file):
        """Process documents from an uploaded file."""
        temp_file_path = "temp_file"
        try:
            self._save_file(file, temp_file_path)
            docs = self._load_documents(temp_file_path, file.type)
            self._process_and_store_documents(docs)
            self._generate_summary()
        except Exception as e:
            st.error(f"An error occurred: {e}")
        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    def process_url(self, url):
        """Process documents from a provided URL."""
        try:
            loader = WebBaseLoader(url)
            docs = loader.load()
            self._process_and_store_documents(docs)
            self._generate_summary()
        except Exception as e:
            st.error(f"An error occurred while processing the URL: {e}")

    def _save_file(self, file, path):
        """Save the uploaded file to a temporary path."""
        with open(path, "wb") as f:
            f.write(file.getbuffer() if hasattr(file, 'getbuffer') else requests.get(file).content)

    def _load_documents(self, path, file_type):
        """Load documents from the specified path based on file type."""
        loader = PyPDFLoader(path) if file_type == "application/pdf" else TextLoader(path)
        return loader.load()

    def _process_and_store_documents(self, docs):
        """Process and store documents in FAISS vector store."""
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        final_documents = text_splitter.split_documents(docs)

        # Add metadata to documents
        for doc in final_documents:
            doc.metadata = {
                'context': doc.page_content,  # or another field that represents context
                'input': doc.page_content  # or another field that represents input
            }

        if final_documents:
            st.session_state.final_documents = final_documents
            st.session_state.vectors = FAISS.from_documents(final_documents, self.embeddings)
            st.success("Content processed successfully!")
        else:
            st.warning("No content found.")

    def _generate_summary(self):
        """Generate a summary of the processed documents."""
        if 'vectors' in st.session_state:
            retriever = st.session_state.vectors.as_retriever()
            retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)
            summary_query = self._get_summary_query()
            start = time.process_time()
            try:
                summary_result = retrieval_chain.invoke({"input": summary_query})
                response_time = time.process_time() - start
                if 'answer' in summary_result:
                    st.write(f"Summary generated in: {response_time:.2f} seconds")
                    st.write("Summary:")
                    st.write(summary_result['answer'])
                else:
                    st.write("No summary available.")
            except Exception as e:
                st.error(f"An error occurred during summary generation: {e}")

    def _get_summary_query(self):
        """Get the summary query based on the current language."""
        return "Please summarize the content of the document." if st.session_state.get('language', 'en') == 'en' else "कृपया दस्तावेज़ की सामग्री का सारांश प्रदान करें।"


class ChatHandler:
    def __init__(self, vectors):
        self.vectors = vectors

    def handle_chat(self, query):
        """Handle chat queries and display responses."""
        st.session_state.chat_histories[st.session_state.chat_option].append({"role": "user", "content": query})
        self._display_chat_history()

        if self.vectors:
            retriever = self.vectors.as_retriever()
            retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)
            start = time.process_time()
            try:
                result = retrieval_chain.invoke({"input": query})  # Updated to invoke
                response_time = time.process_time() - start
                if 'answer' in result:
                    st.write(f"Response time: {response_time:.2f} seconds")
                    st.write(result['answer'])
                    st.session_state.chat_histories[st.session_state.chat_option].append({'role': 'assistant', "content": result['answer']})
                else:
                    st.write("No answer found.")
            except Exception as e:
                st.error(f"An error occurred during retrieval: {e}")
        else:
            st.warning("No documents available for search. Please process content first.")

    def _display_chat_history(self):
        """Display chat history in the Streamlit interface."""
        for message in st.session_state.chat_histories[st.session_state.chat_option]:
            role = message['role']
            st.chat_message(role).write(message['content'])

class ToolsAndAgentsInitializer:
    def __init__(self, model, language):
        self.model = model
        self.language = language
        self.api_key = Config.get_api_key()
        self.llm_model_name = Config.DEFAULT_MODEL

    def initialize_tools_and_agents(self):
        """Initialize tools and agents for the chat interface."""
        tools = self._get_tools()
        if self.api_key:
            model_options = Config.get_model_options(self.api_key)
            if self.model in model_options:
                self.llm_model_name = self.model

            llm = ChatGroq(groq_api_key=self.api_key, model_name=self.llm_model_name, streaming=True)
            global combine_docs_chain
            combine_docs_chain = self._create_combined_chain(llm)

            search_agent = initialize_agent(
                tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, handle_parsing_errors=True
            )
            return search_agent
        else:
            st.warning("API key not found. Please check your configuration.")
            return None

    def _get_tools(self):
        """Get a list of tools available for the chat agent."""
        return [
            DuckDuckGoSearchRun(name="Search Web"),
            ArxivQueryRun(api_wrapper=ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)),
            WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)),
            Tool(
                name="Datetime",
                func=lambda x: datetime.now().isoformat(),
                description="Returns the current datetime",
            )
        ]

    def _create_combined_chain(self, llm):
        """Create a combined chain for processing document contexts and queries."""
        prompt_template = self._get_prompt_templates(self.language)
        #st.write(f"Using prompt templates for language: {self.language}")  # Debugging line
        
        return create_stuff_documents_chain(
            llm,
            ChatPromptTemplate.from_template(prompt_template['summary']),
            document_prompt=ChatPromptTemplate.from_template(prompt_template['qa']),
            document_separator="\n\n"
        )

    def _get_prompt_templates(self, language):
        templates = PROMPTS
        return templates.get(language, templates['en'])

class BharatChatAI:
    def __init__(self):
        load_dotenv()
        Config.setup_langchain()  # Set up LangChain environment variables
        self.embeddings = self._initialize_embeddings()
        st.session_state.embeddings = self.embeddings
        self.document_processor = DocumentProcessor(self.embeddings)
        self.chat_handler = None
        self.search_agent = None

    def _initialize_embeddings(self):
        """Initialize embeddings using HuggingFace BGE."""
        model_name = "all-MiniLM-L12-v2"
        model_kwargs = {"device": "cpu"}
        encode_kwargs = {"normalize_embeddings": True}
        return HuggingFaceBgeEmbeddings(model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs)

    def _get_model_options(self):
        """Fetch and return available model options."""
        try:
            model_options = [Config.DEFAULT_MODEL]
            model_options += list(Config.get_model_options(Config.get_api_key()))
            return model_options
        except Exception as e:
            st.warning(f"Failed to fetch model options: {e}")
            return [Config.DEFAULT_MODEL]
