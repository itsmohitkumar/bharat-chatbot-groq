import os
import requests
import time
from datetime import datetime
from dotenv import load_dotenv
import streamlit as st

from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader, TextLoader
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import (
    ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun, Tool
)
from langchain.agents import initialize_agent, AgentType
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.embeddings import HuggingFaceBgeEmbeddings


class Config:
    API_KEY_ENV_VAR = 'GROQ_API_KEY'
    MODEL_OPTIONS_URL = "https://api.groq.com/openai/v1/models"
    DEFAULT_MODEL = "mixtral-8x7b-32768"

    @staticmethod
    def get_api_key():
        """Retrieve API key from environment variables."""
        api_key = os.getenv(Config.API_KEY_ENV_VAR)
        if not api_key:
            raise ValueError(f"API key not found. Please set the '{Config.API_KEY_ENV_VAR}' environment variable.")
        return api_key

    @staticmethod
    def get_model_options(api_key):
        """Fetch available model options from the Groq API."""
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        response = requests.get(Config.MODEL_OPTIONS_URL, headers=headers)
        response.raise_for_status()
        models = response.json()
        return {model['id'] for model in models['data']}

    @staticmethod
    def setup_langchain():
        """Setup LangChain environment variables and project."""
        langchain_api_key = os.getenv("LANGCHAIN_API_KEY")
        if not langchain_api_key:
            raise ValueError("LANGCHAIN_API_KEY environment variable is not set.")
        
        # Set project name and environment variables
        project_name = "Bharat Chatbot"
        os.environ["LANGCHAIN_PROJECT"] = project_name
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = langchain_api_key

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
            summary_query = "Please summarize the content of the document."
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
                result = retrieval_chain.invoke({"input": query})
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
    def __init__(self, model):
        self.model = model
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
            DuckDuckGoSearchRun(name="Search"),
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
        prompt_template = """
        Answer the questions based on the provided context only.
        Please provide the most accurate response based on the question
        <context>
        {context}
        </context>
        Questions: {input}
        """
        prompt = ChatPromptTemplate.from_template(prompt_template)
        return create_stuff_documents_chain(
            llm,
            prompt,
            document_prompt=ChatPromptTemplate.from_template("{page_content}"),
            document_separator="\n\n"
        )


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
        model_name = "all-MiniLM-L6-v2"
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

    def run_streamlit_app(self):
        """Run the Streamlit app interface."""
        StreamlitInterface(self).render_app()

class StreamlitInterface:
    def __init__(self, chat_ai_instance):
        self.chat_ai = chat_ai_instance

    def render_app(self):
        """Render the Streamlit app interface."""
        st.title("Bharat ChatAI")
        self._initialize_sidebar()
        self._handle_sidebar_selection()

    def _initialize_sidebar(self):
        """Initialize the sidebar with customization options."""
        st.sidebar.title('🔧 Customization')
        st.sidebar.radio("Choose a Chatbot:", ("QA Chatbot", "Chat with PDF", "Chat with URL"), index=0, key="chat_option")
        st.sidebar.selectbox('🔍 Choose a Model', self.chat_ai._get_model_options(), help="Select the AI model you want to use.")
        st.sidebar.slider('🧠 Conversational Memory Length:', 1, 10, value=5, help="Set how many previous interactions the chatbot should remember.")
        st.sidebar.slider('🌡️ Temperature:', 0.0, 1.0, value=0.7, step=0.1, help="Adjust the randomness of the chatbot's responses.")
        st.sidebar.slider('🧩 Max Tokens:', 50, 1000, value=300, step=50, help="Specify the maximum number of tokens for responses.")
        if st.sidebar.button("🗑️ Clear Chat History"):
            st.session_state.chat_histories[st.session_state.chat_option] = []
            st.success("✅ Chat history cleared!")

    def _handle_sidebar_selection(self):
        """Handle user selection from the sidebar and initialize tools and agents."""
        chat_option = st.session_state.chat_option

        # Ensure chat histories are initialized
        if 'chat_histories' not in st.session_state:
            st.session_state.chat_histories = {option: [] for option in ["QA Chatbot", "Chat with PDF", "Chat with URL"]}

        # Ensure vectors are initialized
        if 'vectors' not in st.session_state:
            st.session_state.vectors = None

        # Initialize model options
        model_options = self.chat_ai._get_model_options()
        selected_model = model_options[0] if model_options else Config.DEFAULT_MODEL

        # Initialize tools and agents
        tools_and_agents_initializer = ToolsAndAgentsInitializer(model=selected_model)
        self.chat_ai.search_agent = tools_and_agents_initializer.initialize_tools_and_agents()

        if chat_option == "Chat with URL":
            self._handle_url_chat()
        elif chat_option == "QA Chatbot":
            self._handle_qa_chat()
        elif chat_option == "Chat with PDF":
            self._handle_pdf_chat()
        else:
            st.warning("Please select a chat option to get started.")

    def _handle_url_chat(self):
        """Handle chat interactions when URL is selected."""
        url = st.text_input("Enter the URL of the document:")
        if url:
            self.chat_ai.document_processor.process_url(url)
        query = st.chat_input(placeholder="Input your question here")
        if query:
            self.chat_ai.chat_handler = ChatHandler(st.session_state.vectors)
            self.chat_ai.chat_handler.handle_chat(query)

    def _handle_qa_chat(self):
        """Handle chat interactions for QA chatbot."""
        query = st.chat_input(placeholder="Input your question here")
        if query:
            st.session_state.chat_histories[st.session_state.chat_option].append({"role": "user", "content": query})
            self.chat_ai.chat_handler = ChatHandler(st.session_state.vectors)
            self.chat_ai.chat_handler._display_chat_history()
            self._display_chat_response()

    def _display_chat_response(self):
        """Display the response from the chat agent."""
        with st.chat_message("assistant"):
            st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
            try:
                response = self.chat_ai.search_agent.run(st.session_state.chat_histories[st.session_state.chat_option], callbacks=[st_cb])
                st.session_state.chat_histories[st.session_state.chat_option].append({'role': 'assistant', "content": response})
                st.write(response)
            except Exception as e:
                st.error(f"An error occurred: {e}")

    def _handle_pdf_chat(self):
        """Handle chat interactions when a PDF is uploaded."""
        uploaded_file = st.file_uploader("Upload your files", type=["pdf", "txt"])
        if uploaded_file:
            self.chat_ai.document_processor.process_documents(uploaded_file)
        query = st.chat_input(placeholder="Input your question here")
        if query:
            self.chat_ai.chat_handler = ChatHandler(st.session_state.vectors)
            self.chat_ai.chat_handler.handle_chat(query)


# Streamlit application execution
if __name__ == "__main__":
    BharatChatAI().run_streamlit_app()