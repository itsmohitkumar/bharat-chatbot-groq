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
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    MODEL_NAME = "BAAI/bge-small-en"
    MODEL_KWARGS = {"device": "cpu"}
    ENCODE_KWARGS = {"normalize_embeddings": True}
    PROMPT_TEMPLATE_QA = """
        You are an intelligent assistant.
        Context: {context}
        Human: {input}
        Assistant:
    """
    PROMPT_TEMPLATE_PDF = """
        You are an assistant who provides detailed answers based on the given document content.
        Context: {context}
        Question: {input}
        Assistant:
    """
    PROMPT_TEMPLATE_URL = """
        You are an assistant who discusses information from the provided URL.
        Context: {context}
        Question: {input}
        Assistant:
    """
    DOCUMENT_PROMPT_TEMPLATE = "{page_content}"

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
            f.write(file.getbuffer() if hasattr(file, 'getbuffer') else file.read())

    def _load_documents(self, path, file_type):
        """Load documents from the specified path based on file type."""
        if file_type == "application/pdf":
            loader = PyPDFLoader(path)
        else:
            loader = TextLoader(path)
        return loader.load()

    def _process_and_store_documents(self, docs):
        """Process and store documents in FAISS vector store."""
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=Config.CHUNK_SIZE, chunk_overlap=Config.CHUNK_OVERLAP)
        final_documents = text_splitter.split_documents(docs)
        if final_documents:
            st.session_state.final_documents = final_documents
            st.session_state.vectors = FAISS.from_documents(final_documents, self.embeddings)
            st.success("Content processed successfully!")
        else:
            st.warning("No content found.")

    def _generate_summary(self):
        """Generate a summary of the processed documents."""
        if 'vectors' in st.session_state and st.session_state.vectors:
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
    def __init__(self, model, chatbot_option):
        self.model = model
        self.api_key = Config.get_api_key()
        self.llm_model_name = Config.DEFAULT_MODEL
        self.chatbot_option = chatbot_option

    def initialize_tools_and_agents(self):
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
        if self.chatbot_option == "Chat with PDF":
            prompt = ChatPromptTemplate.from_template(Config.PROMPT_TEMPLATE_PDF)
        elif self.chatbot_option == "Chat with URL":
            prompt = ChatPromptTemplate.from_template(Config.PROMPT_TEMPLATE_URL)
        else:
            prompt = ChatPromptTemplate.from_template(Config.PROMPT_TEMPLATE_QA)

        return create_stuff_documents_chain(
            llm,
            prompt,
            document_prompt=ChatPromptTemplate.from_template(Config.DOCUMENT_PROMPT_TEMPLATE),
            document_separator="\n\n"
        )
    
class BharatChatAI:
    def __init__(self):
        load_dotenv()
        self.embeddings = self._initialize_embeddings()
        st.session_state.embeddings = self.embeddings
        self.document_processor = DocumentProcessor(self.embeddings)
        self.chat_handler = None
        self.search_agent = None

    def _initialize_embeddings(self):
        """Initialize embeddings for document processing."""
        return HuggingFaceBgeEmbeddings(model_name=Config.MODEL_NAME, model_kwargs=Config.MODEL_KWARGS, encode_kwargs=Config.ENCODE_KWARGS)

    def process_file(self, file):
        """Process an uploaded file."""
        self.document_processor.process_documents(file)

    def process_url(self, url):
        """Process content from a provided URL."""
        self.document_processor.process_url(url)

    def handle_chat(self, query):
        """Handle chat queries."""
        if not self.chat_handler:
            self.chat_handler = ChatHandler(st.session_state.vectors)
        self.chat_handler.handle_chat(query)

    def _get_model_options(self):
        """Get the available model options."""
        return Config.get_model_options(Config.get_api_key())


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
        
        # Chatbot Selection
        st.sidebar.header("Choose a Chatbot:")
        chatbot_option = st.sidebar.radio(
            "",
            ["QA Chatbot", "Chat with PDF", "Chat with URL"],
            index=0,
            key="chat_option",
            format_func=lambda x: f"• {x}"
        )
        
        # Model Selection
        model_options = self.chat_ai._get_model_options()
        if model_options:
            st.sidebar.selectbox(
                '🔍 Choose a Model',
                model_options,
                help="Select the AI model you want to use."
            )
        else:
            st.sidebar.text("No models available")

        # Sliders
        st.sidebar.subheader('Settings')
        st.sidebar.slider(
            '🧠 Conversational Memory Length:',
            1, 10, value=5,
            help="Set how many previous interactions the chatbot should remember."
        )
        st.sidebar.slider(
            '🌡️ Temperature:',
            0.0, 1.0, value=0.7, step=0.1,
            help="Adjust the randomness of the chatbot's responses."
        )
        st.sidebar.slider(
            '🧩 Max Tokens:',
            50, 1000, value=300, step=50,
            help="Specify the maximum number of tokens for responses."
        )
        
        # Clear Chat History Button
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
        model_options = list(self.chat_ai._get_model_options())  # Convert set to list
        if model_options:
            selected_model = model_options[0]
        else:
            selected_model = Config.DEFAULT_MODEL  # Fallback to default model if no options available

        # Initialize tools and agents
        tools_and_agents_initializer = ToolsAndAgentsInitializer(model=selected_model, chatbot_option=chat_option)
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
            
if __name__ == "__main__":
    chat_ai_instance = BharatChatAI()
    interface = StreamlitInterface(chat_ai_instance)
    interface.render_app()
