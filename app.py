import streamlit as st
from dotenv import load_dotenv
from src.bharatchat.config import Config
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from src.bharatchat.chatbot import DocumentProcessor, ChatHandler, ToolsAndAgentsInitializer

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
        self.translations = self._load_translations()
        # Set default language if not set
        if 'language' not in st.session_state:
            st.session_state.language = 'en'
        self.current_language = st.session_state.language

    def _load_translations(self):
        return {
            'en': {
                'title': 'Bharat AI ChatBot',
                'chatbot_selection': 'Choose a Chatbot:',
                'model_selection': '🔍 Choose a Model',
                'memory_length': '🧠 Conversational Memory Length:',
                'temperature': '🌡️ Temperature:',
                'max_tokens': '🧩 Max Tokens:',
                'clear_history': '🗑️ Clear Chat History',
                'url_input': 'Enter the URL of the document:',
                'upload_files': 'Upload your files',
                'input_placeholder': 'Input your question here',
                'chatbot_summaries': {
                    'QA Chatbot': 'The QA Chatbot engages in a question-and-answer session, providing accurate and relevant responses to your queries.',
                    'Chat with PDF': 'The Chat with PDF option allows you to upload a PDF document and interact with its content to extract useful information.',
                    'Chat with URL': 'The Chat with URL feature lets you enter a URL of a document to interact with its content and obtain insights.'
                }
            },
            'hi': {
                'title': 'भारत एआई चैटबॉट',
                'chatbot_selection': 'चैटबॉट चुनें:',
                'model_selection': '🔍 मॉडल चुनें',
                'memory_length': '🧠 वार्तालाप की मेमोरी लंबाई:',
                'temperature': '🌡️ तापमान:',
                'max_tokens': '🧩 अधिकतम टोकन:',
                'clear_history': '🗑️ चैट इतिहास साफ़ करें',
                'url_input': 'दस्तावेज़ के URL को दर्ज करें:',
                'upload_files': 'अपनी फ़ाइलें अपलोड करें',
                'input_placeholder': 'यहां अपना सवाल दर्ज करें',
                'chatbot_summaries': {
                    'QA Chatbot': 'QA चैटबॉट प्रश्नोत्तर सत्र में संलग्न होता है, आपकी पूछताछों के सटीक और प्रासंगिक उत्तर प्रदान करता है।',
                    'Chat with PDF': 'PDF के साथ चैट विकल्प आपको एक PDF दस्तावेज़ अपलोड करने और इसके सामग्री के साथ बातचीत करने की अनुमति देता है।',
                    'Chat with URL': 'URL के साथ चैट सुविधा आपको दस्तावेज़ के URL को दर्ज करने और इसके सामग्री के साथ बातचीत करने की अनुमति देती है।'
                }
            }
        }

    def render_app(self):
        """Render the Streamlit app interface."""
        self.current_language = st.session_state.language  # Ensure language is up-to-date
        self._set_background_image()
        self.display_logo_and_title()
        self._initialize_sidebar()
        self._handle_sidebar_selection()
        self._display_chatbot_summary()

    def _set_background_image(self):
        """Set the background image for the app."""
        st.markdown(
            """
            <style>
            .stApp {
                background-image: url("https://example.com/your-background-image.jpg");
                background-size: cover;
                background-position: center;
            }
            </style>
            """,
            unsafe_allow_html=True
        )

    def display_logo_and_title(self):
        """Display the application logo and title side by side with tricolor effect."""
        spacer, col = st.columns([5, 1])
        with col:
            st.image('templates/groqcloud.png', width=100)  # Adjust width as needed
        spacer.markdown(
            f"""
            <div style="font-size: 60px; font-weight: bold; background: linear-gradient(to right, #FF9933, #FFFFFF, #138808); -webkit-background-clip: text; -webkit-text-fill-color: transparent; display: inline;">
                {self.translations[self.current_language]['title']}
            </div>
            <span style="font-size: 80px;">🇮🇳</span>
            """,
            unsafe_allow_html=True
        )

    def _initialize_sidebar(self):
        """Initialize the sidebar with customization options."""
        st.sidebar.title('TweakIt 🎛️')

        # Language selection
        selected_language = st.sidebar.selectbox(
            'Select Language',
            ['en', 'hi'],
            index=['en', 'hi'].index(st.session_state.language),
            format_func=lambda x: 'English' if x == 'en' else 'हिन्दी'
        )

        if selected_language != st.session_state.language:
            st.session_state.language = selected_language
            self.current_language = st.session_state.language
            # Reset chat option and histories to apply language changes
            st.session_state.chat_option = "QA Chatbot"
            st.session_state.chat_histories = {option: [] for option in ["QA Chatbot", "Chat with PDF", "Chat with URL"]}
            st.session_state.vectors = None

        # Ensure chat_option is initialized
        if 'chat_option' not in st.session_state:
            st.session_state.chat_option = "QA Chatbot"

        # Radio button for selecting chat option
        st.sidebar.radio(
            self.translations[self.current_language]['chatbot_selection'],
            ("QA Chatbot", "Chat with PDF", "Chat with URL"),
            index=["QA Chatbot", "Chat with PDF", "Chat with URL"].index(st.session_state.chat_option),
            key="chat_option"
        )

        # Selectbox for choosing a model
        st.sidebar.selectbox(
            self.translations[self.current_language]['model_selection'],
            self.chat_ai._get_model_options(),
            help="Select the AI model you want to use."
        )

        # Slider for conversational memory length
        st.sidebar.slider(
            self.translations[self.current_language]['memory_length'],
            1, 10,
            value=5,
            help="Set how many previous interactions the chatbot should remember."
        )

        # Slider for temperature
        st.sidebar.slider(
            self.translations[self.current_language]['temperature'],
            0.0, 1.0,
            value=0.7,
            step=0.1,
            help="Adjust the randomness of the chatbot's responses."
        )

        # Slider for max tokens
        st.sidebar.slider(
            self.translations[self.current_language]['max_tokens'],
            50, 1000,
            value=300,
            step=50,
            help="Specify the maximum number of tokens for responses."
        )

        # Button to clear chat history
        if st.sidebar.button(self.translations[self.current_language]['clear_history']):
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
        url = st.text_input(self.translations[self.current_language]['url_input'])
        if url:
            self.chat_ai.document_processor.process_url(url)
        self._handle_chat_input()

    def _handle_qa_chat(self):
        """Handle chat interactions for QA chatbot."""
        self._handle_chat_input()

    def _handle_pdf_chat(self):
        """Handle chat interactions when a PDF is uploaded."""
        uploaded_file = st.file_uploader(self.translations[self.current_language]['upload_files'], type=["pdf", "txt"])
        if uploaded_file:
            self.chat_ai.document_processor.process_documents(uploaded_file)
        self._handle_chat_input()

    def _handle_chat_input(self):
        """Handle user chat input and process the chat."""
        query = st.chat_input(placeholder=self.translations[self.current_language]['input_placeholder'])
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

    def _display_chatbot_summary(self):
        """Display the summary of the selected chatbot."""
        chat_option = st.session_state.chat_option
        summary = self.translations[self.current_language]['chatbot_summaries'].get(chat_option, "Select a chatbot to see its summary.")
        st.markdown(f"### {chat_option}")
        st.write(summary)

# Streamlit application execution
if __name__ == "__main__":
    BharatChatAI().run_streamlit_app()
