import streamlit as st
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from src.bharatchat.chatbot import BharatChatAI, ChatHandler, ToolsAndAgentsInitializer

class StreamlitInterface:
    def __init__(self, chat_ai_instance):
        self.chat_ai = chat_ai_instance
        self.translations = self._load_translations()
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
                },
                'no_content_found': 'No content found in the response.',
                'unexpected_format': 'Response is not in the expected format.',
                'error_message': 'An error occurred: {}',
                'initializing_chat_handler': 'Chat handler is not initialized. Initializing now.',
                'content_processed': 'Content processed successfully!'
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
                },
                'no_content_found': 'प्रतिक्रिया में कोई सामग्री नहीं मिली।',
                'unexpected_format': 'प्रतिक्रिया अपेक्षित प्रारूप में नहीं है।',
                'error_message': 'एक त्रुटि हुई: {}',
                'initializing_chat_handler': 'चैट हैंडलर शुरू नहीं किया गया है। अभी शुरू हो रहा है।',
                'content_processed': 'सामग्री सफलतापूर्वक संसाधित की गई!'
            }
        }

    def render_app(self):
        """Render the Streamlit app interface."""
        if 'chat_option' not in st.session_state:
            st.session_state.chat_option = "QA Chatbot"
        if 'chat_histories' not in st.session_state:
            st.session_state.chat_histories = {option: [] for option in ["QA Chatbot", "Chat with PDF", "Chat with URL"]}
        if 'vectors' not in st.session_state:
            st.session_state.vectors = None

        self.current_language = st.session_state.language
        self._set_background_image()
        self.display_logo_and_title()
        self._display_chatbot_summary()  # Display summary based on selected chatbot
        self._initialize_sidebar()
        self._handle_sidebar_selection()

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
        languages = ['en', 'hi']
        selected_language = st.sidebar.selectbox(
            'Select Language',
            languages,
            index=languages.index(st.session_state.get('language', 'en')),
            format_func=lambda x: 'English' if x == 'en' else 'हिन्दी'
        )

        if selected_language != st.session_state.get('language', 'en'):
            st.session_state.language = selected_language
            self.current_language = selected_language
            # Reset chat option and histories if language changes
            st.session_state.chat_option = "QA Chatbot"
            st.session_state.chat_histories = {option: [] for option in ["QA Chatbot", "Chat with PDF", "Chat with URL"]}
            st.session_state.vectors = None

        # Ensure that we don't update session state after widget creation
        chat_options = ["QA Chatbot", "Chat with PDF", "Chat with URL"]
        if 'chat_option' not in st.session_state:
            st.session_state.chat_option = "QA Chatbot"

        selected_option = st.sidebar.radio(
            self.translations[self.current_language]['chatbot_selection'],
            chat_options,
            index=chat_options.index(st.session_state.chat_option),
            key="chat_option"
        )

        # Selectbox for choosing a model
        st.sidebar.selectbox(
            self.translations[self.current_language]['model_selection'],
            self.chat_ai._get_model_options(),
            help="Select the AI model you want to use."
        )

        # Slider for conversational memory length
        memory_length = st.sidebar.slider(
            self.translations[self.current_language]['memory_length'],
            1, 10,
            value=st.session_state.get('memory_length', 5),
            help="Set how many previous interactions the chatbot should remember."
        )
        st.session_state.memory_length = memory_length

        # Slider for temperature
        temperature = st.sidebar.slider(
            self.translations[self.current_language]['temperature'],
            0.0, 1.0,
            value=st.session_state.get('temperature', 0.7),
            step=0.1,
            help="Adjust the randomness of the chatbot's responses."
        )
        st.session_state.temperature = temperature

        # Slider for max tokens
        max_tokens = st.sidebar.slider(
            self.translations[self.current_language]['max_tokens'],
            50, 1000,
            value=st.session_state.get('max_tokens', 300),
            step=50,
            help="Specify the maximum number of tokens for responses."
        )
        st.session_state.max_tokens = max_tokens

        # Button to clear chat history
        if st.sidebar.button(self.translations[self.current_language]['clear_history']):
            st.session_state.chat_histories[st.session_state.chat_option] = []
            st.success("✅ Chat history cleared!")

    def _handle_sidebar_selection(self):
        """Handle user selection from the sidebar and initialize tools and agents."""
        chat_option = st.session_state.get("chat_option", "QA Chatbot")

        selected_model = st.session_state.get("_MODEL", "default_model")  # Assuming default model if not set
        if selected_model != st.session_state.get("_MODEL"):
            st.session_state._MODEL = selected_model

        # Initialize tools and agents
        tools_and_agents_initializer = ToolsAndAgentsInitializer(model=selected_model, language=self.current_language)
        self.chat_ai.search_agent = tools_and_agents_initializer.initialize_tools_and_agents()

        if chat_option == "Chat with URL":
            self._handle_url_chat()
        elif chat_option == "QA Chatbot":
            self._handle_qa_chat()
        elif chat_option == "Chat with PDF":
            self._handle_pdf_chat()
        else:
            st.warning("Please select a chat option to get started.")
        
    def _initialize_chat_handler(self):
        """Initialize the chat handler if it is not already initialized."""
        if not self.chat_ai.chat_handler:
            try:
                # Initialize the chat handler without passing additional arguments
                self.chat_ai.chat_handler = ChatHandler(st.session_state.vectors)
            except Exception as e:
                st.error(f"Failed to initialize chat handler: {e}")
                return False
        return True

    def _handle_qa_chat(self):
        query = st.chat_input(placeholder=self.translations[self.current_language]['input_placeholder'])
        if query:
            if not self._initialize_chat_handler():
                return
            st.session_state.chat_histories[st.session_state.chat_option].append({"role": "user", "content": query})
            self.chat_ai.chat_handler._display_chat_history()
            self._display_chat_response()

    def _handle_pdf_chat(self):
        uploaded_file = st.file_uploader(self.translations[self.current_language]['upload_files'], type=["pdf", "txt"])
        
        if uploaded_file:
            # Initialize ChatHandler if not already done
            if not self._initialize_chat_handler():
                return
            
            # Process the uploaded file
            self.chat_ai.document_processor.process_documents(uploaded_file)
            
            # Display content processed message if it hasn't been displayed yet
            if not st.session_state.get('content_processed_displayed', False):
                st.success(self.translations[self.current_language]['content_processed'])
                st.session_state.content_processed_displayed = False

            # Handle chat input
            query = st.chat_input(placeholder=self.translations[self.current_language]['input_placeholder'])
            if query:
                self.chat_ai.chat_handler.handle_chat(query)
                
                # Clear the flag after handling chat
                st.session_state.content_processed_displayed = False

    def _handle_url_chat(self):
        url = st.text_input(self.translations[self.current_language]['url_input'])
        
        if url:
            # Initialize ChatHandler if not already done
            if not self._initialize_chat_handler():
                return
            
            # Process the URL
            self.chat_ai.document_processor.process_url(url)
            
            # Display content processed message if it hasn't been displayed yet
            if not st.session_state.get('content_processed_displayed', False):
                st.success(self.translations[self.current_language]['content_processed'])
                st.session_state.content_processed_displayed = True

            # Handle chat input
            query = st.chat_input(placeholder=self.translations[self.current_language]['input_placeholder'])
            if query:
                self.chat_ai.chat_handler.handle_chat(query)
                
                # Clear the flag after handling chat
                st.session_state.content_processed_displayed = False

    def _handle_chat_input(self):
        """Handle chat input from the user and display response."""
        user_input = st.text_input(self.translations[self.current_language]['input_placeholder'])
        if st.button("Send"):
            if not self.chat_ai.search_agent:
                st.warning(self.translations[self.current_language]['error_message'].format("Search agent is not initialized."))
                return

            if not self.chat_ai.chat_handler:
                self.chat_ai.chat_handler = ChatHandler(st.session_state.vectors)
                if not self.chat_ai.chat_handler:
                    st.warning(self.translations[self.current_language]['error_message'].format("Failed to initialize ChatHandler."))
                    return

            st.session_state.chat_histories[st.session_state.chat_option].append({"role": "user", "content": user_input})
            self.chat_ai.chat_handler._display_chat_history()
            self._display_chat_response()

    def _display_chat_response(self):
        with st.chat_message("assistant"):
            st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
            try:
                if not self.chat_ai.search_agent:
                    st.warning(self.translations[self.current_language]['error_message'].format("Search agent is not initialized."))
                    return

                response = self.chat_ai.search_agent.run(
                    st.session_state.chat_histories[st.session_state.chat_option],
                    callbacks=[st_cb]
                )

                st.session_state.chat_histories[st.session_state.chat_option].append({'role': 'assistant', "content": response})

                # Ensure response translation based on the current language
                if isinstance(response, str):
                    st.write(response)
                elif isinstance(response, list):
                    content = [entry['content'] for entry in response if entry.get('role') == 'assistant']
                    if content:
                        st.write(content[0])
                    else:
                        st.write(self.translations[self.current_language]['no_content_found'])
                else:
                    st.write(self.translations[self.current_language]['unexpected_format'])

            except Exception as e:
                st.error(self.translations[self.current_language]['error_message'].format(e))
                
    def _display_chatbot_summary(self):
        """Display the summary of the selected chatbot."""
        chat_option = st.session_state.chat_option
        summary = self.translations[self.current_language]['chatbot_summaries'].get(chat_option, "Select a chatbot to see its summary.")
        st.markdown(f"### {chat_option}")
        st.write(summary)

    def run_streamlit_app(self):
        """Run the Streamlit app interface."""
        StreamlitInterface(self).render_app()

# Streamlit app execution
if __name__ == "__main__":
    interface = StreamlitInterface(BharatChatAI())
    interface.render_app()

