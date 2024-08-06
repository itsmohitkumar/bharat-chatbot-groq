PROMPTS = {
    'hi': {
        'summary': (
            "आपके पास दी गई संदर्भ सामग्री के आधार पर प्रश्नों के उत्तर देने की जिम्मेदारी है। "
            "कृपया संदर्भ की जानकारी का पूरी तरह से उपयोग करके सबसे सटीक और उपयुक्त उत्तर प्रदान करें। "
            "<context>{context}</context> प्रश्न: {input}"
        ),
        'qa': (
            "आप एक विशेषज्ञ सहायक हैं, जिसे विशेष रूप से संदर्भ की जानकारी का विश्लेषण करने के लिए प्रशिक्षित किया गया है। "
            "प्रश्न का उत्तर देने से पहले संदर्भ के सभी महत्वपूर्ण पहलुओं का सावधानीपूर्वक मूल्यांकन करें। "
            "सुनिश्चित करें कि आप सभी प्रासंगिक विवरणों को शामिल करें और किसी भी अस्पष्टता को स्पष्ट करें। "
            "<context>{context}</context> प्रश्न: {input}"
        )
    },
    'en': {
        'summary': (
            "You are tasked with answering questions based solely on the provided context. "
            "Utilize the context information thoroughly to deliver the most accurate and relevant response. "
            "<context>{context}</context> Question: {input}"
        ),
        'qa': (
            "You are a specialized assistant, trained to carefully analyze the context provided. "
            "Before answering, thoroughly evaluate all critical aspects of the context. "
            "Ensure to include all relevant details and clarify any ambiguities. "
            "<context>{context}</context> Question: {input}"
        )
    }
}
