# Multimodal Visual AI Chatbot

[![GitHub stars](https://img.shields.io/github/stars/PrachiPatel15/Multimodal-Visual-AI-Chatbot)](https://github.com/PrachiPatel15/Multimodal-Visual-AI-Chatbot/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/PrachiPatel15/Multimodal-Visual-AI-Chatbot)](https://github.com/PrachiPatel15/Multimodal-Visual-AI-Chatbot/network)
[![GitHub issues](https://img.shields.io/github/issues/PrachiPatel15/Multimodal-Visual-AI-Chatbot)](https://github.com/PrachiPatel15/Multimodal-Visual-AI-Chatbot/issues)

A sophisticated Streamlit application that performs comprehensive image analysis using multiple vision models and engages users in natural conversation about visual content.

![Multimodal Visual AI Chatbot Demo](https://raw.githubusercontent.com/PrachiPatel15/Multimodal-Visual-AI-Chatbot/main/assets/demo.png)

## 🌟 Features

- **Dual Model Image Analysis**: Leverages both BLIP and ViT-GPT2 models to provide comprehensive and diverse perspectives on image content
- **Interactive Chat Experience**: Engage in natural conversation about the visual content of uploaded images
- **In-depth Visual Understanding**: Automatically extracts key information through a set of predefined analytical questions
- **GPU Acceleration**: Utilizes CUDA when available for significantly faster processing
- **LLM-powered Responses**: Generates human-like, contextually relevant responses using Together AI's Mistral model
- **User-friendly Interface**: Clean Streamlit UI with separate areas for image upload and conversation

## 📸 Screenshots

<table>
  <tr>
    <td><img src="https://raw.githubusercontent.com/PrachiPatel15/Multimodal-Visual-AI-Chatbot/main/assets/upload_screen.png" alt="Upload Interface" width="100%"></td>
    <td><img src="https://raw.githubusercontent.com/PrachiPatel15/Multimodal-Visual-AI-Chatbot/main/assets/analysis_complete.png" alt="Analysis Complete" width="100%"></td>
  </tr>
  <tr>
    <td><img src="https://raw.githubusercontent.com/PrachiPatel15/Multimodal-Visual-AI-Chatbot/main/assets/chat_example.png" alt="Chat Example" width="100%"></td>
    <td><img src="https://raw.githubusercontent.com/PrachiPatel15/Multimodal-Visual-AI-Chatbot/main/assets/detailed_response.png" alt="Detailed Response" width="100%"></td>
  </tr>
</table>

## 🔧 Technical Architecture

![Architecture Diagram](https://raw.githubusercontent.com/PrachiPatel15/Multimodal-Visual-AI-Chatbot/main/assets/architecture.png)

### Components

1. **BLIP Model**: Provides detailed image captioning and visual question-answering capabilities
2. **ViT-GPT2 Model**: Offers complementary image understanding through a different architectural approach
3. **Standard Question Analysis**: Extracts consistent information across all images through predefined questions
4. **Together AI Integration**: Uses Mistral 7B model for generating conversational responses
5. **Streamlit Interface**: Handles user interactions, image uploads, and displays chat history

## 🚀 Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (optional but recommended)
- Together AI API key

### Step-by-Step Setup

1. **Clone the repository**:
```bash
git clone https://github.com/PrachiPatel15/Multimodal-Visual-AI-Chatbot.git
cd Multimodal-Visual-AI-Chatbot
```

2. **Create and activate a virtual environment**:
```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n visual-chatbot python=3.8
conda activate visual-chatbot
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**:
   - Create a `.env` file in the project root
   - Add your Together AI API key:
   ```
   TOGETHER_API_KEY=your_api_key_here
   ```

5. **Download model weights (optional)**:
   - The models will be downloaded automatically on first run
   - To pre-download and cache them:
   ```python
   from transformers import BlipProcessor, BlipForConditionalGeneration, VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
   
   # Download BLIP
   BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
   BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
   
   # Download ViT-GPT2
   VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
   ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
   AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
   ```

## 🔍 How It Works

1. **Dual-Model Processing**: Both BLIP and ViT-GPT2 generate diverse perspectives on the uploaded image.
2. **Standard Question Analysis**: Consistent data extraction using six predefined questions.
3. **Together AI Integration**: Uses Mistral 7B for enhanced conversational ability based on image context.

## 💪 Contributing

Contributions are welcome! Feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 🙏 Acknowledgements

- [Salesforce BLIP](https://github.com/salesforce/BLIP) for their powerful vision-language model
- [NLP Connect](https://huggingface.co/nlpconnect) for the ViT-GPT2 image captioning model
- [Together AI](https://together.ai/) for their Mistral model API
- [Streamlit](https://streamlit.io/) for the intuitive web application framework
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/index) for model implementations

## 📧 Contact

Prachi Patel - [@PrachiPatel15](https://github.com/PrachiPatel15)
