import nest_asyncio
nest_asyncio.apply()

import streamlit as st
from transformers import (
    VisionEncoderDecoderModel, 
    ViTImageProcessor, 
    AutoTokenizer,
    BlipProcessor, 
    BlipForConditionalGeneration
)
import together
import torch
from PIL import Image
from dotenv import load_dotenv
import json
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)

# Load environment variables
load_dotenv()

class ImprovedVisualChatbot:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Initialize BLIP model for detailed image understanding
        self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
        self.blip_model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-large"
        ).to(self.device)
        
        # Initialize ViT-GPT2 for additional image captioning
        self.vit_gpt2_model = VisionEncoderDecoderModel.from_pretrained(
            "nlpconnect/vit-gpt2-image-captioning"
        ).to(self.device)
        self.vit_gpt2_feature_extractor = ViTImageProcessor.from_pretrained(
            "nlpconnect/vit-gpt2-image-captioning"
        )
        self.vit_gpt2_tokenizer = AutoTokenizer.from_pretrained(
            "nlpconnect/vit-gpt2-image-captioning"
        )
        
        # Initialize session state
        if "messages" not in st.session_state:
            st.session_state.messages = []

    def get_blip_description(self, image: Image) -> str:
        """Get detailed image description using BLIP model"""
        inputs = self.blip_processor(images=image, return_tensors="pt").to(self.device)
        
        # Generate detailed caption
        outputs = self.blip_model.generate(
            **inputs,
            max_length=100,
            num_beams=5,
            temperature=1.0,
            repetition_penalty=1.2,
            length_penalty=1.0
        )
        
        return self.blip_processor.decode(outputs[0], skip_special_tokens=True)

    def get_vit_gpt2_description(self, image: Image) -> str:
        """Get additional perspective using ViT-GPT2 model"""
        pixel_values = self.vit_gpt2_feature_extractor(
            images=image, return_tensors="pt"
        ).pixel_values.to(self.device)
        
        output_ids = self.vit_gpt2_model.generate(
            pixel_values,
            max_length=50,
            num_beams=4,
            temperature=0.8,
            do_sample=True
        )
        
        return self.vit_gpt2_tokenizer.decode(output_ids[0], skip_special_tokens=True)

    def get_visual_qa(self, image: Image, question: str) -> str:
        """Get answer for specific question about the image using BLIP"""
        inputs = self.blip_processor(image, question, return_tensors="pt").to(self.device)
        
        outputs = self.blip_model.generate(
            **inputs,
            max_length=50,
            num_beams=4,
            temperature=0.8,
            do_sample=True
        )
        
        return self.blip_processor.decode(outputs[0], skip_special_tokens=True)

    def analyze_image(self, image: Image) -> dict:
        """Comprehensive image analysis using multiple models"""
        # Get descriptions from both models
        blip_desc = self.get_blip_description(image)
        vit_gpt2_desc = self.get_vit_gpt2_description(image)
        
        # Get answers to predetermined questions for better understanding
        standard_questions = [
            "What is the main subject of this image?",
            "What is the setting or location?",
            "What is the lighting and time of day?",
            "Are there any people in the image?",
            "What activities are happening?",
            "What colors are prominent?"
        ]
        
        qa_results = {}
        for question in standard_questions:
            qa_results[question] = self.get_visual_qa(image, question)
        
        return {
            "blip_description": blip_desc,
            "vit_gpt2_description": vit_gpt2_desc,
            "detailed_analysis": qa_results
        }

    def get_chat_response(self, prompt: str, analysis_results: dict) -> str:
        """Generate response using Together AI's Mistral model"""
        system_prompt = f"""You are an advanced visual AI assistant analyzing an image.
                Image Analysis Results:
        1. Primary Description (BLIP): {analysis_results['blip_description']}
        2. Secondary Description (ViT-GPT2): {analysis_results['vit_gpt2_description']}
        3. Detailed Analysis:
        {json.dumps(analysis_results['detailed_analysis'], indent=2)}
                
        Guidelines:
        1. Use all available descriptions to provide accurate information.
        2. When descriptions differ, mention both perspectives.
        3. If asked about details not covered in the analysis, acknowledge the limitation.
        4. Maintain a natural, conversational tone while being precise.
        5. If there's uncertainty, explain why and what can be confidently stated.
                
        Please respond to the user's query based on this comprehensive analysis.
        """
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        response = together.Complete.create(
            prompt=json.dumps(messages),
            model="mistralai/Mistral-7B-Instruct-v0.2",
            max_tokens=1024,
            temperature=0.7,
            top_k=50,
            top_p=0.7,
            repetition_penalty=1.1
        )
        
        # Ensure clean text output
        if isinstance(response, dict) and 'choices' in response:
            raw_text = response['choices'][0]['text'].strip()
            
            # If the raw text appears to be JSON (starts with { or [)
            if raw_text.startswith('{') or raw_text.startswith('['):
                try:
                    # First, attempt to parse as JSON
                    json_obj = json.loads(raw_text)
                    
                    # Case 1: If it's a list of messages like [{"name": "assistant", ...}]
                    if isinstance(json_obj, list):
                        for item in json_obj:
                            if isinstance(item, dict) and (item.get("role") == "assistant" or item.get("name") == "assistant"):
                                return item.get("content", "Error: Content not found.")
                    
                    # Case 2: If it's a single message object like {"role": "assistant", ...}
                    elif isinstance(json_obj, dict):
                        if "content" in json_obj:
                            return json_obj["content"]
                        elif json_obj.get("role") == "assistant" or json_obj.get("name") == "assistant":
                            return json_obj.get("content", "Error: Content not found.")
                    
                    # If we couldn't extract content but it parsed as JSON, return the stringified pretty version
                    return json.dumps(json_obj, indent=2)
                    
                except json.JSONDecodeError:
                    # Not valid JSON, return the raw text
                    return raw_text
            else:
                # Not JSON format, just return the raw text
                return raw_text
        
        return "Error: Unable to fetch a valid response."

def main():
    st.set_page_config(page_title="Multimodal Visual AI Chatbot", layout="wide")
    st.title("🤖 Multimodal Visual AI Chatbot")

    # Initialize chatbot
    chatbot = ImprovedVisualChatbot()

    # Create sidebar for image upload and analysis details
    with st.sidebar:
        st.header("Upload Image")
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)

            # Analyze image
            if "analysis_results" not in st.session_state:
                with st.spinner("Analyzing image (this may take a moment)..."):
                    analysis_results = chatbot.analyze_image(image)
                    st.session_state.analysis_results = analysis_results

            # Display a message after successful analysis
            st.success("✅ You can now chat with the image!")

    # Main chat interface
    st.header("Chat")

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask about the image..."):
        if "analysis_results" not in st.session_state:
            st.warning("Please upload an image first!")
            return

        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        # Get chatbot response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = chatbot.get_chat_response(
                    prompt,
                    st.session_state.analysis_results
                )

                # Ensure the response is a string (handle list output issue)
                if isinstance(response, list):
                    response = " ".join(response)

                st.write(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
