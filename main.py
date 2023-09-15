# Import necessary libraries
import tempfile
import streamlit as st
from langchain.agents import initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain.chains.conversation.memory import ConversationBufferWindowMemory

from tools import ImageCaptionTool, ObjectDetectionTool

# Initialize tools for processing images
image_caption_tool = ImageCaptionTool()
object_detection_tool = ObjectDetectionTool()
tools = [image_caption_tool, object_detection_tool]

# Initialize conversational memory
conversational_memory = ConversationBufferWindowMemory(
    memory_key='chat_history',
    k=10,
    return_messages=True
)

# Initialize ChatOpenAI model
openai_api_key = "#PUT YOUR OPEN-AI API KEY HERE"
llm = ChatOpenAI(
    openai_api_key=openai_api_key,
    temperature=0.01,
    model_name="gpt-3.5-turbo"
)

# Initialize agent for chat interaction
agent = initialize_agent(
    agent="chat-conversational-react-description",
    tools=tools,
    llm=llm,
    max_iterations=8,
    verbose=True,
    memory=conversational_memory,
    early_stopping_memory='generate'
)

# Set up the Streamlit web application
st.title("Ask Questions About Your Image")
st.header("Upload Your Image")

# Upload an image file
file = st.file_uploader("Choose an image file", type=["jpeg", "png", "gif", "bmp", "tiff", "webp", "svg", "heic", "raw", "ico"])

if file:
    # Display the uploaded image
    st.image(file, use_column_width=True)
    
    # Text Input for user's question
    question = st.text_input("Ask a question about the image")

    # Compute the agent's response
    with tempfile.NamedTemporaryFile(dir='.') as temp_file:
        temp_file.write(file.getbuffer())
        image_path = temp_file.name
        if question and question != "":
            with st.spinner(text="Computing..."):
                response = agent.run('{}, this is the image path: {}'.format(question, image_path))
                st.write(response)
