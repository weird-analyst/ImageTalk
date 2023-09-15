# Interactive-Image-Communication-Web-App

## Overview
In this project, I have developed an innovative web application that leverages cutting-edge technologies to facilitate communication with images. The Interactive Image Communication Web App is powered by Langchain, a sophisticated language model, and harnesses the capabilities of Hugging Face's Transformers for image captioning and object recognition. Additionally, it integrates OpenAI's API to establish seamless interactions with images.

With this web app, users can simply upload an image, and the application will provide descriptive captions for the image's content and recognize objects within it. Furthermore, the Langchain-powered conversational agent allows users to engage in a natural dialogue with the uploaded image. This project not only showcases the potential of AI and language models but also opens up new possibilities for interactive image-based communication.

## Project Structure
- `main.py`: The main Python script for the Streamlit web application.
- `tools.py`: Contains two essential tools: the Image Caption Generator and the Object Detection Tool.
- `requirements.txt`: Lists the project's dependencies.

## Setup and Usage
1. Clone the repository to your local machine:

   ```shell
   git clone https://github.com/weird-analyst/Interactive-Image-Communication-Web-App
   ```
2. Install the project dependencies:

   ```shell
    pip install -r requirements.txt
   ```
3. Run the Streamlit web application:

    ```shell
    streamlit run main.py
    ```
4. Access the web application by opening the provided URL in your web browser.

## Features
### Image Caption Generator

    This tool generates a short caption for the uploaded image.
    It utilizes the transformers library and the "Salesforce/blip-image-captioning-large" model.
    The generated caption provides a brief description of the image.

### Object Detection Tool

    This tool detects objects in the uploaded image.
    It utilizes the transformers library and the "facebook/detr-resnet-50" model.
    Detected objects are listed in the format "[x1, y1, x2, y2] class_name confidence_score."

### ChatGPT Interaction

    Users can upload an image and ask questions about it.
    ChatGPT, integrated with LangChain, processes user queries and provides responses.
    Responses are displayed in the web application, facilitating interactive communication with the image.

## Configuration

    Modify the openai_api_key variable in main.py with your OpenAI API key.
    Adjust model parameters and settings in the ChatOpenAI initialization to meet your requirements.

## Dependencies

    LangChain: 0.0.171
    Streamlit: 1.22.0
    OpenAI: 0.27.6
    Tabulate: 0.9.0
    Timm: 0.9.2
    Transformers: (latest version)

## Contributing

Contributions to this project are welcome! Feel free to fork the repository, make improvements, and submit pull requests.
## License

This project is licensed under the MIT License.

## Acknowledgments

    LangChain: https://github.com/langchain
    Streamlit: https://streamlit.io/
    OpenAI: https://beta.openai.com/
    Transformers: https://github.com/huggingface/transformers

Happy communicating with images!



Replace `"yourusername"` and `"your-repo-name"` with your GitHub username and repository
