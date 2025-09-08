# MiniDr — AI Health Assistant

MiniDr is an AI-powered health assistant built with Streamlit and Groq. It supports general chat, medical Q&A, and medical image analysis (CT, X-ray, skin photos) using state-of-the-art LLMs.

## Features

- **General Chat:** Fast, multilingual chat using Llama models.
- **Medical Chat:** Answers health-related questions with safety notes and actionable advice (not diagnostic).
- **Vision Analysis:** Upload medical images for AI-generated observations and guidance.
- **Language Support:** Responds in Hindi, English, or Hinglish, matching user input.
- **Model Fallbacks:** Automatically switches models if the primary is unavailable.

## Project Structure

```
MiniDr.-main/
    frontend.py                # Streamlit app entry point
    MiniDr/
        frontend.py            # Main Streamlit UI and logic
        main.py                # CLI entry point (prints hello)
        pyproject.toml         # Project metadata and dependencies
        README.md              # Project documentation
        requirements.txt       # Python dependencies
        uv.lock                # Dependency lock file
```

## Getting Started

### Prerequisites

- Python 3.13+
- [Groq API key](https://groq.com/)
- [Streamlit](https://streamlit.io/)

### Installation

1. Clone the repository:

    ```sh
    git clone <repo-url>
    cd MiniDr.-main
    ```

2. Install dependencies:

    ```sh
    pip install -r MiniDr/requirements.txt
    ```

3. Set up your Groq API key:

    - Create a `.env` file in `MiniDr/` with:
      ```
      GROQ_API_KEY=your_groq_api_key_here
      ```

### Running the App

Start the Streamlit app:

```sh
streamlit run frontend.py
```
or
```sh
streamlit run MiniDr/frontend.py
```

## Usage

- **Chat Tab:** Type questions in Hindi, English, or Hinglish. The app routes prompts to the appropriate model.
- **Vision Tab:** Upload a medical image and enter a prompt for AI analysis.

## Disclaimer

> ⚠️ MiniDr is **not a medical device** and does not provide diagnosis or treatment. For emergencies or concerning symptoms, contact a qualified healthcare professional.

## License

MIT License

## Acknowledgements

- [Groq](https://groq.com/)
- [Streamlit](https://streamlit.io