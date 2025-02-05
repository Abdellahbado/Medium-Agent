# AI Report Generator

This repository contains an AI-powered report generation workflow using LangChain, LangGraph, and Groq's LLM. The agent automates content planning, research, and writing based on a given topic, it follows this workflow
![image](https://github.com/user-attachments/assets/3bb5f0d8-e905-4f68-a31f-06f19c4e8d5c)


## Features
- **Automated Content Planning**: Generates structured sections for a report.
- **Research Integration**: Uses Tavily search results to incorporate relevant and up-to-date information.
- **LLM-Powered Writing**: Creates detailed report sections with a specified tone.
- **Final Report Compilation**: Ensures a coherent and stylistically consistent output.

## Installation
1. Clone the repository:
   ```sh
   git clone https://github.com/Abdellahbado/Medium-Agent
   cd Medium-Agent
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Set up environment variables in a `.env` file:
   ```
   GROQ_API_KEY=<your_api_key>
   ```

## Usage
Run the workflow with an initial topic:
```python
from agent import workflow

initial_state = {
    "topic": "AI agents vs RL agents: Understanding the difference",
    "sections": [],
    "completed_sections": [],
    "language_tone": "simple, casual",
    "final_report": ""
}

result = workflow.invoke(initial_state)
print(result.get("final_report", "No final report generated."))
```

## Dependencies
- `langchain_groq`
- `langgraph`
- `pydantic`
- `python-dotenv`
- `langchain_community`

## License
This project is licensed under the MIT License.

