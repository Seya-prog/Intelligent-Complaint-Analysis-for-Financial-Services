# Intelligent Complaint Analysis for Financial Services

## Project Overview
This project implements a Retrieval-Augmented Generation (RAG) powered chatbot that helps financial institutions analyze and understand customer complaints. The system provides an intuitive ChatGPT-like interface for querying and analyzing complaint data, ensuring responses are grounded in actual complaint records.

## Key Features
- Natural language querying of complaint data
- Real-time response streaming
- Source citation for transparency
- Clean, modern ChatGPT-like interface
- Conversation history management
- Modular RAG pipeline architecture

## Project Structure
- **data/**: Contains complaint datasets and vector stores
- **notebooks/**: Jupyter notebooks for exploratory data analysis
- **src/**: Core implementation
  - `text_processing/`: Text chunking and embedding utilities
  - `rag/`: RAG pipeline implementation
  - `evaluate_rag.py`: System evaluation tools
  - `process_complaints.py`: Complaint data processing
- **test/**: Test suite for system components

## Getting Started

### Prerequisites
- Python 3.11+
- Git
- Virtual environment (recommended)

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/Intelligent-Complaint-Analysis-for-Financial-Services.git
   cd Intelligent-Complaint-Analysis-for-Financial-Services
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Application
1. Start the Streamlit interface:
   ```bash
   streamlit run app.py
   ```

2. Open your browser and navigate to the displayed URL (typically http://localhost:8501)

## Technical Architecture

### RAG Pipeline
The system uses a modular RAG architecture:
1. **Text Processing**: Chunks documents and generates embeddings
2. **Retrieval**: Finds relevant complaint examples using vector similarity
3. **Generation**: Produces contextual responses using retrieved information

### User Interface
Built with Streamlit, featuring:
- Clean, minimalist design
- Real-time response streaming
- Conversation history management
- Source citation with expandable details

## Contributing
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License
This project is licensed under the MIT License - see the LICENSE file for details. 