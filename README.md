# Qforia: Query Fan-Out Simulator for AI Surfaces

🔍 **Qforia** is a Streamlit application that simulates Google's AI Mode query fan-out process for generative search systems. It uses Google's Gemini AI to generate multiple related queries from a single input query, helping you explore different search angles and user intents.

## Features

### 🎯 Single Query Processing
- **Interactive Query Input**: Enter a single query and get multiple related queries
- **Two Search Modes**: 
  - **AI Overview (simple)**: Generates 10+ focused queries
  - **AI Mode (complex)**: Generates 20+ comprehensive queries
- **Real-time Processing**: Uses Gemini AI for intelligent query expansion
- **Detailed Analysis**: Shows model reasoning and generation statistics

### 📊 Batch Processing (NEW!)
- **CSV Upload**: Upload a CSV file with multiple queries
- **Automatic Validation**: Validates CSV format and content
- **Progress Tracking**: Real-time progress bar during batch processing
- **Combined Results**: Single CSV download with all generated queries
- **Summary Statistics**: Overview of processing results

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd qforia
   ```

2. **Install dependencies**:
   ```bash
   cd streamlit
   pip install -r requirements.txt
   ```

3. **Get a Gemini API Key**:
   - Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Create a new API key
   - Copy the key for use in the application

## Usage

### Running the Application

```bash
cd streamlit
streamlit run qforia.py
```

The application will open in your browser at `http://localhost:8501`.

### Single Query Mode

1. **Enter your Gemini API Key** in the sidebar
2. **Select Search Mode**: Choose between "AI Overview (simple)" or "AI Mode (complex)"
3. **Enter your query** in the text area
4. **Click "Run Fan-Out 🚀"** to generate related queries
5. **View results** in the interactive dataframe
6. **Download results** as a CSV file

### Batch Processing Mode

1. **Switch to "Batch Processing (CSV Upload)"** mode
2. **Upload a CSV file** containing your queries
3. **Review the preview** of uploaded queries
4. **Click "Process Batch Queries 🚀"** to process all queries
5. **Monitor progress** with the real-time progress bar
6. **View summary statistics** and combined results
7. **Download the combined CSV** with all generated queries

### CSV Format Requirements

For batch processing, your CSV file should:
- Contain at least one column with queries
- Have column names like: `query`, `queries`, `keyword`, `keywords`, `text`, `input`, `prompt`
- Include one query per row
- Empty rows are automatically filtered out

**Example CSV format**:
```csv
query
What's the best electric SUV for driving up mt rainier?
How to make the perfect sourdough bread?
Best hiking trails in Colorado
```

## Output Format

The application generates queries with the following structure:

| Column | Description |
|--------|-------------|
| `query` | The generated related query |
| `type` | Query type (reformulation, related, implicit, comparative, entity expansion, personalized) |
| `user_intent` | The inferred user intent |
| `reasoning` | Explanation of why this query was generated |
| `original_query` | The input query (batch mode only) |
| `query_index` | Position in batch (batch mode only) |

## Example Output

For the query "What's the best electric SUV for driving up mt rainier?", the app might generate:

- "Electric SUVs with best mountain driving performance"
- "Range of electric vehicles for mountain terrain"
- "Charging stations near Mount Rainier"
- "Best electric SUVs for cold weather and elevation"
- "Electric vehicle battery performance in mountains"

## Technical Details

- **AI Model**: Google Gemini 1.5 Flash
- **Framework**: Streamlit
- **Data Processing**: Pandas
- **Query Types**: 6 different transformation types for comprehensive coverage

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Make your changes
4. Commit your changes: `git commit -m 'Add feature'`
5. Push to the branch: `git push origin feature/your-feature`
6. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 
