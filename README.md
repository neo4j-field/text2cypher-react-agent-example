# Text2Cypher ReAct Agent Example

A conversational AI agent that connects to a Neo4j Movies database and can answer movie-related questions using natural language. Built with LangGraph's implementation of a ReAct agent, the Neo4j Cypher MCP server, and a custom movie recommendations tool.

## Features

- **Natural Language to Cypher**: Ask questions in plain English and get answers from your Neo4j database
- **ReAct Agent Pattern**: Uses reasoning and acting loops for complex reasoning
- **Schema-Aware**: Automatically retrieves and uses database schema for accurate query generation
- **Interactive CLI**: Chat-based interface for easy interaction

## Prerequisites

- Python 3.10 or higher
- Neo4j Aura account or local Neo4j instance with Movies database
- OpenAI API key
- `uv` package manager (recommended) or `pip`

## Installation

### Option 1: Using uv (Recommended)

1. **Install uv** (if not already installed):

    [Install Documentation](https://docs.astral.sh/uv/getting-started/installation/)
   ```bash
   pip install uv
   ```

2. **Clone and setup the project**:
   ```bash
   git clone neo4j-field/text2cypher-react-agent-example
   cd text2cypher-react-agent-example
   ```

3. **Install dependencies**:
   ```bash
   uv sync
   ```

### Option 2: Using pip

1. **Clone and setup the project**:
   ```bash
   git clone neo4j-field/text2cypher-react-agent-example
   cd text2cypher-react-agent-example
   ```

2. **Create and activate virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

1. **Copy the example environment file**:
   ```bash
   cp .env.example .env
   ```

2. **Edit `.env` with your credentials**:
   ```bash
   OPENAI_API_KEY=your_openai_api_key_here
   NEO4J_USERNAME=neo4j
   NEO4J_PASSWORD=your_neo4j_password
   NEO4J_URI=neo4j+s://your-instance.databases.neo4j.io
   NEO4J_DATABASE=neo4j
   ```

## Agent Components

### Core Agent (`agent.py`)
- **LangGraph ReAct Agent**: Implements reasoning and acting loops for complex queries
- **Neo4j Cypher MCP Server**: Provides schema introspection and query execution
- **Custom Recommendations Tool**: Custom tool for movie recs
- **Interactive CLI**: Command-line chat interface

### Key Tools Available to the Agent

1. **`get_neo4j_schema`**: Retrieves database schema for informed query writing
2. **`read_neo4j_cypher`**: Executes read-only Cypher queries against the database  
3. **`find_movie_recommendations`**: Custom recommendation engine that finds movies liked by users who also enjoyed a target movie

## Running the Agent

### Using uv (Recommended)
```bash
make run-agent-uv
# or
uv run python3 agent.py
```

### Using pip/standard Python
```bash
make run-agent
# or
python3 agent.py
```

## Usage Examples

Once running, you can ask questions like:

**Example Queries:**
- "What movies are in the database?"
- "Tell me about The Matrix"
- "Recommend me some films like The Dark Knight."

## Exit Commands
To exit the agent, type any of:
- `exit`
- `quit` 
- `q`

## Development

### Code Formatting
```bash
make format
```

```

## Dependencies

**Core Libraries:**
- `langchain` - LangChain framework
- `langchain-mcp-adapters` - MCP (Model Context Protocol) adapters
- `langchain-openai` - OpenAI integration
- `langgraph` - Graph-based agent framework
- `neo4j` - Neo4j Python driver
- `openai` - OpenAI API client
- `pydantic` - Data validation

**Development:**
- `ruff` - Code formatting and linting

## Troubleshooting

**Connection Issues:**
- Verify your Neo4j credentials in `.env`
- Ensure your Neo4j instance is running and accessible

**OpenAI Issues:**
- Verify your OpenAI API key is valid
- Check your API usage limits

**MCP Server Issues:**
- Ensure `uvx` is available in your PATH
- The agent automatically installs `mcp-neo4j-cypher@0.3.0` via uvx

**Python Issues:**
- Ensure Python 3.10+ is installed
- Try recreating your virtual environment if using pip

## License

This project is provided as an example for educational and demonstration purposes.
