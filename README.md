# Financials

A financial data analysis and visualization application.

## High Level Description of the System

This system is composed of four components: 
* a RAG system (with a Weaviate database)
* a chat system composed of:
  * a very basic UI powered by Streamlit
  * an "OptimizedFinancialExtractor" class that extracts and presents financial data
* a CLI "finanalyze" that loads company data files into the RAG system

## Hight Level Description of the Flow

1. load all company data into the RAG
2. given a user query, extract chunks relevant to the query
3. use the "OptimizedFinancialExtractor" to extract financial data (semanticaly):
    *  values (take into account unit "million USD", etc..)
    *  unit (USD, etc...)
    *  entity (revenue, income, etc...)
    *  date range
4. format reponse in csv files, [example output](output/)
5. format responce and update prompt

## Development Environment

### Quick Start

1. Clone the repository
2. export OPENAI_API_KEY="your-api-key"
3. Start docker-compose
4. Process a PDF file
5. Open the UI at http://localhost:8501

```bash
git clone git@github.com:Red64llc/financials.git
cd financials
docker compose up -d
docker compose exec financials-ui uv run finanalyze process ./data/goog-10-k-2024.pdf
```

#### Using Docker

##### Running with Docker Compose

To start all services (Financials app and Weaviate database):

```bash
# From the project root directory
docker-compose up
```

This will:
- Build the Financials application image if it doesn't exist
- Start the Financials service on port 8501
- Start the Weaviate vector database on port 8080
- Set up the necessary volumes and network connections

To run in detached mode (background):

```bash
docker-compose up -d
```

To stop all services:

```bash
docker-compose down
```

##### Calling CLI from Docker

Running the CLI using the installed command:
```bash
# From the project root directory
➜  financials git:(main) ✗ docker compose exec financials-ui uv run finanalyze process -h
usage: finanalyze process [-h] pdf_path

positional arguments:
  pdf_path              Path to the PDF file

optional arguments:
  -h, --help            show this help message and exit
```

###### Processing Files from Host Machine

To process a PDF file that exists on your host machine:

```bash
# Place your PDF file in the data directory
# Then run the command (assuming your PDF is at data/example.pdf):
docker-compose exec financials-cli finanalyze process /app/data/example.pdf
```

## Screenshots

### Loading Models
The first time starting the Docker instance and starting the UI will load all the models. This can take a few minutes. 
![Screenshot](/screenshots/loading-models.png)

### Chat Loaded Successfully
Once the models are loaded, the UI will show a success message and the chat interface will be ready to use.
If you have loaded a PDF or several PDFs, you will see the number of entries in the collection on the sidebar.
![Screenshot](/screenshots/chat-loaded-successfully.png)

### Sample Response 1
By opening the log from Docker as you are chatting, you can see the chunks that are being extracted from the Rack system on the law in the log. 
![Screenshot](/screenshots/sample-response-1.png)

### Sample Response 2
The system first retrieves chunks that are relevant and then processes them to extract and format the information into financial information.
That involves trying to extract or understand semantic dates, entities, currency, and values. 
You can see the reasoning on the Docker Compose log. 
![Screenshot](/screenshots/sample-response-2.png)

### Extracted Financial Information
The system will extract the financial information and save it as csv file in the output directory.
![Screenshot](/screenshots/extracted-financial-information.png)
