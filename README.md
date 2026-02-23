# Biomedical GraphRAG

![Neo4j UI](static/image.png)

<div align="center">

<!-- Project Status -->

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python version](https://img.shields.io/badge/python-3.13.8-blue.svg)](https://www.python.org/downloads/)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)

<!-- Providers -->

[![Qdrant](https://img.shields.io/badge/Qdrant-1.15.1-5A31F4?logo=qdrant&logoColor=white)](https://qdrant.tech/)
[![Neo4j](https://img.shields.io/badge/Neo4j-5.28.2-008CC1?logo=neo4j&logoColor=white)](https://neo4j.com/)
[![OpenAI](https://img.shields.io/badge/OpenAI-2.3.0-412991?logo=openai&logoColor=white)](https://openai.com/)

</div>

## Table of Contents

- [Biomedical GraphRAG](#biomedical-graphrag)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Project Structure](#project-structure)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Usage](#usage)
    - [Configuration](#configuration)
    - [Data Collection](#data-collection)
    - [Infrastructure Setup](#infrastructure-setup)
      - [Neo4j Graph Database](#neo4j-graph-database)
      - [Qdrant Vector Database](#qdrant-vector-database)
    - [Query Commands](#query-commands)
      - [Qdrant Vector Search](#qdrant-vector-search)
      - [Hybrid Neo4j + Qdrant Queries](#hybrid-neo4j--qdrant-queries)
      - [Available Query Types](#available-query-types)
      - [Sample Queries](#sample-queries)
    - [Testing](#testing)
    - [Quality Checks](#quality-checks)
  - [License](#license)

## Overview

A comprehensive GraphRAG (Graph Retrieval-Augmented Generation) system designed for biomedical research. It combines knowledge graphs with vector search to provide intelligent querying and analysis of biomedical literature and genomic data.

**Video:**

<div align="center">

<a href="https://www.youtube.com/watch?v=3NWTi90i6C4&t=200s" target="_blank">
  <img src="images/pubmed_navigator.jpeg" width="600" alt="YouTube Video">
</a>

</div>

&nbsp;

**Article:**

 [Building a Biomedical GraphRAG: When Knowledge Graphs Meet Vector Search](https://aiechoes.substack.com/p/building-a-biomedical-graphrag-when)

**Key Features:**

- **Hybrid Query System**: Combines Neo4j graph database with Qdrant vector search for comprehensive biomedical insights
- **Data Integration**: Processes PubMed papers, gene data, and research citations
- **Intelligent Querying**: Uses LLM-powered tool selection for graph enrichment and semantic search
- **Biomedical Schema**: Specialized graph schema for papers, authors, institutions, genes, and MeSH terms
- **Async Processing**: High-performance async data collection and processing

## Project Structure

```text
biomedical-graphrag-pipeline/
├── .github/                    # GitHub workflows and templates
├── data/                       # Dataset storage (PubMed, Gene data)
├── docs/                       # Documentation
├── src/
│   └── biomedical_graphrag/
│       ├── application/        # Application layer
│       │   ├── cli/            # Command-line interfaces
│       │   └── services/       # Business logic services
│       ├── config.py           # Configuration management
│       ├── data_sources/       # Data collection modules
│       ├── domain/             # Domain models and entities
│       ├── infrastructure/     # Database and external service adapters
│       └── utils/              # Utility functions
├── static/                     # Static assets (images, etc.)
├── tests/                      # Test suite
├── LICENSE                     # MIT License
├── Makefile                    # Build and development commands
├── pyproject.toml              # Project configuration and dependencies
├── README.md                   # This file
└── uv.lock                     # Dependency lock file
```

## Prerequisites

| Requirement                                            | Description                             |
| ------------------------------------------------------ | --------------------------------------- |
| [Python 3.13+](https://www.python.org/downloads/)      | Programming language                    |
| [uv](https://docs.astral.sh/uv/)                       | Package and dependency manager          |
| [Neo4j](https://neo4j.com/)                            | Graph database for knowledge graphs     |
| [Qdrant](https://qdrant.tech/)                         | Vector database for embeddings          |
| [OpenAI](https://openai.com/)                          | LLM provider for queries and embeddings |
| [PubMed](https://www.ncbi.nlm.nih.gov/books/NBK25501/) | Biomedical literature database          |

## Installation

1. Clone the repository:

   ```bash
   git clone git@github.com:benitomartin/biomedical-graphrag.git
   cd biomedical-graphrag
   ```

1. Create a virtual environment:

   ```bash
   uv venv
   ```

1. Activate the virtual environment:

   ```bash
   source .venv/bin/activate
   ```

1. Install the required packages:

   ```bash
   uv sync --all-groups --all-extra
   ```

1. Create a `.env` file in the root directory:

   ```bash
    cp env.example .env
   ```

## Usage

### Configuration

Configure API keys, model names, and other settings by editing the `.env` file:

```bash
# OpenAI Configuration
OPENAI__API_KEY=your_openai_api_key_here
OPENAI__MODEL=gpt-4o-mini
OPENAI__TEMPERATURE=0.0
OPENAI__MAX_TOKENS=1500

# Neo4j Configuration
NEO4J__URI=bolt://localhost:7687
NEO4J__USERNAME=neo4j
NEO4J__PASSWORD=your_neo4j_password
NEO4J__DATABASE=neo4j

# Qdrant Configuration
QDRANT__URL=http://localhost:6333
QDRANT__API_KEY=your_qdrant_api_key
QDRANT__COLLECTION_NAME=biomedical_papers
QDRANT__EMBEDDING_MODEL=text-embedding-3-small
QDRANT__EMBEDDING_DIMENSION=1536

# PubMed Configuration (optional)
PUBMED__EMAIL=your_email@example.com
PUBMED__API_KEY=your_pubmed_api_key

# Data Paths
JSON_DATA__PUBMED_JSON_PATH=data/pubmed_dataset.json
JSON_DATA__GENE_JSON_PATH=data/gene_dataset.json
```

### Data Collection

The system includes data collectors for biomedical and gene datasets:

```bash
# Collect PubMed papers and metadata
make pubmed-data-collector-run
```

```bash
# Collect gene information related to the pubmed dataset
make gene-data-collector-run
```

### Infrastructure Setup

#### Neo4j Graph Database

```bash
# Create the knowledge graph from datasets
make create-graph

# Delete all graph data (clean slate)
make delete-graph
```

#### Qdrant Vector Database

```bash
# Create vector collection for embeddings
make create-qdrant-collection

# Ingest embeddings into Qdrant
make ingest-qdrant-data

# Delete vector collection
make delete-qdrant-collection
```

### Query Commands

#### Qdrant Vector Search

```bash
# Run a custom query on the Qdrant vector store
make custom-qdrant-query QUESTION="Which institutions have collaborated most frequently on papers about 'Gene Editing' and 'Immunotherapy'?"

# Or run directly with the CLI
uv run src/biomedical_graphrag/application/cli/query_vectorstore.py --ask "Which institutions have collaborated most frequently on papers about 'Gene Editing' and 'Immunotherapy'?"
```

#### Hybrid Neo4j + Qdrant Queries

```bash
# Run example queries on the Neo4j graph using GraphRAG
make example-graph-query

# Run a custom natural language query using hybrid GraphRAG
make custom-graph-query QUESTION="What are the latest research trends in cancer immunotherapy?"

# Or run directly with the CLI
uv run src/biomedical_graphrag/application/cli/fusion_query.py "What are the latest research trends in cancer immunotherapy?"
```

#### Available Query Types

**Qdrant Queries:**

- Semantic search across paper abstracts and content
- Similarity-based retrieval using embeddings
- Direct vector similarity queries

**Hybrid Queries:**

- Combines semantic search (Qdrant) with graph enrichment (Neo4j):
  - Author collaboration networks
  - Citation analysis and paper relationships
  - Gene-paper associations
  - MeSH term relationships
  - Institution affiliations
- LLM-powered automatic tool selection

#### Sample Queries

- Who collaborates with Jennifer Doudna on CRISPR research?
  Which researchers work with Emmanuelle Charpentier on gene editing or genome engineering papers?

- Who are George Church’s collaborators publishing on synthetic biology and genome sequencing?

- List scientists collaborating with Feng Zhang on neuroscience studies

- Which papers are related to PMID 31295471 based on shared MeSH terms?

- Find papers similar to the CRISPR-Cas9 genome editing study with PMID 31295471

- Show other studies linked by MeSH terms to PMID 27562951

- Which genes are mentioned in the same papers as gag?

- What genes appear together with HIF1A in cancer research?

- Which genes are frequently co-mentioned with TP53?

### Testing

Run all tests:

```bash
make tests
```

### Quality Checks

Run all quality checks (lint, format, type check, clean):

```bash
make all-check
make all-fix
```

Individual Commands:

- Display all available commands:

  ```bash
  make help
  ```

- Check code static typing

  ```bash
  make mypy
  ```

- Clean cache and build files:

  ```bash
  make clean
  ```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
