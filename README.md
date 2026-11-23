# RAG with Amazon Bedrock and Amazon OpenSearch Serverless

A Retrieval-Augmented Generation (RAG) application built on AWS. It answers questions grounded in your own documents by retrieving relevant chunks from an Amazon Bedrock Knowledge Base (backed by OpenSearch Serverless) and passing them to Amazon Nova Lite for generation.

In the demo, the knowledge base contains Amazon's Q1 2026 earnings release. A plain LLM cannot answer "What is Amazon's net income in 2026?", but the RAG pipeline answers correctly: **$30.3 billion for Q1 2026**.

## Architecture

```
User Question
      |
      v
Embed question (Titan Text Embeddings V2)
      |
      v
Vector search (Amazon OpenSearch Serverless)
      |
      v
Retrieve top relevant chunks (Bedrock Retrieve API)
      |
      v
Context + Question --> Amazon Nova Lite (Invoke Model API)
      |
      v
Grounded answer
```

### AWS services used

| Service | Role |
|---|---|
| Amazon Bedrock | Foundation model (Amazon Nova Lite), embeddings (Titan Text Embeddings V2), Knowledge Base |
| Amazon OpenSearch Serverless | Vector database for embeddings |
| Amazon S3 | Stores the source documents |
| Boto3 (Python SDK) | Calls the Bedrock APIs |

## Why RAG?

- Reduces hallucinations by grounding answers in retrieved documents
- Uses up-to-date private data without retraining the model
- Knowledge is updated simply by adding documents and re-syncing

## Prerequisites

- An AWS account (Free Tier is fine to start)
- Python 3.9+ (tested on 3.12)
- Jupyter Notebook or JupyterLab
- AWS credentials configured locally (`aws configure`) for an IAM user
- Model access enabled in Bedrock for **Amazon Nova Lite** and **Titan Text Embeddings V2** (region `us-east-1`)

```bash
pip install boto3 jupyter
```

## Setup

1. **Create an AWS account** and sign in.
2. **Create an IAM user.** Do not use the root user for development.
3. **Grant permissions** to the IAM user:
   - `AmazonBedrockFullAccess`
   - `AmazonS3FullAccess`
   - `AmazonOpenSearchServiceFullAccess`
   - Optional: `IAMFullAccess`, `CloudWatchLogsFullAccess`

   > For learning, `AdministratorAccess` also works. In production, follow the principle of least privilege.
4. **Log in as the IAM user** and open the Amazon Bedrock console.
5. **Create a Knowledge Base:**
   - Vector store: Amazon OpenSearch Serverless
   - Embedding model: Titan Text Embeddings V2
   - Data source: a new or existing S3 bucket with your documents
   - Sync the data source
6. **Verify the Knowledge Base:** confirm documents are indexed and embeddings generated, then test retrieval in the Bedrock console.
7. **Copy the Knowledge Base ID** and replace `JWA3IE8GLM` in the notebook with your own.

## Usage

Open the notebook and run the cells in order:

```bash
jupyter notebook "RAG-Amazon Bedrock.ipynb"
```

The notebook walks through four stages:

1. **Build a client.** Create a `bedrock-runtime` client and set the model ID `amazon.nova-lite-v1:0`.
2. **Baseline LLM call.** Ask the model without any context. It cannot answer about 2026 financials.
3. **Retrieve.** Use the `bedrock-agent-runtime` client's `retrieve` API to fetch relevant chunks from the Knowledge Base.
4. **Generate.** Join the chunks into a context, build a prompt, and call `invoke_model` on Nova Lite.

### Core pipeline

```python
import boto3, json

kb_client = boto3.client("bedrock-agent-runtime", region_name="us-east-1")
runtime_client = boto3.client("bedrock-runtime", region_name="us-east-1")

query = "What is the net income of Amazon in 2026?"

# 1. Retrieve
retrieved = kb_client.retrieve(
    knowledgeBaseId="<YOUR_KNOWLEDGE_BASE_ID>",
    retrievalQuery={"text": query},
)
context = "\n\n".join(r["content"]["text"] for r in retrieved["retrievalResults"])

# 2. Build prompt
prompt = f"""You are a helpful assistant.

Context:
{context}

Question:
{query}

Answer based only on the context above."""

# 3. Generate
body = {"messages": [{"role": "user", "content": [{"text": prompt}]}]}
response = runtime_client.invoke_model(
    modelId="amazon.nova-lite-v1:0",
    body=json.dumps(body),
)
result = json.loads(response["body"].read())
print(result["output"]["message"]["content"][0]["text"])
```

## The two Bedrock APIs

| API | Purpose |
|---|---|
| **Retrieve** (`bedrock-agent-runtime`) | Returns the most relevant document chunks from the Knowledge Base. Does not generate answers. |
| **Invoke Model / Converse** (`bedrock-runtime`) | Sends context plus the question to the LLM and returns the generated answer. |

## Example output

Without RAG:

> I'm unable to provide real-time or future financial data, including the net income of Amazon in 2026...

With RAG:

> The net income of Amazon in the first quarter of 2026 was $30.3 billion.

## Project structure

```
.
├── RAG-Amazon Bedrock.ipynb   # End-to-end walkthrough
├── images/                    # Architecture diagrams used in the notebook
└── README.md
```

## Notes and tips

- Never commit AWS access keys or hard-code credentials. Use `aws configure` or environment variables.
- Delete the OpenSearch Serverless collection and Knowledge Base when finished. OpenSearch Serverless is billed while it exists and is not covered by the Free Tier.
- Use the `converse` API for a simpler, model-agnostic request format. The notebook uses `invoke_model` for the final step.
- Retrieval quality depends on chunking and embedding settings. Tune them in the Knowledge Base data source if answers look incomplete.

## Possible extensions

- Use `retrieve_and_generate` for a single-call RAG flow
- Add Bedrock Guardrails for safety filtering
- Return source citations alongside answers
- Wrap the pipeline in a Streamlit or Flask UI

## License

Add your preferred license (for example, MIT).
