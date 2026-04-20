# CSA AI Agent — LangGraph

An AI-powered Customer Support Agent built with [LangGraph](https://github.com/langchain-ai/langgraph) and [LangChain](https://github.com/langchain-ai/langchain). Given a customer query, the agent automatically classifies the category, analyzes sentiment, determines the ITSM request type, and routes the query to the appropriate handler — or escalates it to a human agent when needed.

## How It Works

The agent runs a stateful graph with the following nodes:

```
categorize → analyze_sentiment → classify_request_type → router
                                                             ├── handle_technical
                                                             ├── handle_billing
                                                             ├── handle_general
                                                             └── escalate
```

| Node | Description |
|---|---|
| `categorize` | Classifies query as **Technical**, **Billing**, or **General** |
| `analyze_sentiment` | Detects sentiment: **Positive**, **Neutral**, or **Negative** |
| `classify_request_type` | Maps to ITSM type: **Incident**, **Service Request**, **Change**, or **Problem** |
| `router` | Routes to handler based on sentiment + category; negative sentiment always escalates |
| `handle_technical` | Generates a technical support response |
| `handle_billing` | Generates a billing support response |
| `handle_general` | Generates a general support response |
| `escalate` | Flags query for human agent handoff |

---

### OpenAI API Usage
 
Every node that generates or classifies text (`categorize`, `analyze_sentiment`, `classify_request_type`, and whichever handler is invoked) makes a direct call to the OpenAI API via `ChatOpenAI` from `langchain-openai`. A single query will therefore trigger **4 separate OpenAI API calls** — three for the classification/analysis pipeline and one for the final handler response. Escalated queries only make 3 calls, since the `escalate` node returns a static message with no API call.
 
Be mindful of this when estimating costs or rate-limit usage.
 
---

## Prerequisites

- Python 3.10+
- An [OpenAI API key](https://platform.openai.com/api-keys)

---

## Installation

**1. Clone the repository**

```bash
git clone https://github.com/your-org/csa-ai-langgraph.git
cd csa-ai-langgraph
```

**2. Create and activate a virtual environment**

```bash
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
.venv\Scripts\activate           # Windows
```

**3. Install dependencies**

```bash
pip install langgraph langchain-core langchain-openai python-dotenv
```

**4. Configure environment variables**

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=sk-...
```

---

## Usage

```bash
python csa-agent-langgraph.py "<your customer query>"
```

The agent prints four fields:

| Field | Description |
|---|---|
| `Category` | Technical / Billing / General |
| `Sentiment` | Positive / Neutral / Negative |
| `Request Type` | Incident / Service Request / Change / Problem |
| `Response` | Agent reply or escalation notice |

---

## Examples

### Technical — Incident

```bash
python csa-agent-langgraph.py "My internet connection keeps dropping every few minutes."
```

```
Category:     Technical
Sentiment:    Neutral
Request Type: Incident
Response:     Please restart your router and check for outages in your area...
```

---

### Billing — Service Request

```bash
python csa-agent-langgraph.py "I'd like to upgrade my plan and understand the new pricing."
```

```
Category:     Billing
Sentiment:    Positive
Request Type: Service Request
Response:     Our available plans are... You can upgrade directly from your account dashboard...
```

---

### Negative Sentiment — Escalation

```bash
python csa-agent-langgraph.py "I've been charged twice this month and nobody is helping me fix it!"
```

```
Category:     Billing
Sentiment:    Negative
Request Type: Incident
Response:     Your query has been escalated to a human agent.
```

---

### General — Change Request

```bash
python csa-agent-langgraph.py "Can you update the email address on my account?"
```

```
Category:     General
Sentiment:    Neutral
Request Type: Change
Response:     To update your email address, please go to Account Settings...
```

---

### Out-of-Scope Query — Escalation

```bash
python csa-agent-langgraph.py "Can you write a poem about my dog?"
```

```
Category:     General
Sentiment:    Positive
Request Type: Service Request
Response:     Your query has been escalated to a human agent.
```

---

## Project Structure

```
csa-ai-langgraph/
├── csa-agent-langgraph.py   # Main agent
├── .env                     # Environment variables (not committed)
└── README.md
```

---

## License

MIT
