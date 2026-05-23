# DOS Tech Drive 2026 RAG Knowledge Base

Purpose: clean Markdown document for Dify Knowledge indexing. Use this instead of complex PDF or DOCX files when testing RAG.

Recommended Dify settings:
- Document type: Markdown
- Chunking mode: Automatic or Parent-child
- Retrieval: Vector search or Hybrid search
- Top K: 5
- Score threshold: 0.3 to 0.5

---

## Chunk 1: Workshop Overview

DOS Tech Drive 2026 is a hands-on session about building AI tools and agents using Dify and E2B.

The core agenda is to help participants understand how to build tools, expose those tools to agents, and then connect agents into agent-to-agent workflows.

The main tools used in the session are:
- Dify Cloud
- E2B Sandbox
- Better E2B Sandbox plugin
- Amazon Bedrock
- Web Scraper workflow
- Data Analyst workflow
- A2A Research Agent
- A2A Data Agent
- A2A Agents Orchestrator

The main learning outcome is to understand the difference between a workflow with tools and an autonomous agent with tools.

---

## Chunk 2: Dify App Types

Dify supports several app patterns.

Workflow is a fixed pipeline. The builder decides every step. Workflow is useful when data movement must be predictable and reliable.

Chatflow is a conversational workflow. It keeps a chat-style interface but still follows a designed process.

Agent is an autonomous app. The agent receives a goal and tools. The agent decides which tool to call.

Chatbot is the simplest pattern. It is one LLM conversation without complex tool routing.

Agent-to-agent, or A2A, is a pattern where specialist agents exchange outputs. One agent may research source material, another may analyze structured data, and another may synthesize final conclusions.

---

## Chunk 3: Workflow With Tools Versus Agent With Tools

A workflow with tools means the workflow decides which tools run.

An agent with tools means the agent decides which tools to call.

This distinction is important for the Tech Drive session.

If the Data Analyst workflow is called directly by a workflow node, that is workflow orchestration.

If the Data Agent receives a query and chooses to call the Data Analyst tool, that is agentic tool use.

A2A should show agents exchanging messages, not only a workflow calling multiple tools in sequence.

---

## Chunk 4: E2B Sandbox

E2B is the code execution layer.

When a workflow needs Python execution, E2B starts a sandbox, runs code, returns output, and can then terminate the sandbox.

The Data Analyst workflow uses E2B to run pandas, charting, LazyPredict, and FLAML.

The custom E2B template for this session is named:

dos-lazypredict-flaml

The Python packages installed in this template are:
- lazypredict
- flaml[automl]

If the Data Analyst workflow fails at the Create Sandbox node, likely causes are:
- E2B API key missing
- Better E2B plugin credentials not saved
- template name incorrect
- template not built yet
- E2B team ID mismatch

---

## Chunk 5: Data Analyst Workflow

The Data Analyst workflow analyzes CSV data using Python inside E2B.

The workflow accepts:
- csv_data: uploaded CSV file
- csv_url: direct raw CSV URL
- analysis_question: user question

For agent usage, csv_url is preferred. Agents should not pass full CSV text because it may exceed model or tool limits.

The Data Analyst workflow performs:
- dataset shape check
- column listing
- descriptive statistics
- correlation analysis
- missing value summary
- categorical summaries
- chart generation
- LazyPredict model comparison
- FLAML AutoML model selection
- LLM narrative summary

The workflow writes either an uploaded CSV or a CSV URL into the sandbox before analysis.

For CSV URL mode, the workflow writes the URL to:

/home/user/data/input.url

Then Python downloads the file into:

/home/user/data/input.csv

---

## Chunk 6: Data Analyst Model Output

The Data Analyst workflow can run machine learning model comparison.

LazyPredict compares multiple sklearn-compatible models and returns a leaderboard.

FLAML searches for a strong model and hyperparameters within a time budget.

The expected model output fields are:
- target_column
- task_type
- ml_sample_size
- features_used
- lazypredict_leaderboard
- flaml_best_model

For monthly gross rent analysis, the workflow should use a rent-related numeric column as the prediction target.

If model output is missing, check the Send Sandbox Input node logs.

Common causes of missing model output include:
- lazypredict package missing
- flaml package missing
- insufficient rows after cleaning
- no usable numeric target
- CSV download failure
- sandbox timeout

---

## Chunk 7: Web Scraper Workflow

The Web Scraper workflow reads webpage content and converts it into clean markdown-like text.

The workflow accepts:
- url: full webpage URL
- instruction: what to extract from the page

The Web Scraper is useful for:
- market commentary
- official releases
- policy pages
- reports
- primary source pages

The Web Scraper should not invent facts. If the page is blocked, empty, paywalled, or returns an error page, the workflow should say so.

For the rental market demo, the Web Scraper extracts current market signals, dates, numbers, source caveats, and claims that should be checked against structured data.

---

## Chunk 8: Research Agent

The A2A Research Agent is a specialist agent.

Its tool is:

Web Scraper

The Research Agent should call Web Scraper when given a webpage URL.

The Research Agent should extract:
- current market signals
- dates
- numbers
- policy context
- source caveats
- claims that should be validated against structured data

The Research Agent should not analyze CSV files or make model claims. Those tasks belong to the Data Agent.

The Research Agent should end with a section named:

Message to Data Agent

---

## Chunk 9: Data Agent

The A2A Data Agent is a specialist agent.

Its tool is:

Data Analyst

The Data Agent should call Data Analyst when given a CSV URL or asked for structured data analysis.

The Data Agent should report:
- dataset shape
- key statistics
- correlations
- missing values
- model target
- model task
- features used
- best model
- model metrics
- caveats

The Data Agent should compare its findings with the Research Agent message.

The Data Agent should end with a section named:

Message to Synthesis Agent

---

## Chunk 10: A2A Orchestrator

The A2A Agents Orchestrator coordinates multiple agents.

The orchestrator has structured inputs:
- user_question
- csv_url
- web_url
- research_agent_api_key
- data_agent_api_key

The orchestrator should send a full chat query to each agent.

The agent apps themselves do not need separate form fields. Their mission comes through the chat query.

The orchestrator can call:
- Research Agent
- Data Agent
- Synthesis Agent

The A2A goal is to show message passing between agents.

---

## Chunk 11: Demo URLs

The sample CSV URL for the session is:

https://raw.githubusercontent.com/satyaram413/DOS-Tech-Test-Drive/master/URA%2021%2026.csv

An example web URL for Singapore private residential rental commentary is:

https://www.ura.gov.sg/Corporate/Media-Room/Media-Releases/pr25-29

The default Dify API base is:

https://api.dify.ai/v1

---

## Chunk 12: Recommended Demo Question

Use this question for the A2A demo:

Compare current Singapore private residential rental market commentary with historical URA rental data. What drives monthly gross rent, what is the best performing predictive model, and should a young professional rent a 2-bedroom non-landed home now?

This question works well because it requires:
- web evidence from Research Agent
- structured data analysis from Data Agent
- model interpretation
- evidence comparison
- final synthesis

---

## Chunk 13: Troubleshooting Data Analyst

If Data Analyst fails at Create Sandbox, check:
- E2B API key
- Better E2B plugin credentials
- E2B team ID
- template alias
- template build status

If Data Analyst fails at Write CSV URL File, check:
- sandbox ID
- file path
- csv_url value

If Data Analyst fails at Send Sandbox Input, check:
- Python package availability
- CSV URL accessibility
- pandas parsing errors
- LazyPredict errors
- FLAML errors
- command timeout

If Data Analyst works alone but fails inside another workflow, check:
- nested API timeout
- wrong app API key
- tool input names
- blocking response mode

---

## Chunk 14: Troubleshooting Web Scraper

If Web Scraper fails, check:
- URL includes https://
- webpage is public
- page is not blocked by CAPTCHA
- page is not paywalled
- API key or scraper credentials are valid

If output looks like an error page, do not summarize it as source content.

Use another public source if the target page blocks scraping.

---

## Chunk 15: Troubleshooting Agents

If an agent does not call its tool, check:
- tool is attached to the agent
- tool description is clear
- prompt explicitly says when to use the tool
- query includes the URL or CSV URL
- max iterations are high enough

If the Research Agent does not call Web Scraper, make the query explicit:

Use your Web Scraper tool to read this URL: <WEB_URL>

If the Data Agent does not call Data Analyst, make the query explicit:

Use your Data Analyst tool on this CSV URL: <CSV_URL>

---

## Chunk 16: Key Placeholders

Use placeholders in documentation and screenshots.

Do not show real secrets in shared material.

Common placeholders:
- <E2B_API_KEY>
- <E2B_TEAM_ID>
- <BEDROCK_OR_AWS_CREDENTIALS>
- <RESEARCH_AGENT_API_KEY>
- <DATA_AGENT_API_KEY>
- <WEB_SCRAPER_API_KEY>
- <DATA_ANALYST_API_KEY>

---

## Chunk 17: Participant Mental Model

Participants should remember three levels.

Level 1: Workflow with tools.
The workflow decides which tools run.

Level 2: Agent with tools.
The agent decides which tool to call.

Level 3: A2A.
Agents pass messages to other agents, and later agents react to earlier agent outputs.

The session starts with workflows because workflows are reliable.

The session moves to agents because agents can choose tools.

The session ends with A2A because specialist agents can collaborate.

