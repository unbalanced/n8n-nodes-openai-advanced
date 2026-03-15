# n8n-nodes-openai-advanced

An n8n community node that extends the built-in OpenAI Chat Model to support **non** Open AI models and features for models that are proxied/consumed via  LiteLLM.

The Lib enables support to **prompt caching** and **Anthropic Tool Search Tool**.

## Features

- Drop-in replacement for n8n's built-in OpenAI Chat Model node
- **Prompt caching** via LiteLLM — reduce costs and latency for repeated system prompts
- **Cache TTL** control — choose between 5-minute (default) or 1-hour cache duration
- **Cache usage logging** — monitor cache hits/writes/misses in n8n logs
- **Tool Search** via LiteLLM — Anthropic's dynamic tool discovery for large tool catalogs (Regex or BM25 variants)
- Base URL override for LiteLLM or other OpenAI-compatible proxies
- Dynamic model list from your API endpoint
- All standard ChatOpenAI options: temperature, top P, frequency/presence penalty, max tokens, response format, reasoning effort, timeout, retries

## Installation

### In n8n (Community Nodes)

1. Go to **Settings > Community Nodes**
2. Install: `n8n-nodes-openai-advanced`

### Manual Installation

```bash
cd ~/.n8n/nodes
npm install n8n-nodes-openai-advanced
```

Restart n8n after installation.

## Usage

1. Add the **OpenAI Chat Model Advanced** node to your workflow
2. Connect it to an AI Agent or Chain as the language model
3. Configure your OpenAI API credentials (reuses the built-in `openAiApi` credential)
4. Select a model from the list or enter a model ID manually

### Prompt Caching (LiteLLM)

To enable prompt caching for Anthropic models through LiteLLM:

1. Set the **Base URL** to your LiteLLM proxy endpoint
2. Enable **Enable Prompt Caching (LiteLLM)** in the Options
3. Optionally set **Cache TTL** to `1 Hour` (default is `5 Minutes`)

#### How it works

When prompt caching is enabled, the node:

- Adds the `anthropic-beta: prompt-caching-2024-07-31` header
- Injects `cache_control_injection_points` into the request body, telling LiteLLM to add `cache_control` blocks to system messages
- Logs cache usage to n8n's logger (visible in container logs)

#### Requirements

- A LiteLLM proxy configured with an Anthropic model (e.g., `claude-sonnet-4-20250514`)
- System prompt must be at least **1024 tokens** for caching to activate (Anthropic requirement)

#### Monitoring Cache Usage

When prompt caching is enabled, the node logs cache activity:

```
[PromptCache] prompt=1500 completion=200 cache_creation=1024 cache_read=0
[PromptCache] Cache WRITE — 1024 tokens written to cache
```

On subsequent requests within the TTL:

```
[PromptCache] prompt=1500 completion=180 cache_creation=0 cache_read=1024
[PromptCache] Cache HIT — 1024 tokens read from cache
```

Logs are written to n8n's standard logger (visible in Docker/container stdout).

### Tool Search

Tool Search enables Claude to dynamically discover and load tools on-demand from a large catalog. Instead of sending all tool definitions upfront, tools are marked as deferred and Claude searches for the ones it needs.

To enable Tool Search:

1. Set the **Base URL** to your LiteLLM proxy endpoint
2. Enable **Enable Tool Search** in the Options
3. Optionally change the **Tool Search Variant** (default: Regex)

#### How it works

When Tool Search is enabled, the node:

- Injects a tool search tool (`tool_search_tool_regex` or `tool_search_tool_bm25`) into the request
- Marks all connected tools with `defer_loading: true` so they are loaded on-demand
- LiteLLM automatically adds the required `anthropic-beta: advanced-tool-use-2025-11-20` header

Claude then searches for relevant tools using regex patterns (Regex variant) or natural language queries (BM25 variant) before calling them.

#### Variants

| Variant | Description | Availability |
|---------|-------------|--------------|
| **Regex** | Claude constructs regex patterns to search tools by name, description, and arguments. Faster. | All providers |
| **BM25** | Claude uses natural language queries for semantic tool matching. Better for large catalogs. | Not available on Bedrock |

#### Monitoring

When Tool Search is enabled, the node logs injection activity:

```
[ToolSearch] Injected tool_search_tool_regex + defer_loading on 5 tools
```

The LiteLLM response usage will also include `tool_search_requests` count.

#### Requirements

- A LiteLLM proxy configured with a Claude model via Vertex AI, Bedrock, or direct Anthropic API
- At least one tool connected to the agent

## Options

| Option | Description | Default |
|--------|-------------|---------|
| Base URL | Override the API endpoint (e.g., LiteLLM proxy) | `https://api.openai.com/v1` |
| Enable Prompt Caching | Inject LiteLLM cache control headers and injection points | `false` |
| Cache TTL | Cache duration: 5 Minutes or 1 Hour | `5 Minutes` |
| Enable Tool Search | Enable Anthropic Tool Search for dynamic tool discovery | `false` |
| Tool Search Variant | Regex (pattern matching) or BM25 (semantic search) | `Regex` |
| Frequency Penalty | Penalize repeated tokens (-2 to 2) | `0` |
| Maximum Number of Tokens | Max tokens to generate | `-1` (unlimited) |
| Max Retries | Number of retry attempts | `2` |
| Presence Penalty | Penalize new topic tokens (-2 to 2) | `0` |
| Reasoning Effort | For o-series/gpt-5 models: low, medium, high | `medium` |
| Response Format | Text or JSON mode | `text` |
| Sampling Temperature | Controls randomness (0 to 2) | `0.7` |
| Timeout | Request timeout in milliseconds | `300000` |
| Top P | Nucleus sampling threshold (0 to 1) | `1` |

## Development

```bash
npm install
npm run build
```

## License

MIT
