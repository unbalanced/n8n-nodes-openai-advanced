import { ChatOpenAI, type ChatOpenAIFields, type ClientOptions } from '@langchain/openai';
import { getProxyAgent, makeN8nLlmFailedAttemptHandler, N8nLlmTracing } from '@n8n/ai-utilities';
import {
	NodeConnectionTypes,
	type INodeProperties,
	type INodeType,
	type INodeTypeDescription,
	type ISupplyDataFunctions,
	type ILoadOptionsFunctions,
	type SupplyData,
} from 'n8n-workflow';

interface FetchWrapperOptions {
	enableCacheLogging?: boolean;
	enableDebugLogging?: boolean;
	toolSearch?: {
		variant: 'regex' | 'bm25';
	};
}

function createCustomFetch(
	logger: ISupplyDataFunctions['logger'],
	wrapperOptions: FetchWrapperOptions,
): typeof fetch {
	return async (input: string | URL | Request, init?: RequestInit): Promise<Response> => {
		const url =
			typeof input === 'string'
				? input
				: input instanceof URL
					? input.toString()
					: (input as Request).url;

		// Intercept request body for chat completions
		if (url.includes('/chat/completions') && init?.body) {
			try {
				const reqBody = JSON.parse(init.body as string);

				// Inject tool search tool into the tools array
				if (wrapperOptions.toolSearch) {
					const variant = wrapperOptions.toolSearch.variant;
					const toolSearchType =
						variant === 'bm25'
							? 'tool_search_tool_bm25_20251119'
							: 'tool_search_tool_regex_20251119';
					const toolSearchName =
						variant === 'bm25' ? 'tool_search_tool_bm25' : 'tool_search_tool_regex';

					if (Array.isArray(reqBody.tools) && reqBody.tools.length > 0) {
						// Add defer_loading to all existing function tools
						for (const tool of reqBody.tools) {
							if (tool.type === 'function' && tool.function) {
								tool.defer_loading = true;
							}
						}

						// Prepend the tool search tool
						reqBody.tools.unshift({
							type: toolSearchType,
							name: toolSearchName,
						});

						if (wrapperOptions.enableDebugLogging) {
							logger.info(
								`[ToolSearch] Injected ${toolSearchName} + defer_loading on ${reqBody.tools.length - 1} tools`,
							);
						}
					}

					init = { ...init, body: JSON.stringify(reqBody) };
				}

				// Log cache injection points
				if (wrapperOptions.enableDebugLogging && wrapperOptions.enableCacheLogging) {
					logger.info(
						`[PromptCache] request body keys: ${JSON.stringify(Object.keys(reqBody))}`,
					);
					if (reqBody.cache_control_injection_points) {
						logger.info(
							`[PromptCache] injection_points: ${JSON.stringify(reqBody.cache_control_injection_points)}`,
						);
					} else {
						logger.info(
							`[PromptCache] WARNING: cache_control_injection_points NOT in request body`,
						);
					}
				}
			} catch {}
		}

		const response = await fetch(input, init);

		if (!url.includes('/chat/completions')) return response;

		// Log cache usage from response
		if (wrapperOptions.enableDebugLogging && wrapperOptions.enableCacheLogging) {
			const cloned = response.clone();
			cloned
				.json()
				.then((body: any) => {
					const usage = body?.usage;
					if (!usage) return;

					logger.info(`[PromptCache] raw usage: ${JSON.stringify(usage)}`);

					const cacheCreation = usage.cache_creation_input_tokens ?? 0;
					const cacheRead = usage.cache_read_input_tokens ?? 0;
					const promptTokens = usage.prompt_tokens ?? 0;
					const completionTokens = usage.completion_tokens ?? 0;

					logger.info(
						`[PromptCache] prompt=${promptTokens} completion=${completionTokens} cache_creation=${cacheCreation} cache_read=${cacheRead}`,
					);

					if (cacheRead > 0) {
						logger.info(`[PromptCache] Cache HIT — ${cacheRead} tokens read from cache`);
					} else if (cacheCreation > 0) {
						logger.info(
							`[PromptCache] Cache WRITE — ${cacheCreation} tokens written to cache`,
						);
					} else {
						logger.info(`[PromptCache] Cache MISS — no cache activity detected`);
					}
				})
				.catch(() => {});
		}

		return response;
	};
}

type ModelOptions = {
	baseURL?: string;
	frequencyPenalty?: number;
	maxTokens?: number;
	responseFormat?: 'text' | 'json_object';
	presencePenalty?: number;
	temperature?: number;
	reasoningEffort?: 'low' | 'medium' | 'high';
	timeout?: number;
	maxRetries?: number;
	topP?: number;
	enablePromptCaching?: boolean;
	cacheTtl?: string;
	enableDebugLogging?: boolean;
	enableToolSearch?: boolean;
	toolSearchVariant?: 'regex' | 'bm25';
};

const INCLUDE_JSON_WARNING: INodeProperties = {
	displayName:
		'If using JSON response format, you must include word "json" in the prompt in your chain or agent. Also, make sure to select latest models released post November 2023.',
	name: 'notice',
	type: 'notice',
	default: '',
};

const completionsResponseFormat: INodeProperties = {
	displayName: 'Response Format',
	name: 'responseFormat',
	default: 'text',
	type: 'options',
	options: [
		{
			name: 'Text',
			value: 'text',
			description: 'Regular text response',
		},
		{
			name: 'JSON',
			value: 'json_object',
			description:
				'Enables JSON mode, which should guarantee the message the model generates is valid JSON',
		},
	],
};

export class LmOpenAiAdvanced implements INodeType {
	methods = {
		listSearch: {
			async searchModels(
				this: ILoadOptionsFunctions,
				filter?: string,
			) {
				const credentials = await this.getCredentials('openAiApi');
				const baseURL =
					(this.getNodeParameter('options.baseURL', '') as string) ||
					(credentials.url as string) ||
					'https://api.openai.com/v1';

				const { data } = (await this.helpers.httpRequestWithAuthentication.call(
					this,
					'openAiApi',
					{
						method: 'GET',
						url: `${baseURL}/models`,
						json: true,
					},
				)) as { data: Array<{ id: string }> };

				const url = baseURL && new URL(baseURL);
				const isCustomAPI = !!(
					url && !['api.openai.com', 'ai-assistant.n8n.io'].includes(url.hostname)
				);

				const filteredModels = data.filter((model) => {
					// For custom APIs, include all models
					if (isCustomAPI) {
						if (!filter) return true;
						return model.id.toLowerCase().includes(filter.toLowerCase());
					}

					// For OpenAI, only include chat-compatible models
					const id = model.id;
					const includeModel =
						id.startsWith('ft:') ||
						id.startsWith('o1') ||
						id.startsWith('o3') ||
						id.startsWith('o4') ||
						id.startsWith('gpt-5') ||
						(id.startsWith('gpt-') && !id.includes('instruct'));

					if (!filter) return includeModel;
					return includeModel && id.toLowerCase().includes(filter.toLowerCase());
				});

				filteredModels.sort((a, b) => a.id.localeCompare(b.id));

				return {
					results: filteredModels.map((model) => ({
						name: model.id,
						value: model.id,
					})),
				};
			},
		},
	};

	description: INodeTypeDescription = {
		displayName: 'OpenAI Chat Model Advanced',
		name: 'lmOpenAiAdvanced',
		icon: { light: 'file:openAiLight.svg', dark: 'file:openAiLight.dark.svg' },
		group: ['transform'],
		version: 1,
		description: 'OpenAI Chat Model with LiteLLM prompt caching support',
		defaults: {
			name: 'OpenAI Chat Model Advanced',
		},
		codex: {
			categories: ['AI'],
			subcategories: {
				AI: ['Language Models', 'Root Nodes'],
				'Language Models': ['Chat Models (Recommended)'],
			},
			resources: {
				primaryDocumentation: [
					{
						url: 'https://docs.n8n.io/integrations/builtin/cluster-nodes/sub-nodes/n8n-nodes-langchain.lmchatopenai/',
					},
				],
			},
		},

		inputs: [],
		outputs: [NodeConnectionTypes.AiLanguageModel],
		outputNames: ['Model'],
		credentials: [
			{
				name: 'openAiApi',
				required: true,
			},
		],
		requestDefaults: {
			ignoreHttpStatusErrors: true,
			baseURL:
				'={{ $parameter.options?.baseURL?.split("/").slice(0,-1).join("/") || $credentials?.url?.split("/").slice(0,-1).join("/") || "https://api.openai.com" }}',
		},
		properties: [
			{
				...INCLUDE_JSON_WARNING,
				displayOptions: {
					show: {
						'/options.responseFormat': ['json_object'],
					},
				},
			},
			{
				displayName: 'Model',
				name: 'model',
				type: 'resourceLocator',
				default: { mode: 'list', value: 'gpt-4.1-mini' },
				required: true,
				modes: [
					{
						displayName: 'From List',
						name: 'list',
						type: 'list',
						placeholder: 'Select a model...',
						typeOptions: {
							searchListMethod: 'searchModels',
							searchable: true,
						},
					},
					{
						displayName: 'ID',
						name: 'id',
						type: 'string',
						placeholder: 'gpt-4.1-mini',
					},
				],
				description: 'The model. Choose from the list, or specify an ID.',
			},
			{
				displayName:
					'When using non-OpenAI models via "Base URL" override, not all models might be chat-compatible or support other features, like tools calling or JSON response format',
				name: 'notice',
				type: 'notice',
				default: '',
				displayOptions: {
					show: {
						'/options.baseURL': [{ _cnd: { exists: true } }],
					},
				},
			},
			{
				displayName: 'Options',
				name: 'options',
				placeholder: 'Add Option',
				description: 'Additional options to add',
				type: 'collection',
				default: {},
				options: [
					{
						displayName: 'Base URL',
						name: 'baseURL',
						default: 'https://api.openai.com/v1',
						description: 'Override the default base URL for the API',
						type: 'string',
					},
					{
						displayName: 'Cache TTL',
						name: 'cacheTtl',
						type: 'options',
						displayOptions: {
							show: {
								enablePromptCaching: [true],
							},
						},
						options: [
							{
								name: '5 Minutes (Default)',
								value: '5m',
								description:
									'Standard ephemeral cache. Writes cost 1.25x base input tokens.',
							},
							{
								name: '1 Hour',
								value: '1h',
								description: 'Extended cache. Writes cost 2.0x base input tokens.',
							},
						],
						default: '5m',
						description: 'How long the cache should be held before expiring',
					},
					{
						displayName: 'Enable Prompt Caching (LiteLLM)',
						name: 'enablePromptCaching',
						default: false,
						description:
							'Whether to inject cache_control_injection_points and Anthropic headers for LiteLLM prompt caching',
						type: 'boolean',
					},
					{
						displayName: 'Enable Debug Logging',
						name: 'enableDebugLogging',
						default: false,
						description:
							'Whether to log debug information for prompt caching and tool search to n8n logs',
						type: 'boolean',
					},
					{
						displayName: 'Enable Tool Search',
						name: 'enableToolSearch',
						default: false,
						description:
							'Whether to enable Anthropic Tool Search, which allows Claude to dynamically discover tools from a large catalog. All connected tools will be marked as deferred and loaded on-demand.',
						type: 'boolean',
					},
					{
						displayName: 'Tool Search Variant',
						name: 'toolSearchVariant',
						type: 'options',
						displayOptions: {
							show: {
								enableToolSearch: [true],
							},
						},
						options: [
							{
								name: 'Regex',
								value: 'regex',
								description:
									'Claude constructs regex patterns to search tools. Best for exact pattern matching (faster).',
							},
							{
								name: 'BM25',
								value: 'bm25',
								description:
									'Claude uses natural language queries to search tools. Better for semantic matching. Not available on Bedrock.',
							},
						],
						default: 'regex',
						description: 'The search algorithm variant to use for tool discovery',
					},
					{
						displayName: 'Frequency Penalty',
						name: 'frequencyPenalty',
						default: 0,
						typeOptions: { maxValue: 2, minValue: -2, numberPrecision: 1 },
						description:
							"Positive values penalize new tokens based on their existing frequency in the text so far, decreasing the model's likelihood to repeat the same line verbatim",
						type: 'number',
					},
					{
						displayName: 'Maximum Number of Tokens',
						name: 'maxTokens',
						default: -1,
						description:
							'The maximum number of tokens to generate in the response. Set to -1 for model default. Claude models support up to 64K output tokens.',
						type: 'number',
					},
					{
						displayName: 'Max Retries',
						name: 'maxRetries',
						default: 2,
						description: 'Maximum number of retries to attempt',
						type: 'number',
					},
					{
						displayName: 'Presence Penalty',
						name: 'presencePenalty',
						default: 0,
						typeOptions: { maxValue: 2, minValue: -2, numberPrecision: 1 },
						description:
							"Positive values penalize new tokens based on whether they appear in the text so far, increasing the model's likelihood to talk about new topics",
						type: 'number',
					},
					{
						displayName: 'Reasoning Effort',
						name: 'reasoningEffort',
						default: 'medium',
						description:
							'Controls the amount of reasoning tokens to use. A value of "low" will favor speed and economical token usage, "high" will favor more complete reasoning at the cost of more tokens generated and slower responses.',
						type: 'options',
						options: [
							{
								name: 'Low',
								value: 'low',
								description: 'Favors speed and economical token usage',
							},
							{
								name: 'Medium',
								value: 'medium',
								description: 'Balance between speed and reasoning accuracy',
							},
							{
								name: 'High',
								value: 'high',
								description:
									'Favors more complete reasoning at the cost of more tokens generated and slower responses',
							},
						],
						displayOptions: {
							show: {
								'/model': [
									{ _cnd: { regex: '(^o1([-\\d]+)?$)|(^o[3-9].*)|(^gpt-5.*)' } },
								],
							},
						},
					},
					completionsResponseFormat,
					{
						displayName: 'Sampling Temperature',
						name: 'temperature',
						default: 0.7,
						typeOptions: { maxValue: 2, minValue: 0, numberPrecision: 1 },
						description:
							'Controls randomness: Lowering results in less random completions. As the temperature approaches zero, the model will become deterministic and repetitive.',
						type: 'number',
					},
					{
						displayName: 'Timeout',
						name: 'timeout',
						default: 300000,
						description:
							'Maximum amount of time a request is allowed to take in milliseconds. Default is 5 minutes to support large context windows.',
						type: 'number',
					},
					{
						displayName: 'Top P',
						name: 'topP',
						default: 1,
						typeOptions: { maxValue: 1, minValue: 0, numberPrecision: 1 },
						description:
							'Controls diversity via nucleus sampling: 0.5 means half of all likelihood-weighted options are considered. We generally recommend altering this or temperature but not both.',
						type: 'number',
					},
				],
			},
		],
	};

	async supplyData(this: ISupplyDataFunctions, itemIndex: number): Promise<SupplyData> {
		const credentials = await this.getCredentials('openAiApi');

		const modelName = this.getNodeParameter('model', itemIndex, '', {
			extractValue: true,
		}) as string;

		const options = this.getNodeParameter('options', itemIndex, {}) as ModelOptions;

		const configuration: ClientOptions = {};

		if (options.baseURL) {
			configuration.baseURL = options.baseURL;
		} else if (credentials.url) {
			configuration.baseURL = credentials.url as string;
		}

		if (configuration.baseURL) {
			configuration.fetchOptions = {
				dispatcher: getProxyAgent(configuration.baseURL ?? 'https://api.openai.com/v1'),
			};
		}

		const defaultHeaders: Record<string, string> = {};
		if (
			credentials.header &&
			typeof credentials.headerName === 'string' &&
			credentials.headerName &&
			typeof credentials.headerValue === 'string'
		) {
			defaultHeaders[credentials.headerName] = credentials.headerValue;
		}

		// Inject Anthropic prompt caching header for LiteLLM
		if (options.enablePromptCaching) {
			defaultHeaders['anthropic-beta'] = 'prompt-caching-2024-07-31';
		}

		if (Object.keys(defaultHeaders).length > 0) {
			configuration.defaultHeaders = defaultHeaders;
		}

		// Extra options to send to OpenAI, that are not directly supported by LangChain
		const modelKwargs: Record<string, unknown> = {};

		if (options.responseFormat) {
			modelKwargs.response_format = { type: options.responseFormat };
		}

		if (options.reasoningEffort && ['low', 'medium', 'high'].includes(options.reasoningEffort)) {
			modelKwargs.reasoning_effort = options.reasoningEffort;
		}

		// Inject LiteLLM prompt caching injection points
		if (options.enablePromptCaching) {
			const injectionPoint: Record<string, string> = {
				location: 'message',
				role: 'system',
			};

			if (options.cacheTtl === '1h') {
				injectionPoint.ttl = '1h';
			}

			modelKwargs.cache_control_injection_points = [injectionPoint];
		}

		// Set up custom fetch wrapper for cache logging and/or tool search injection
		const fetchWrapperOptions: FetchWrapperOptions = {};

		if (options.enableDebugLogging) {
			fetchWrapperOptions.enableDebugLogging = true;
		}

		if (options.enablePromptCaching) {
			fetchWrapperOptions.enableCacheLogging = true;
		}

		if (options.enableToolSearch) {
			fetchWrapperOptions.toolSearch = {
				variant: options.toolSearchVariant ?? 'regex',
			};
		}

		if (Object.keys(fetchWrapperOptions).length > 0) {
			configuration.fetch = createCustomFetch(this.logger, fetchWrapperOptions);
		}

		const fields: ChatOpenAIFields = {
			apiKey: credentials.apiKey as string,
			model: modelName,
			frequencyPenalty: options.frequencyPenalty,
			maxTokens: options.maxTokens,
			presencePenalty: options.presencePenalty,
			temperature: options.temperature,
			topP: options.topP,
			timeout: options.timeout ?? 300000,
			maxRetries: options.maxRetries ?? 2,
			configuration,
			callbacks: [new N8nLlmTracing(this) as any],
			modelKwargs,
			onFailedAttempt: makeN8nLlmFailedAttemptHandler(this),
		};

		const model = new ChatOpenAI(fields);

		return {
			response: model,
		};
	}
}
