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
	stripToolSearchArtifacts?: boolean;
	toolSearch?: {
		variant: 'regex' | 'bm25';
	};
}

function createCustomFetch(
	logger: ISupplyDataFunctions['logger'],
	wrapperOptions: FetchWrapperOptions,
): typeof fetch {
	let callCounter = 0;

	return async (input: string | URL | Request, init?: RequestInit): Promise<Response> => {
		const url =
			typeof input === 'string'
				? input
				: input instanceof URL
					? input.toString()
					: (input as Request).url;

		if (!url.includes('/chat/completions')) return fetch(input, init);

		const callId = ++callCounter;
		const isDebug = wrapperOptions.enableDebugLogging;

		// ── Request processing ──────────────────────────────────────────────
		if (init?.body) {
			try {
				const reqBody = JSON.parse(init.body as string);

				// Debug: log full request summary
				if (isDebug) {
					const msgSummary = Array.isArray(reqBody.messages)
						? reqBody.messages.map((m: Record<string, unknown>) => {
								const role = m.role as string;
								const toolCalls = Array.isArray(m.tool_calls)
									? m.tool_calls.map(
											(tc: Record<string, unknown>) =>
												(tc as Record<string, Record<string, string>>).function
													?.name ?? tc.id,
										)
									: undefined;
								const contentLen =
									typeof m.content === 'string'
										? m.content.length
										: Array.isArray(m.content)
											? (m.content as unknown[]).length + ' blocks'
											: 0;
								return { role, contentLen, ...(toolCalls ? { toolCalls } : {}) };
							})
						: 'no messages';
					const toolNames = Array.isArray(reqBody.tools)
						? reqBody.tools.map(
								(t: Record<string, Record<string, string>>) =>
									t.function?.name ?? t.name ?? t.type,
							)
						: [];
					logger.info(
						`[Debug][${callId}] REQUEST model=${reqBody.model} messages=${JSON.stringify(msgSummary)} tools=[${toolNames.join(',')}]`,
					);
					if (reqBody.cache_control_injection_points) {
						logger.info(
							`[Debug][${callId}] cache_control_injection_points=${JSON.stringify(reqBody.cache_control_injection_points)}`,
						);
					}
				}

				// Strip tool search artifacts from conversation history
				if (wrapperOptions.stripToolSearchArtifacts && Array.isArray(reqBody.messages)) {
					let requestStripped = 0;

					for (const msg of reqBody.messages) {
						if (msg.role === 'assistant' && Array.isArray(msg.tool_calls)) {
							const beforeLen = msg.tool_calls.length;
							msg.tool_calls = msg.tool_calls.filter(
								(tc: any) =>
									!tc.function?.name?.startsWith('tool_search_tool_') &&
									!tc.id?.startsWith('srvtoolu_'),
							);
							requestStripped += beforeLen - msg.tool_calls.length;
							if (msg.tool_calls.length === 0) {
								delete msg.tool_calls;
							}
						}
						if (Array.isArray(msg.content)) {
							const beforeLen = msg.content.length;
							msg.content = msg.content.filter(
								(block: any) =>
									!(block.type === 'tool_use' && block.id?.startsWith('srvtoolu_')) &&
									!(
										block.type === 'tool_result' &&
										block.tool_use_id?.startsWith('srvtoolu_')
									),
							);
							requestStripped += beforeLen - msg.content.length;
						}
					}
					const beforeMsgCount = reqBody.messages.length;
					reqBody.messages = reqBody.messages.filter((msg: any) => {
						if (msg.role === 'tool' && msg.tool_call_id?.startsWith('srvtoolu_'))
							return false;
						return true;
					});
					requestStripped += beforeMsgCount - reqBody.messages.length;

					if (isDebug && requestStripped > 0) {
						logger.info(
							`[ToolSearch][${callId}] Stripped ${requestStripped} artifact(s) from request history (${reqBody.messages.length} messages remain)`,
						);
					}

					init = { ...init, body: JSON.stringify(reqBody) };
				}

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
						for (const tool of reqBody.tools) {
							if (tool.type === 'function' && tool.function) {
								tool.defer_loading = true;
							}
						}

						reqBody.tools.unshift({
							type: toolSearchType,
							name: toolSearchName,
						});

						if (isDebug) {
							logger.info(
								`[ToolSearch][${callId}] Injected ${toolSearchName} + defer_loading on ${reqBody.tools.length - 1} tools`,
							);
						}
					}

					init = { ...init, body: JSON.stringify(reqBody) };
				}
			} catch (err) {
				if (isDebug) {
					logger.warn(
						`[Debug][${callId}] Failed to process request body: ${(err as Error).message}`,
					);
				}
			}
		}

		// ── Execute fetch ───────────────────────────────────────────────────
		const response = await fetch(input, init);

		// ── Response processing ─────────────────────────────────────────────
		const needsStripping = wrapperOptions.stripToolSearchArtifacts;

		if (needsStripping || isDebug) {
			let body: Record<string, any>;
			try {
				body = (await response.json()) as Record<string, any>;
			} catch (err) {
				if (isDebug) {
					logger.warn(
						`[Debug][${callId}] Failed to parse response JSON: ${(err as Error).message}`,
					);
				}
				return response;
			}

			// Debug: log full response
			if (isDebug) {
				const choices = Array.isArray(body.choices)
					? body.choices.map((c: Record<string, any>) => ({
							finish_reason: c.finish_reason,
							content: c.message?.content
								? typeof c.message.content === 'string'
									? c.message.content.substring(0, 200) +
										(c.message.content.length > 200 ? '...' : '')
									: `[${(c.message.content as unknown[]).length} blocks]`
								: null,
							tool_calls: Array.isArray(c.message?.tool_calls)
								? c.message.tool_calls.map((tc: Record<string, any>) => ({
										id: tc.id,
										name: tc.function?.name,
										args:
											typeof tc.function?.arguments === 'string'
												? tc.function.arguments.substring(0, 100)
												: tc.function?.arguments,
									}))
								: null,
						}))
					: 'no choices';
				logger.info(
					`[Debug][${callId}] RESPONSE status=${response.status} choices=${JSON.stringify(choices)}`,
				);
				if (body.usage) {
					logger.info(`[Debug][${callId}] usage=${JSON.stringify(body.usage)}`);
				}
			}

			// Strip tool search artifacts
			if (needsStripping && Array.isArray(body.choices)) {
				for (const choice of body.choices) {
					if (Array.isArray(choice.message?.tool_calls)) {
						const before = choice.message.tool_calls.length;

						choice.message.tool_calls = choice.message.tool_calls.filter(
							(tc: Record<string, any>) =>
								!tc.function?.name?.startsWith('tool_search_tool_') &&
								!tc.id?.startsWith('srvtoolu_'),
						);

						if (isDebug && choice.message.tool_calls.length !== before) {
							logger.info(
								`[ToolSearch][${callId}] Stripped ${before - choice.message.tool_calls.length} of ${before} tool_calls, ${choice.message.tool_calls.length} remaining`,
							);
						}

						if (choice.message.tool_calls.length === 0) {
							delete choice.message.tool_calls;
							if (!choice.message.content) choice.message.content = '';
						}
					}

					// Fix finish_reason when no real tool_calls remain
					if (
						(choice.finish_reason === 'tool_calls' || choice.finish_reason === 'tool_use') &&
						(!Array.isArray(choice.message?.tool_calls) ||
							choice.message.tool_calls.length === 0)
					) {
						choice.finish_reason = 'stop';
						if (isDebug) {
							logger.info(
								`[ToolSearch][${callId}] No real tool_calls remain — changed finish_reason to 'stop'`,
							);
						}
					}
				}
			}

			// Reconstruct response with clean headers
			const newBody = JSON.stringify(body);
			const newHeaders = new Headers(response.headers);
			newHeaders.delete('content-length');
			newHeaders.delete('content-encoding');
			newHeaders.delete('transfer-encoding');

			return new Response(newBody, {
				status: response.status,
				statusText: response.statusText,
				headers: newHeaders,
			});
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
	stripToolSearchArtifacts?: boolean;
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
						displayName: 'Strip Tool Search Artifacts',
						name: 'stripToolSearchArtifacts',
						default: true,
						description:
							'Whether to strip server-side tool search entries (srvtoolu_) from API responses and conversation history. Prevents agent loops and "unexpected tool_use_id in tool_result blocks" errors in multi-turn conversations.',
						type: 'boolean',
						displayOptions: {
							show: {
								enableToolSearch: [true],
							},
						},
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

		// Only create custom fetch wrapper when tool search or debug logging is enabled.
		// Prompt caching works entirely through headers + modelKwargs and does not
		// need a fetch interceptor.
		if (options.enableToolSearch) {
			fetchWrapperOptions.toolSearch = {
				variant: options.toolSearchVariant ?? 'regex',
			};

			if (options.stripToolSearchArtifacts !== false) {
				fetchWrapperOptions.stripToolSearchArtifacts = true;
			}
		}

		if (options.enableDebugLogging) {
			fetchWrapperOptions.enableDebugLogging = true;
		}

		// Create the wrapper only when there's something to do
		if (fetchWrapperOptions.toolSearch || fetchWrapperOptions.enableDebugLogging) {
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
