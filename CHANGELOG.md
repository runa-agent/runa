# Changelog

All notable changes to Runa are documented here.

## [0.3.0] - 2026-09-24

### Features

- Evals/<agent_name>.jsonl is a complete eval, agent resolved from filename ([b35d034](https://github.com/runa-agent/runa/commit/b35d03492f9b7b5e36cd4896badde8a0df0f0c01))

- Generate agent also creates its evals/<name>.jsonl ([f55955a](https://github.com/runa-agent/runa/commit/f55955a8be261c8b57961a77d9a96af737e15b0e))

- Add traces to evals, flag regressions against the last run, bare-string cases ([dd1ddb5](https://github.com/runa-agent/runa/commit/dd1ddb5fb521d5889c28058a65ce023407b93b12))

- Parallel eval cases, case-to-trace links, regressions in ui, dedupe --add ([f4df943](https://github.com/runa-agent/runa/commit/f4df943e347febce7f744a7e28797ca96d8c9560))

- Run a message's tool calls concurrently, honor parallel_tool_calls=False ([cd65cfd](https://github.com/runa-agent/runa/commit/cd65cfdddb79d4174bf029f57e2c60c590569d9e))

- Run_streamed pauses for approval and resumes from a RunState ([45a16da](https://github.com/runa-agent/runa/commit/45a16da51b7bb7fab210eb56320ace972d01a7dd))

- Agent is the single run API: paused Runs resume, delegates pause their caller, output_type parses, model calls retry ([b1b5ca0](https://github.com/runa-agent/runa/commit/b1b5ca0e6face808783a97b2b0ff7812405c3b5b))

- Run carries the guardrail audit trail for every status ([80e1892](https://github.com/runa-agent/runa/commit/80e1892d44f82467225d735d3e23b7b24e0a21e6))


### Bug Fixes

- Async guardrail call ([e2f0a60](https://github.com/runa-agent/runa/commit/e2f0a609f1c46b41369de6c3fb92f86c92e892cc))

- Key eval reports by agent name, drop stale DeepEval mentions ([2d293d3](https://github.com/runa-agent/runa/commit/2d293d3d3dc2a103fe223d9b75b4d7cd61c9db10))

- Feed malformed tool arguments back to the model instead of crashing ([3022059](https://github.com/runa-agent/runa/commit/3022059b3be8e05beceb8bbb181a97aeaaabbda3))

- Compact on current context size, not cumulative run usage ([ef617ed](https://github.com/runa-agent/runa/commit/ef617ed27e75a309d9e89f41d4e2ae549079efd7))

- Resume runs approved calls from mixed-approval turns and persists to the session ([10b2941](https://github.com/runa-agent/runa/commit/10b2941aa9d906c6e2fd40e7b9b6c17d21851d38))

- Record the real cause on tool spans, close cancelled siblings, place retrieval before its message ([9528811](https://github.com/runa-agent/runa/commit/9528811fd18b20d514f06e96b8271170bb51984d))


### Refactor

- Run_streamed shares run's turn loop, gaining guardrails, tracing, sessions and memory ([3ced549](https://github.com/runa-agent/runa/commit/3ced549edce30855d451bc9254e7b68b5a05a1db))

- Share run and resume finishing, report partial items on errors ([1c81564](https://github.com/runa-agent/runa/commit/1c815647c1c75a0468ff54687a5beda6c5fe9acf))


### Documentation

- Mention runa ui at end of tracing section ([3dd8b71](https://github.com/runa-agent/runa/commit/3dd8b711b572b506ceba51278d532c4b55fea167))

- Document loading eval datasets from jsonl ([7afb5c3](https://github.com/runa-agent/runa/commit/7afb5c3f1be76cfa83294770cce64592282b8e42))


## [0.2.0] - 2026-09-21

### Features

- Automate changelog generation with git-cliff ([bdf2df7](https://github.com/runa-agent/runa/commit/bdf2df79e29d2a8bf0d02b6cd31e4d38fbd7829f))

- Support image input in messages, auto-detected from a plain string list ([f6f50eb](https://github.com/runa-agent/runa/commit/f6f50ebd6d9019978a91a76ae7e878a2b5a82cd8))

- Add langfuse integration ([9c40b32](https://github.com/runa-agent/runa/commit/9c40b3248027372519e0ec4642bba40fd0eb16ca))

- Update traces in ui ([3912821](https://github.com/runa-agent/runa/commit/39128210bd354beae19b06602fa6b79ee6f97055))


### Bug Fixes

- Read llm span usage from output, not unused attributes field ([17cb27b](https://github.com/runa-agent/runa/commit/17cb27b5d9153b3cb47d9d90488aef384303d7c6))

- Rebuild RedisCache's client when the event loop changes ([c5f356a](https://github.com/runa-agent/runa/commit/c5f356a30cb9a73ab2d84be1ba0aa3c891921608))

- Async tool call ([63b08aa](https://github.com/runa-agent/runa/commit/63b08aadc11afee87b84c84944448982354ecc45))


### Documentation

- Fix links ([68eed32](https://github.com/runa-agent/runa/commit/68eed32d92031bdedb90cc8489666b33198ada67))

- Add examples ([e87e4fe](https://github.com/runa-agent/runa/commit/e87e4fe23e902f7a321bc297a6794813ea94e25f))

- Remove redundancy ([3dae8ab](https://github.com/runa-agent/runa/commit/3dae8abb11b0c7ff7504e05912d0cbf45477f82d))

- Add cache example ([ab417ab](https://github.com/runa-agent/runa/commit/ab417abc3dfac7d6d0c57960daae6cd389ae92aa))


## [0.1.2] - 2026-09-12

### Bug Fixes

- Make runa generate tool take NAME positionally, matching agent/guardrail/evaluation ([9e715e4](https://github.com/runa-agent/runa/commit/9e715e467aa1df1fa3e6d30b2a88584e92b3ea93))


## [0.1.1] - 2026-09-12

### Bug Fixes

- Bump version to 0.1.1, 0.1.0 already used on PyPI ([9d92067](https://github.com/runa-agent/runa/commit/9d920679a25c8c1026f196053a84c79a5d3687d9))


## [0.1.0] - 2026-09-12

### Features

- Define agent runtime boundary ([c6aeb71](https://github.com/runa-agent/runa/commit/c6aeb717b0f77036635a9b8cc00ef844884e55aa))

- Add model runtime resolution ([6431a82](https://github.com/runa-agent/runa/commit/6431a82358cd09dc8f66e76a6d42c61863d761b7))

- Add model runtime resolution ([70f3576](https://github.com/runa-agent/runa/commit/70f357606ce6aa5fb129a3053415cfae9cd8dc24))

- Resolve models by provider convention ([b18a475](https://github.com/runa-agent/runa/commit/b18a4753a88a3e0a4119f9b9300daa19c988c3fe))

- Introduce application configuration ([3da497d](https://github.com/runa-agent/runa/commit/3da497d414a17c613ac949377aac9a9f7a757587))

- Add tools ([cce42d5](https://github.com/runa-agent/runa/commit/cce42d5b63be6a973677706c4d13a1d36b07e930))

- Make Run identity first-class ([9283d55](https://github.com/runa-agent/runa/commit/9283d559770f0ad995ceea0ff3566e01b53d2548))

- Add identity to Run ([7dd1d05](https://github.com/runa-agent/runa/commit/7dd1d05e29b845a918296498aba15efca5de8e5a))

- Assign identity to agent runs ([f230e3b](https://github.com/runa-agent/runa/commit/f230e3bf0264adb6ceabaa8c431bd045eeda3eaa))

- Make Run identity first-class ([530f557](https://github.com/runa-agent/runa/commit/530f557e1a526ca03c8147c81f43808d4a0bd9ce))

- Add input to Run ([030646e](https://github.com/runa-agent/runa/commit/030646ec1183f8e9ed78b8601cfe78f11a5da02c))

- Expose tools through Runa agent factory ([a31b51c](https://github.com/runa-agent/runa/commit/a31b51ce66a033b7e452e379769ba6544f2897da))

- Add completion status to Run ([ca0c0c7](https://github.com/runa-agent/runa/commit/ca0c0c79ca0f77c45ba023f8c925777c489adc12))

- *(core)* Add core layer ([6bb4494](https://github.com/runa-agent/runa/commit/6bb44942f4c7bbdddd8727ac9ee9189ec25aac57))

- Add agent and tool layers ([0f469db](https://github.com/runa-agent/runa/commit/0f469dbdc5799bf9021701c0663ae4c1b917bb3b))

- Add runtime/ ([5a27536](https://github.com/runa-agent/runa/commit/5a27536e823cdf7eb8c33692ef51113da8af78a8))

- Add runtime/ ([b272de0](https://github.com/runa-agent/runa/commit/b272de0b6f393288cff9200e5b7ed1ce491b989e))

- Add providers/ ([39a133f](https://github.com/runa-agent/runa/commit/39a133fac0285dd47488cd76563996dee02bae18))

- Add providers/ ([7f1daf2](https://github.com/runa-agent/runa/commit/7f1daf24f6ce3fab487420b56f4d38d335bf97b0))

- Add persistence/ ([9701841](https://github.com/runa-agent/runa/commit/97018418b87a3d57ab73ece36dca603c51db8254))

- Add background and approval.py ([50b7258](https://github.com/runa-agent/runa/commit/50b72584a0d6b6b5514a5732cc04639ac876f2dc))

- Add observability/ ([8895ad3](https://github.com/runa-agent/runa/commit/8895ad3b9c0a4cdb31d87b890945727b20a408de))

- Add eval/ ([8b37b1f](https://github.com/runa-agent/runa/commit/8b37b1fdfe7e9c8b190f63129077c0e7cf20a367))

- Add cli/ ([7f7493f](https://github.com/runa-agent/runa/commit/7f7493ff5cd9ea5fad66d62f4da0f39027de00ae))

- Add config ([d673ff4](https://github.com/runa-agent/runa/commit/d673ff439236c9eb77a86a125dbd5f37d6339c66))

- Add examples/ ([38229bb](https://github.com/runa-agent/runa/commit/38229bb887362f132889804b869929c6c778003f))

- Add contributing.md and smoking tests ([0639244](https://github.com/runa-agent/runa/commit/06392448c0774b93fbc956c71184a924b2101970))

- Define user repo structure ([c37fc54](https://github.com/runa-agent/runa/commit/c37fc54422ca4bae66bf3d3fd10883a9a66e2e3c))

- Add SQLiteRunStore, a durable RunStore backend ([7f08ebb](https://github.com/runa-agent/runa/commit/7f08ebbc0653e5598b3292465db3f8187d219ac8))

- Add RetryStrategy ([e2514a7](https://github.com/runa-agent/runa/commit/e2514a7afdcd665de5504b7b426d3e68f75e9277))

- Add ThreadQueue ([cbeeaf4](https://github.com/runa-agent/runa/commit/cbeeaf4a643ad47e5a922a563c682c63745b9de9))

- Wire plan() and review() hooks into the Executor lifecycle ([a1d6c95](https://github.com/runa-agent/runa/commit/a1d6c954d12d02b50acc9271093441e0a90ed707))

- Add Conversation for multi-turn state across separate Agent.run() call ([ec49fb1](https://github.com/runa-agent/runa/commit/ec49fb1d9e2d726f41f83e8ffb765de2f7a2b1e9))

- Add `runa eval` and `runa runs show <id>` CLI commands ([b197396](https://github.com/runa-agent/runa/commit/b197396974d799ea94b8b01f741d6bc86c9a7c51))

- Add Agent.as_tool() for agent-to-agent delegation ([b2eddfc](https://github.com/runa-agent/runa/commit/b2eddfca6a9a0597e236bcc9f9b65ea031576c22))

- Add Agent.as_tool() for agent-to-agent delegation ([4ddff96](https://github.com/runa-agent/runa/commit/4ddff96cf70c52c113c16c424166e758781eab2d))

- Add AsyncExecutor and AsyncProvider for concurrent tool execution ([80af1d3](https://github.com/runa-agent/runa/commit/80af1d3ace9d471b6b7bf2c79f5f3c089b4de355))

- Add runs approve/deny/pending CLI and deepen tool schema inference for list/optional/enum/dataclass type ([6ffec34](https://github.com/runa-agent/runa/commit/6ffec34312ede3ca1e79f0181387b2c69aed16ce))

- Add Judge for LLM-graded semantic evals (manifesto §12) ([3e29d82](https://github.com/runa-agent/runa/commit/3e29d82169d2f5b919ea86c4f3861d621571deb3))

- Add hello_anthropic example to exercise AnthropicProvider end-to-end (manifesto §17) ([5768188](https://github.com/runa-agent/runa/commit/576818845fec1d9d5d484ba1f07b0536bf69ff8e))

- Add `runa test` CLI verb for app/tests/ (manifesto §12) ([d097923](https://github.com/runa-agent/runa/commit/d0979235b7f91aeea84bf165d9a850879f8419a2))

- Auto-record an Artifact when a Tool returns one (manifesto §10) ([e7953d3](https://github.com/runa-agent/runa/commit/e7953d34b0a3fd490feaa5597a69eaaacc384123))

- Let review() revise the Run's result, add plan/review reference example (manifesto §6) ([5a216bd](https://github.com/runa-agent/runa/commit/5a216bd68d74c24a79dd87a8ae152ef295ed172c))

- Add SQLiteQueue and DurableQueue protocol for crash-durable background executio ([b8d5ed2](https://github.com/runa-agent/runa/commit/b8d5ed2e570a6e9a95aec0ab1d397e2072d38944))

- Add webhook subscriber for external event export ([a186c21](https://github.com/runa-agent/runa/commit/a186c21c23312f3861c4a835847eb95294694671))

- Add RunStore filtering/`runa runs list` and Agent.as_async_tool() for concurrent delegatio ([034dc7c](https://github.com/runa-agent/runa/commit/034dc7c9d150ef4e192dde699297991db1ce0b25))

- Add Tool.idempotent and ToolCall.effect so RetryStrategy won't blindly retry non-idempotent side effect ([f44b77c](https://github.com/runa-agent/runa/commit/f44b77c35c2253875649b661630df3a7fb1720f5))

- Stamp Run.agent_id/agent_version at seed time and add agent_id filtering to RunStore/CL ([ae2fcaa](https://github.com/runa-agent/runa/commit/ae2fcaad3eec002e382f19f1f627bc973bc6d363))

- Add Run.parent_run_id and ParentRunAware so delegated Runs record their lineag ([169f92c](https://github.com/runa-agent/runa/commit/169f92c75f0050301d2906e816b8beae54cfda24))

- Add recover_pending() to automate SQLiteQueue's manual crash-recovery pat ([44650b6](https://github.com/runa-agent/runa/commit/44650b6857698b5b94ea232055984113a8151fed))

- Add Expectation.to_meet_the_goal()/RUBRIC_GOAL, matching what the docs already documented ([0306e43](https://github.com/runa-agent/runa/commit/0306e436642f8d415f26bb296ff9585b735bf75b))

- Add Run.completed property to match the assert run.completed example in all four doc ([cae8649](https://github.com/runa-agent/runa/commit/cae8649f561991d214342a1999d15214317912bc))

- Give Run.context a real Context type instead of a bare dict ([e011268](https://github.com/runa-agent/runa/commit/e011268c28cb6054bd13415e4062f60ddc5e8d6e))

- Add Agent.policies, a programmatic allow/deny check that runs before approval ([e4d19e4](https://github.com/runa-agent/runa/commit/e4d19e42b1ac72f1db8892dab12bcf023009786b))

- Export Context and Policy from the top-level runa package ([99094ff](https://github.com/runa-agent/runa/commit/99094ffa032adb8af3922ef0b378b89f27c79bb2))

- Add Run.request_cancel() and `runa runs cancel` for safe Run cancellation ([20b22a1](https://github.com/runa-agent/runa/commit/20b22a1aa6f9bfbe66823bddf9783ece3b3ff397))

- Add Executor/AsyncExecutor timeout, checked at the same step boundary as max_step ([ba8c4a7](https://github.com/runa-agent/runa/commit/ba8c4a7aadd499a06758539ddb149c12815db6dc))

- Add Run.error (mirroring Run.result) and Expectation.to_have_error() ([092dd33](https://github.com/runa-agent/runa/commit/092dd33317ed1d6484aad13ef1d9f63502545870))

- Include arguments in TOOL_CALLED events and the timeline summary, so concurrent calls are distinguishable ([14b3e2d](https://github.com/runa-agent/runa/commit/14b3e2dc7d406611d67af9be0aa4afa1c6760a4a))

- Give MODEL_CALLED/MODEL_RESPONDED events real data (model, content, tool call count ([faae97e](https://github.com/runa-agent/runa/commit/faae97e2845df1c26c93b1acc8f0f2acdf97e668))

- Add Run.usage, summing token accounting across a Run's model calls ([ada735b](https://github.com/runa-agent/runa/commit/ada735b34a16232cc9263ceb38bb303bfa863871))

- Add RetryingProvider/AsyncRetryingProvider for transient model-API failures ([564e7c0](https://github.com/runa-agent/runa/commit/564e7c010891263ecc2885f9fba343ad558da80d))

- Add generate tool/evaluation, clean AppLoadError, and next-step hints for new/generate ([21ca1c0](https://github.com/runa-agent/runa/commit/21ca1c0f1f89d24891f504f94d14f93ec6d9088b))

- Give the `runa` CLI a top-level --help description naming the Agent → Run → Result model ([b214c46](https://github.com/runa-agent/runa/commit/b214c46c0f5ceed9cd7829b058284786b8a0959c))

- Add Application as the app-wide configuration primitive, with runa.configure() as sugar for the default instanc ([fa88732](https://github.com/runa-agent/runa/commit/fa8873272c2e9d2e85f0b019b3efd5acdab93969))

- Resolve provider="openai" string shorthand to a Provider instance in Application.configure() ([bacfa92](https://github.com/runa-agent/runa/commit/bacfa922b4c36a53d401f0b81808c4fa65121b51))

- Scaffold .env and load_dotenv() in runa new so credentials never need shell export ([4cf92ba](https://github.com/runa-agent/runa/commit/4cf92bab4105d294e48ddd6396807cd54b515a30))

- Create agent class ([d05b38f](https://github.com/runa-agent/runa/commit/d05b38f3cfe2cea539b291dc2f88587c59f24329))

- Let assistant auto-decide handoff vs delegate for bare subagent ([eefce96](https://github.com/runa-agent/runa/commit/eefce96d791c75858a306426251992f3e4d8357d))

- Accept dict form for subagents (handoff/delegate/auto buckets ([9ed3ea9](https://github.com/runa-agent/runa/commit/9ed3ea98cd136aef0b3b482c420486c660ef5a6e))

- Add guardrail.input/guardrail.output deco ([de5940c](https://github.com/runa-agent/runa/commit/de5940c2069f85c3c768e44218574457e3b1f892))

- Rework guardrail as a single @guardrail decorator with .input/.output binding ([ca1ed6d](https://github.com/runa-agent/runa/commit/ca1ed6d4f66a3c74fc89f7ce55c1554d89901ac1))

- Add guardtail to tool ([b33e896](https://github.com/runa-agent/runa/commit/b33e896c1f965bfc9f1d9960cd99a56e5514df37))

- Add guardrail to tool ([ae3ffb6](https://github.com/runa-agent/runa/commit/ae3ffb6774801bb1032e692b520dba8479b9112b))

- Add dynamic instructions ([e9b76df](https://github.com/runa-agent/runa/commit/e9b76df029ce0866eb8224d6f62ea21406096850))

- Forward hooks through Agent.run/run_sync ([017cc05](https://github.com/runa-agent/runa/commit/017cc05d2d4f035ca46cb69f3531331897a905ec))

- Add LoggingRunHooks as the default hooks for every run ([f53f0c6](https://github.com/runa-agent/runa/commit/f53f0c6b09221b2d953bf491dee3d34348e9dbed))

- Add LoggingRunHooks ([5ea6ef4](https://github.com/runa-agent/runa/commit/5ea6ef4820dcb5fd28b580429a91daa31cfde93d))

- Add MetricsRunHook ([d1dedd2](https://github.com/runa-agent/runa/commit/d1dedd229cae94d7ca7ae39b5286b96fe89ac40f))

- Add TracingRunHook ([8d69cf5](https://github.com/runa-agent/runa/commit/8d69cf5a2c415790eba97a2cadbe120bfa922839))

- Add AuditRunHook ([03c9231](https://github.com/runa-agent/runa/commit/03c9231cdcca7011d79706c5e6fb763b96fe91fb))

- Combine all built-in RunHooks as the default for Agent.run ([a684a77](https://github.com/runa-agent/runa/commit/a684a77d1b65ae99a8369b18f6a524d158a1f89d))

- Add CompositeAgentHooks combining logging, metrics, and audit ([31b5480](https://github.com/runa-agent/runa/commit/31b548084c6529f67502d36d61d1b9a1bcb31a0b))

- Re-export RunHooks and AgentHooks from runa ([c310eab](https://github.com/runa-agent/runa/commit/c310eaba3e224773208a8ae548636dfec8f9a91c))

- Re-export hosted SDK tools via runa.tools ([1d1db73](https://github.com/runa-agent/runa/commit/1d1db73df9f8f3649e924760673f91cfd2ec01b2))

- Add Agent().run_streamed() for event-streamed turns ([4faf6f4](https://github.com/runa-agent/runa/commit/4faf6f49c3f697775f426cd268460be679a16b73))

- Add websocket_session for previous_response_id-chained streaming ([e9076d6](https://github.com/runa-agent/runa/commit/e9076d652e829868a4d0762a66a23506618a17e9))

- Add @approval decorator for named-param needs_approval predicate ([034376b](https://github.com/runa-agent/runa/commit/034376b978607a0ca4380421114d96ba513f338c))

- Re-export SQLiteSession from run ([c1357f6](https://github.com/runa-agent/runa/commit/c1357f62c5db7a5381ca52c9047aac259e2b1808))

- Add session= param to Agent.run and run_sync ([27e5e60](https://github.com/runa-agent/runa/commit/27e5e60c43ffb6042c92a5e741e886585699fd6b))

- Re-export AsyncSQLiteSession and RedisSession from runa ([1caf96e](https://github.com/runa-agent/runa/commit/1caf96ec07293cb8196ad99d2bec6506a9cc7f22))

- Add .compact() session sugar for OpenAI responses compaction ([50da009](https://github.com/runa-agent/runa/commit/50da0094571e84baf400ee8a63b9643572e62dfa))

- Re-export SQLAlchemySession from runa ([b6731a5](https://github.com/runa-agent/runa/commit/b6731a53bb7eff13760788a5a86b84ef0976deee))

- Track token usage per-call and cumulatively on Agent ([68ebd4b](https://github.com/runa-agent/runa/commit/68ebd4b88c662f086909f09044953d6d5bff22df))

- Re-export MCPServerStreamableHttp from runa ([56e4adb](https://github.com/runa-agent/runa/commit/56e4adb4f7f0f917f88921ca95907d9356c2d3a1))

- Add MCPServer builder with .http()/.stdio() transport ([598cc19](https://github.com/runa-agent/runa/commit/598cc19616037de410e70467f2feb3b2caa2673a))

- Add mcp= shorthand for mcp_servers on Agent ([2473de1](https://github.com/runa-agent/runa/commit/2473de189d3188add9d1b1c55279b84de83c445c))

- Replace eval harness with Case/Dataset/agent.evaluate() and DeepEval-backed metric ([a530f69](https://github.com/runa-agent/runa/commit/a530f69e8bf5d89b4ae16d1f6d844138dc00d7cc))

- Return a Run object (output/trace/usage/status/error) from Agent.run/run_syn ([579530c](https://github.com/runa-agent/runa/commit/579530c7fefd754fd2296da39f018c56e4811d7f))

- Add runa.tracing — automatic hierarchical observability off the Agents SDK's own tracing ([e62f6bd](https://github.com/runa-agent/runa/commit/e62f6bd4cf75f93e13ae79907a44fe39735c28b5))

- Remove DeepEval dependenc ([984b19d](https://github.com/runa-agent/runa/commit/984b19dd793d9a78cb338722e445a3c0369bcc6a))

- Add agent.graph ([0a6db45](https://github.com/runa-agent/runa/commit/0a6db45a85ab89d8c83a58dcd9479e736498d9c7))

- Add runa chat for interactive agent conversations ([9690212](https://github.com/runa-agent/runa/commit/9690212bed79ca3acd13e94beed3347388cfc912))

- Add logging.py ([79b270f](https://github.com/runa-agent/runa/commit/79b270f704bb3735fac163829396a39ec73cb613))

- Start a new chat session by default, add --continue/--resume to pick up a past one ([47aa6a2](https://github.com/runa-agent/runa/commit/47aa6a28e875993d935875eaf8ba583ec22eb6bd))

- Remove anthropic and openai ([ceb8d05](https://github.com/runa-agent/runa/commit/ceb8d05c7528c20f16b92309467e7afa59e2e211))

- Remove llmlite ([63e248e](https://github.com/runa-agent/runa/commit/63e248e7203ee59d0362d896200881eb43870fd3))

- Update readme ([195c083](https://github.com/runa-agent/runa/commit/195c083ab78b95afd624ab043f42ccd4edc247b8))

- Update runa new cli ([94cce23](https://github.com/runa-agent/runa/commit/94cce23024e1e9beb8da60e3ed13cf72dfe0e60e))

- Replace openai-agents/openai with an in-house agent runtime ([89c3481](https://github.com/runa-agent/runa/commit/89c34817c1048011884396edc2fb9ef5e303ef1f))

- Define final tools convention ([abb38ed](https://github.com/runa-agent/runa/commit/abb38ed5a89ea46276415f5c606f21c390af0395))

- Auto-create app/prompts/<name>.md from the generate-prompt stub when missing ([01b15af](https://github.com/runa-agent/runa/commit/01b15af717188c22e615ed17fbd58e1fac31ba9b))

- Add agentic memory ([8b0628a](https://github.com/runa-agent/runa/commit/8b0628adbe2b585a1e41fe99a150ccbde6cbf7ad))

- Add knowledge ([c722b6a](https://github.com/runa-agent/runa/commit/c722b6ae80495b7985bf1edc00b82785453a111a))

- Trace memory/knowledge retrieval and extraction as span ([fd2ed49](https://github.com/runa-agent/runa/commit/fd2ed49ae781568ba13ea01fc507eadff384b934))

- Add escape hatches ([469c8ba](https://github.com/runa-agent/runa/commit/469c8bab201505512db9e6ef9cedf6ca5c5f83df))

- Add cache ([8f1c892](https://github.com/runa-agent/runa/commit/8f1c89222f60eeec205f1ef022280d9ec0a71bf4))

- Add fastapi and docker endpoint ([b53893e](https://github.com/runa-agent/runa/commit/b53893e5108e53acd51cefb2949af32ad1cde2b3))

- Add ui ([135e797](https://github.com/runa-agent/runa/commit/135e7975a7d711f66cb9820debc3faa0a23abd7a))

- Add durable runstate,  sticky approval, replay guard, context forking for delegate call, guardrail audit trail, pydantic ([a2ffb8e](https://github.com/runa-agent/runa/commit/a2ffb8eafa2949f6e017b19b3e35f45990f8a690))

- Update the zen of runa ([1824788](https://github.com/runa-agent/runa/commit/1824788c14ca0f05f5cb65e0275074505579ad24))

- Update the zen of runa ([bbf5030](https://github.com/runa-agent/runa/commit/bbf5030e6f605933c195e9ed9291e6347350c32e))

- Update the zen of runa ([8b1ae52](https://github.com/runa-agent/runa/commit/8b1ae52db714312bbcf8e74dda0c75ce8a3917ac))

- Add issue template ([e3e3207](https://github.com/runa-agent/runa/commit/e3e320709b95e2238a611e6be5a031c7c2e839a0))

- Add issue template ([4d85c92](https://github.com/runa-agent/runa/commit/4d85c92480d4f76b0f8e5965eae066d44577e867))


### Bug Fixes

- Make Executor/AsyncExecutor.run() a no-op on an already-terminal Run, so after_run only fires once ([2581503](https://github.com/runa-agent/runa/commit/2581503544924feb78f7f877cc4dd16a2924371b))

- Make run_later() stamp agent provenance and save to the RunStore before dispatch, so recover_pending() can actually find orphaned run ([c1488b6](https://github.com/runa-agent/runa/commit/c1488b6105efc9db3f043d3807269e2751b403c6))

- Find the current turn's pending tool calls by owning message ([2a2cedf](https://github.com/runa-agent/runa/commit/2a2cedf0b8a89e0ef91bae3ebcd22f0abeee8800))

- Make TOOL_COMPLETED/TOOL_FAILED events and persistence handle non-JSON-safe tool results, and surface them in the timelin ([ffa0e51](https://github.com/runa-agent/runa/commit/ffa0e51a604ecb1c4dc05de93e9b9e66bdadefeb))

- Make Run.state/Run.context/Conversation.state persistence handle non-JSON-safe values ([74fc720](https://github.com/runa-agent/runa/commit/74fc72009c76f229e9460a1a8e4c90746bb57ce1))

- Serialize SQLite store/queue connection access to prevent cross-thread corruption ([d7e35cb](https://github.com/runa-agent/runa/commit/d7e35cbce0147bd58371eec953ff90f407cb6fb1))

- Serialize SQLite store/queue connection access to prevent cross-thread corruption ([36d6c92](https://github.com/runa-agent/runa/commit/36d6c92523ae1416057862af9bacbb6a55df22d5))

- Save the Run's terminal status back to the store when a durable background job finishe ([29269c4](https://github.com/runa-agent/runa/commit/29269c4722773d0c184d266ca31eff8acf60e271))

- Don't let a raising Agent hook or observability subscriber crash or falsify a Run ([2b698c9](https://github.com/runa-agent/runa/commit/2b698c9d1f311db81ec3c39bbc5e971c886dac9a))

- Fall back to str() for a non-JSON-safe Run.input/result instead of crashing persistence ([0366d4d](https://github.com/runa-agent/runa/commit/0366d4d0bccff04cc3925dde5e4e64120acaad80))

- Print a clean error instead of a raw traceback for routine `runa` CLI mistake ([5c06d14](https://github.com/runa-agent/runa/commit/5c06d14c9f2685291abc639745a718f07a911595))

- Identify the pending approval call by event, not by approved==None ([e7120f3](https://github.com/runa-agent/runa/commit/e7120f37c24cec80f8590da68a88f7492d659486))

- Guard Conversation.record() against concurrent tearing, document the real guarantee ([4028d47](https://github.com/runa-agent/runa/commit/4028d47bd7fd844ca15ec0c4e53829184781f746))

- Raise instead of silently corrupting a Run driven by two Executors at once ([50347b0](https://github.com/runa-agent/runa/commit/50347b0278db2c1d797b91cdae467372d6aba478))

- Capture Run-seeding exceptions as Run failures, not silent worker loss ([9dc179e](https://github.com/runa-agent/runa/commit/9dc179e9a5e5ba9629c58b8199f893dad53d857a))

- Set dummy OPENAI_API_KEY before loading eval/plan_and_review examples in tests ([d113207](https://github.com/runa-agent/runa/commit/d113207f9319807ade36ff230d8003ef9a23a372))

- Set dummy OPENAI_API_KEY before loading eval/plan_and_review examples in tests ([256b31d](https://github.com/runa-agent/runa/commit/256b31db7bfde3d9fbccc74609dd0baf54e23108))

- Annotate _Mode.mode to prevent literal widening ([3d76305](https://github.com/runa-agent/runa/commit/3d763059bd379845922c16cb529d35e257bb1939))

- Add missing runa/run.py, left out of the Run-object commit ([0507161](https://github.com/runa-agent/runa/commit/0507161d68724cc9fe81898a861fa522ba10d7bb))

- Default SQLite storage to db/runa.db, creating db/ if missing ([24bafbd](https://github.com/runa-agent/runa/commit/24bafbd72a294441d830f8ca4ec8b1bd6270dfe1))

- Agents and tools cli ([610f59b](https://github.com/runa-agent/runa/commit/610f59ba42f64e1fec24154ee38343b4fe548435))

- Cli ([7d873b9](https://github.com/runa-agent/runa/commit/7d873b9fe25bff203025d71d74119398d6676925))

- Model provider issues ([130295e](https://github.com/runa-agent/runa/commit/130295e42fc9e4cd33606af6816f545e40f34ae8))

- Point uv_build at src/runa module name ([a9db1a4](https://github.com/runa-agent/runa/commit/a9db1a4b5ad3f77f6f4345bae4c20f7eb210998a))

- Correct dev-group self-reference from runa to runa-ai ([351f2c2](https://github.com/runa-agent/runa/commit/351f2c20013ed2dd95b49b7bba2df91526b088d9))

- Correct dev-group self-reference and expose runa.__version__ ([9438832](https://github.com/runa-agent/runa/commit/94388329be4950db3ab7cd1d92e6871cbc930e1d))

- Documentation landing page and uv add runa-ai ([026f585](https://github.com/runa-agent/runa/commit/026f585d797e75f76e3cd73e0a7c9cac8320f545))

- Use zensical, update docstrings and pyproject.yml ([4763a21](https://github.com/runa-agent/runa/commit/4763a21d818f80d84153da1b2b2cb4c829ace71f))


### Refactor

- Rename Run.agent_id to agent_name for consistency with Agent.agent_name()/Tool.tool_name() ([6d3d099](https://github.com/runa-agent/runa/commit/6d3d0992832d3fff0c72f169b9989e61b0bffc60))

- Clean ([9611d38](https://github.com/runa-agent/runa/commit/9611d38fe47292190d54d989f280dd71cf260dbf))

- Remove everything to start from scratch ([7df60a5](https://github.com/runa-agent/runa/commit/7df60a56d3374bf8ae057426f73d9201adb1ec24))

- Drop redundant runa_ prefix from runa.db table name ([f087bef](https://github.com/runa-agent/runa/commit/f087bef32f6f6691e79655feb2ae183191160e1a))

- Extract shared _sqlite.py connect helper for runa.db module ([7396b89](https://github.com/runa-agent/runa/commit/7396b894d8ae4cd5e30e10e89fff4186810f9448))

- Drop .compact() sugar, re-export SDK SQLiteSession directly ([9af9ed4](https://github.com/runa-agent/runa/commit/9af9ed4c3f3392162612b00814a9675cf1a7dc61))

- Add _models/ ([879798d](https://github.com/runa-agent/runa/commit/879798d15118fb33422c2486c8859b305b9198c5))

- Simplifying ([f85ac20](https://github.com/runa-agent/runa/commit/f85ac2032f28351f7558c6b960d50c22d3333b8b))

- Db ([873e728](https://github.com/runa-agent/runa/commit/873e7283321b8a95f881c6710c1c68950bd7fb43))

- Repo structure ([de2eb3a](https://github.com/runa-agent/runa/commit/de2eb3a075e5be43787847b2dd0dd9ba00af8947))


### Documentation

- Establish Runa project foundation ([3c29ecb](https://github.com/runa-agent/runa/commit/3c29ecbef7490bec10ff9198e39c98d62fea2534))

- Establish Runa project foundations ([4b372c8](https://github.com/runa-agent/runa/commit/4b372c8641ae46686d7e41c940e45a601d658dca))

- Define Runa contribution guidelines ([ae6b2c2](https://github.com/runa-agent/runa/commit/ae6b2c2169908be6969f8317979c20a4ea286c97))

- Update README for uv workflow makefile ([1adc1a9](https://github.com/runa-agent/runa/commit/1adc1a945d40e5fed95f93ed323ace7987349ab3))

- Update manifesto and principles7 ([5e9ece2](https://github.com/runa-agent/runa/commit/5e9ece2f8743026b794ba227acdf608072e1d7f3))

- Update manifesto and principles7 ([c821710](https://github.com/runa-agent/runa/commit/c821710e9cd47bc68d0c3179cb72d2df48549964))

- *(architecture.md)* Add architecture ([ad30524](https://github.com/runa-agent/runa/commit/ad305240fb50616b309a42314ba051d90b6a2f3b))

- Update claude.md and readme.md ([13fe902](https://github.com/runa-agent/runa/commit/13fe9028c9bd411fc2ff72b02466a6186333c1ef))

- Sync architecture/concepts/guides/manifesto/rails-to-runa with implementation ([a40ac70](https://github.com/runa-agent/runa/commit/a40ac70b8f8ed4364342ee5cf3ba81c19d1df05d))

- Sync README with Context, Agent.policies, and crash recovery ([5d83faf](https://github.com/runa-agent/runa/commit/5d83faf6bf5d3d3b20d040fa6a1584ea4a412860))

- Stop claiming recover_pending() resumes a Run mid-flight — it restarts i ([6ce248f](https://github.com/runa-agent/runa/commit/6ce248f2a74c4d2674fb14e839dea7ffda11cfdc))

- Document SIGTERM's effect on ThreadQueue/SQLiteQueue background jobs ([774263c](https://github.com/runa-agent/runa/commit/774263c1677d8a079aa241b7777cd576984b6df2))

- Document that Conversation history grows unbounded, with the escape hatch ([e166831](https://github.com/runa-agent/runa/commit/e1668319d99d6c217abf1b8566ac82ab44f98eff))

- Lead with @tool as the preferred simple API in getting_started/guides ([db64646](https://github.com/runa-agent/runa/commit/db646465568846e1f8621ec435f52cbd04de3d81))

- Add examples/observe_run.py demonstrating live and after-the-fact Run observability ([64c6463](https://github.com/runa-agent/runa/commit/64c646367a8bf04821ea3c6133bee8f185bb03d0))

- Update ([48d2b9e](https://github.com/runa-agent/runa/commit/48d2b9e3b15dc402899698f1d4d7ca3aca34d7a5))

- Update ([b1f50cb](https://github.com/runa-agent/runa/commit/b1f50cb4ca63ded90a40f30cb3e810b9b47d834d))

- Update README.md ([37db6c3](https://github.com/runa-agent/runa/commit/37db6c308c338d9eed95b23a973a9f12c6e06499))

- Update ([b1b13b2](https://github.com/runa-agent/runa/commit/b1b13b2641ef3f8fcde3b112e9b88d3ccea2b401))

- Add mkdocs ([8763d3c](https://github.com/runa-agent/runa/commit/8763d3cc7efcc24601d24fdb4921c1259e9157f4))

- Add mkdocs ([630eaae](https://github.com/runa-agent/runa/commit/630eaae8d911b3f0cecae86992834c2ec053af87))

- Update readme ([8e2d1b1](https://github.com/runa-agent/runa/commit/8e2d1b1e1092c43e97cd4dc11e4e5631bba9d882))

- Update mkdocs ([044e748](https://github.com/runa-agent/runa/commit/044e748325de3088c1d4d5ce495d71cc8fe3c7d2))

- Update ([21344f5](https://github.com/runa-agent/runa/commit/21344f56ae356c3cfdcefe21b56f3d28a46d2c53))

- Introduce primitives ([0e6b9c5](https://github.com/runa-agent/runa/commit/0e6b9c5e031ae00f3419ffc954199524312ba4e3))

- Update ([0edf3c2](https://github.com/runa-agent/runa/commit/0edf3c27eb19fb07a54c561f818513f5c209ec55))

- Update ([877790e](https://github.com/runa-agent/runa/commit/877790ebe6d65df3d5d057695e026bef6677ffbc))

- Update ([1f3d8cc](https://github.com/runa-agent/runa/commit/1f3d8cc3c4af27717a4b3be5b1be684a8648e33b))

- Update ([406bf81](https://github.com/runa-agent/runa/commit/406bf81407b6f57e036472a31bbb8cd54855acbf))


### Testing

- Cover Run identity ([41b13a1](https://github.com/runa-agent/runa/commit/41b13a1304e3e4cd52c6ef9642d147d990ee55b0))

- Cover handoff/delegate/auto subagent wiring ([db80949](https://github.com/runa-agent/runa/commit/db8094967eb4e3b424e2c37e46ddd7f7e585fca8))


### Miscellaneous

- Update contribution policy ([9b4b40d](https://github.com/runa-agent/runa/commit/9b4b40d6b74a55ad63d15e15faa43de4f24cc716))


### Other

- Add GitHub Actions checks ([8439f83](https://github.com/runa-agent/runa/commit/8439f83320455802453009bfc135886f34023548))

- Remove python installation ([5dcebed](https://github.com/runa-agent/runa/commit/5dcebed644d2714750b341f5de8cb22a0cbbba89))

- Update ([23bb6bb](https://github.com/runa-agent/runa/commit/23bb6bb3a018ef2741760ecbc4d387db3e159626))

- Add release workflow ([48df812](https://github.com/runa-agent/runa/commit/48df812130efea0451ed152ad33be65b78b26a0b))

- Add release workflow ([195f59a](https://github.com/runa-agent/runa/commit/195f59a39efde0afe18158683d7b5fec1550b2f7))

- Add release workflow ([0c7101c](https://github.com/runa-agent/runa/commit/0c7101cd99af4e381ca3d7720e489fd894cb7d7e))


