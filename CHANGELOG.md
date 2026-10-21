# Changelog

All notable changes to Runa are documented here.

## [0.9.0] - 2026-10-21

### Bug Fixes

- Let an MCP server carry its tools' approval and guardrail policy ([10a3588](https://github.com/runa-agent/runa/commit/10a3588ce88ce5d737984edf2533ce18b53d6d0f))

- Keep the content blocks an assistant turn's two slots can't hold ([71133bb](https://github.com/runa-agent/runa/commit/71133bb5ea0444f3fbc1293c585295caa87db2ac))

- Ingest an agent's knowledge at startup, not inside a user's first turn ([906212e](https://github.com/runa-agent/runa/commit/906212e4c8ecaccd8a2172e540bd9e6783f6ce31))

- Commit AgentLike and make CI import what HEAD ships ([39c4e22](https://github.com/runa-agent/runa/commit/39c4e22fcd89a7164e245b56622f05a3681a1fad))

- Build each subagent class once per agent graph, so a cycle closes instead of recursing ([ee01a6f](https://github.com/runa-agent/runa/commit/ee01a6f9a03e30dbe96ec62e6a322fc3177f8259))

- Close every run-internal span on the way out, so a failed step's span records its error ([6a86c5a](https://github.com/runa-agent/runa/commit/6a86c5a2a3088f7ce8292ecc6d95b945fe4fa8fd))

- Raise UserError for every mistake in a declared value, keeping TypeError for what a signature raises ([c692a02](https://github.com/runa-agent/runa/commit/c692a02fbe48a2bff9ddaa791a6f8de6c7628a7f))

- Run a delegate as a nested run of its caller's, sharing its hooks and trace, and drop governance from the tool signature ([ad02e90](https://github.com/runa-agent/runa/commit/ad02e906d7fd21ffc6d631399faceb7ac5f87244))

- Close every closable per-loop resource in runa._loop.run, so run_sync also shuts down a subagent's MCP servers ([c462540](https://github.com/runa-agent/runa/commit/c462540b4b214fe376ae7b798c93e6655c52e175))

- Hand the run's user to tools as RunContext.user_id, so memory=llm searches the same scope as auto ([a7a4d96](https://github.com/runa-agent/runa/commit/a7a4d9677a74848274ae8c376fa7c92f4e6f1fb9))

- Queue a run's trace export instead of awaiting it, flushing in run_sync and at serve shutdown ([21e4a1c](https://github.com/runa-agent/runa/commit/21e4a1c44fbf5c38ee3a908186a335f9af4947a5))

- Close the public surface's one-shape gaps: export Model, split RunStatus/EvalStatus, add Run.tool_calls, drop Run._governance, fix list-shadowed store annotations ([c49913b](https://github.com/runa-agent/runa/commit/c49913bf8b215771edd70cfc3e36767c942c145a))

- Move the conversation's rules into Transcript, so compaction sees rewrites and the loop no longer keeps its invariants ([6ad8599](https://github.com/runa-agent/runa/commit/6ad8599b48ca3d4a95d3cadaedbff9cd9b57e8a5))

- Close every run-internal span on the way out, so a failed step's span records its error ([360c222](https://github.com/runa-agent/runa/commit/360c22231f9f06e0bb29da2c1729894e80ebbe9c))

- Join dashboard link ids with + so CodeQL sees they can't supply a redirect host ([fe3e4d6](https://github.com/runa-agent/runa/commit/fe3e4d6b53911d6048c1f952d1aab7bdb4be7479))

- Give delegates their own Delegate value beside Handoff, and record each delegate call on the caller's Run.tool_calls with the nested calls as children ([4a26e6c](https://github.com/runa-agent/runa/commit/4a26e6c748ed398521c06e99ddc9b27ba576e5ee))

- Build Handoffs in Agent.__init__ like Delegates, so a handoff's tool_name/tool_description override is honoured ([5e6d6c2](https://github.com/runa-agent/runa/commit/5e6d6c221623effe3bb95f2b74bb92ac0fcff4f3))

- Stream Runa-owned TextDelta/ToolCalled/ToolOutput/AgentChanged events instead of a private provider delta ([197ac03](https://github.com/runa-agent/runa/commit/197ac03803a1b42bc47a5021c0c46b8229815d7d))

- Raise UserError for an unknown embedding size, and cut rationale and history from docstrings ([40c4402](https://github.com/runa-agent/runa/commit/40c4402497a0d26f332fbda6f7f85c9911f7a900))

- Keep guardrails and compact as declared on Agent, resolving them on AgentShape so one name has one meaning ([2e01c35](https://github.com/runa-agent/runa/commit/2e01c357e8c42308d5fd7fb95c7e27933dbf2ca2))

- Ci ([eaf6d9b](https://github.com/runa-agent/runa/commit/eaf6d9ba7629ed898cad142c8fddb13634cbd00d))

- Re-apply the schema to a recreated SQLite file even when Linux reuses its inode ([0f18215](https://github.com/runa-agent/runa/commit/0f182157dd5eb0bb1f262594adfad35683566dd9))

- Make tool and agent names unique keys and prefix MCP tools with their server's name ([c1a533c](https://github.com/runa-agent/runa/commit/c1a533c1205fd8dfa01c5ece48dda977e30294e6))

- Charge context sources' model and embedding calls to the run's usage and trace ([41bd93e](https://github.com/runa-agent/runa/commit/41bd93ebebac11e75276b99394dae8920e1a261d))

- Add the untracked runa._metering module that HEAD already imports ([f97e2d3](https://github.com/runa-agent/runa/commit/f97e2d3230eebd036e7f9a5997261fd7b0873856))

- Take user_id on Agent.run instead of Session so session-less runs scope memory per user ([6e668aa](https://github.com/runa-agent/runa/commit/6e668aa854d5bf5339521c95c8e30c9e22ca5203))

- Keep paused runs in runa.db and resume them by a single-use run_id instead of a client-held state blob ([d413adf](https://github.com/runa-agent/runa/commit/d413adf9625a79d1c8505a0637ad1e178cb04e86))

- Skip a context source only on ServiceUnavailable and end the run on anything else, a refused key included ([69c5cd0](https://github.com/runa-agent/runa/commit/69c5cd036844f3c7f59ada05831f6012bf43e3cc))

- Record a session's agent on agent_sessions instead of matching it from the session id prefix ([303a0cf](https://github.com/runa-agent/runa/commit/303a0cf2220993ab6ad8db0f8cd4de4bf1c69228))

- Rename ContextSource.context to retrieve and Case.context to passages, and pass a case's context to its run ([e1fccda](https://github.com/runa-agent/runa/commit/e1fccda25c565c9e1619ca11d3b30d950ad89b83))

- Apply a shared Postgres's schema once per process and let a read create nothing ([5af3551](https://github.com/runa-agent/runa/commit/5af3551cbdb82275c074d82f96f1eee1cb92979b))

- Run every sync call on one background loop so run_sync reuses connections across turns ([d7a09d9](https://github.com/runa-agent/runa/commit/d7a09d9e89b9efd4378af57da9a4894a5f17fee4))

- Keep every pause in runa.db and claim it on resume, so a RunState resumes once in any process ([6715821](https://github.com/runa-agent/runa/commit/67158216c1de9dd6e12cb94d69ae2df96e06cf3a))

- Dispatch hook events as typed Hooks methods so a misspelled event is a type error ([b3ab511](https://github.com/runa-agent/runa/commit/b3ab51179d3152065a1a9eafba6b3128c3f91f01))


### Refactor

- Give "an agent" one type instead of Any and a getattr default ([5bc2946](https://github.com/runa-agent/runa/commit/5bc2946517a76543234898147ec0832b27f1ef2f))

- State each store's query policy twice in one file, not two ([9d7dcdb](https://github.com/runa-agent/runa/commit/9d7dcdb2391a140a3ff66dacdb1cafb3649d6373))

- Hand a guardrail one value, whatever its phase ([3bd0022](https://github.com/runa-agent/runa/commit/3bd0022b592042fe56bc9ec163ca05eca7c23b95))

- Hold which app this process reads in one leaf that db and project both read ([f13b58d](https://github.com/runa-agent/runa/commit/f13b58df30303ade43e1174ed4ac25ef2e7e5c90))

- Close the dict forms of subagents/guardrails and mcp_servers=, one shape per declaration ([6518938](https://github.com/runa-agent/runa/commit/65189384cb5d9f79a8590bd28dc8b345ae200e15))

- Hand each tool guardrail phase its raw value, dropping the input-named envelope ([0fda4c6](https://github.com/runa-agent/runa/commit/0fda4c66461d00cfb1c00205111e47beda6a34b9))

- Hold one run's conversation in a single Transcript shared by _Run, RunState and Run ([f1f6eb7](https://github.com/runa-agent/runa/commit/f1f6eb79eaf50831b67c6453405aecf3fa6253e9))

- Hand user code a RunContext of context and usage, keeping the run's governance internal ([369ac02](https://github.com/runa-agent/runa/commit/369ac02cf25facf9452ab3e83b89bf08a5a52234))

- Hold every process fact in runa._app behind one reset, instead of five globals with their own ([1a76aed](https://github.com/runa-agent/runa/commit/1a76aed6690d39f00e4b73d69f7edbe5e4c1af71))

- Build memory_store and knowledge_store on vector_store, so the vector backend is chosen once ([2813926](https://github.com/runa-agent/runa/commit/28139260fe4ea22549d01e736404e51a1ca414b2))

- Reduce AgentShape to the agent plus its resolved tools, handoffs and model, so the loop reads settings through AgentLike ([aed593b](https://github.com/runa-agent/runa/commit/aed593bd052c5b2368a6f770ca922a9403259cb6))

- Remove RunConfig, read run ceilings off AgentLike and resolve the provider in agent_shape ([10b2ba0](https://github.com/runa-agent/runa/commit/10b2ba022de3e989afa0e03c29567ca37ec51969))

- Give each run a _Frame and one _within boundary, so run, resume and delegate only differ in how they build their frame ([2af05fa](https://github.com/runa-agent/runa/commit/2af05fadd50bf3fbd661d52433511a1e98533941))

- Have Memory and Knowledge hold a VectorStore directly, deleting the MemoryStore/KnowledgeStore layers and their per-backend pairings ([1dc907a](https://github.com/runa-agent/runa/commit/1dc907aa2ab8e1784a5351f0af82a75643953953))

- Keep per-call approval decisions on the shared ApprovalLedger, deleting PausedDelegates' ownership map ([0866f7e](https://github.com/runa-agent/runa/commit/0866f7e2e0c7ee8bbd8ba9241460dc4463941ef1))

- Give Memory and Knowledge one ContextSource contract, so the turn loop keeps only the run's own retrieval policy ([f9be8b5](https://github.com/runa-agent/runa/commit/f9be8b5038020309bc130db5c755a2cca5528ada))

- Have AgentShape compute the ToolSchemas a model is offered, so _models no longer knows what a delegate or handoff is ([0c14abc](https://github.com/runa-agent/runa/commit/0c14abc0724f43d022066dc81e86b3151f5fc3f3))

- Give the runtime one resolved AgentShape built by Agent.__init__ and delete AgentLike ([e08884b](https://github.com/runa-agent/runa/commit/e08884bc668a450dcfb53d194b7d335f83d0689c))

- Hold a run in one durable record the loop works on and RunState serializes as is ([1920dda](https://github.com/runa-agent/runa/commit/1920ddacc7db0838884da05012de7ea1237f4eef))

- Give _Run intent-level methods so the tool helpers stop reaching through its frame ([ce14e20](https://github.com/runa-agent/runa/commit/ce14e202a94697c824b879d991ade02c42c073fd))

- Run memory:// as in-memory SQLite through the SQLite adapters and delete the ephemeral backend ([ea3e473](https://github.com/runa-agent/runa/commit/ea3e473f2523a49a3baaaedad987c27818c5a6e5))

- Narrow Session to get_items, add_items and set_items, and drop pop_item and get_items' limit ([2f0dca6](https://github.com/runa-agent/runa/commit/2f0dca61d39b2cad3c9c9fca20a89a01a770b64a))

- Send embeddings through the model layer's HTTP transport so they share its client, timeout and retries ([bd9faa4](https://github.com/runa-agent/runa/commit/bd9faa4b7e5e6120198be31ed6fc89739836d95e))

- Charge each model call to the innermost open span instead of six explicit meter sites ([171026f](https://github.com/runa-agent/runa/commit/171026f83c470bd8c48c8000dd06e94d55c0348b))

- Put every SQL concern on one async Database handle instead of a SQLite and a Postgres adapter each ([1789711](https://github.com/runa-agent/runa/commit/1789711de16ff1b188e41e790d911e70d634a3b9))

- Move subagent_edges and reachable into agent_shape and read an agent's shape through shape_of ([705a9e0](https://github.com/runa-agent/runa/commit/705a9e04cad4df97dcf37f1d6c6a92a9ef73df8b))

- Give a session its agent when it is resolved instead of the run setting it before a save ([fd5af17](https://github.com/runa-agent/runa/commit/fd5af17d283b9a1b65cb8bb913988e24b077e7c2))

- Make every store async and call run_sync at the sync edges instead of hiding it behind @sync ([56a8af3](https://github.com/runa-agent/runa/commit/56a8af3b1d05dacdd98573dac04b9251ab8bb7dc))

- Move project_path to _app and keep tracing's default policy in tracing behind an on_reset hook ([828dcab](https://github.com/runa-agent/runa/commit/828dcab4ae53a20fbfc2b83df121b4a76c0383c6))

- Replace backend-named Session/Exporter/Cache classes with db.at(url) ([2c473fd](https://github.com/runa-agent/runa/commit/2c473fd9a27b8d3acb34424eb9d5a7fa3e351636))


### Documentation

- Make docstrings state contracts, not history, and fix four that drifted from the code ([1629962](https://github.com/runa-agent/runa/commit/1629962d449511a5a979265bcfe7a24616943b16))

- Fix docstrings and RUNA.md that describe designs the code has moved past ([4824a94](https://github.com/runa-agent/runa/commit/4824a94995501dffbe3aa5f9e3bbeb40a7f8e1df))


### Testing

- Pin that loaded_app refuses a second project on the way in ([fe9d4d8](https://github.com/runa-agent/runa/commit/fe9d4d8eac2fa2930f4eac10ea292a9a171cdbbf))

- Drive the turn loop through Agent.run with a scripted model instead of _run_async] ([c921373](https://github.com/runa-agent/runa/commit/c92137318bdefd94c7fafd4368dfe422305956df))

- Update tests and demo to the required Run.trace and the single mcp setting ([7099ad1](https://github.com/runa-agent/runa/commit/7099ad199be2f3fb3004532805855e9edaa09592))

- Tracing ([495f359](https://github.com/runa-agent/runa/commit/495f359a2e2c72147829ca6cc47a3e2c5d21f4a3))


### Other

- *(deps)* Bump asyncpg from 0.31.0 to 0.32.0 ([341c2f5](https://github.com/runa-agent/runa/commit/341c2f5bce5b2e34a002fc5f0dfbc34217c72f39))

- *(deps)* Bump opentelemetry-exporter-otlp-proto-http ([efd6fde](https://github.com/runa-agent/runa/commit/efd6fde712bad065f2d49a6c763d6c942936331d))

- *(deps)* Bump fastapi from 0.142.2 to 0.143.0 ([80822ef](https://github.com/runa-agent/runa/commit/80822ef7d9d9776a4f81a921cd6863ea3a150662))

- *(deps)* Bump opentelemetry-sdk from 1.45.0 to 1.45.1 ([235d451](https://github.com/runa-agent/runa/commit/235d451bee8fa45ca200e2cbc6aa30e35fc94106))

- *(deps)* Bump anthropic from 1.7.0 to 1.13.0 ([28dbbc5](https://github.com/runa-agent/runa/commit/28dbbc523b7efe20621c64bcd0fea79b8bf15ab0))


## [0.8.0] - 2026-10-13

### Bug Fixes

- Import a served app once, at startup, not on every request ([db583dc](https://github.com/runa-agent/runa/commit/db583dcc6a314977969dd9ebee3d61c91c4f3181))

- Stamp a knowledge corpus after its chunks land, not before ([0269448](https://github.com/runa-agent/runa/commit/026944807d3b35be388f864ca36fadbe0402a419))

- Stop re-raising a closed-loop RuntimeError out of retrieval ([05a629b](https://github.com/runa-agent/runa/commit/05a629b5288961386450185026ecf8c60e48a16d))

- Make loading an app a process-global fact instead of a per-request action ([612b421](https://github.com/runa-agent/runa/commit/612b4218e281b1b30b6afaf5ba76f02b158c78df))

- Honor ModelSettings.reasoning_effort in both backends instead of nothing ([41a4ba0](https://github.com/runa-agent/runa/commit/41a4ba07f38cb956755620ca8ea9dbacf6279df3))

- Drop runa serve --workers, which uvicorn could never honor ([8762792](https://github.com/runa-agent/runa/commit/876279248e1b1f0ac34e6520c5f7b1b730ea4152))

- Keep "errors never raise out of a run" at the boundary, not by exception type ([c7fac35](https://github.com/runa-agent/runa/commit/c7fac352788c9a104f8e769cb88cb814081fe681))

- Export a finished trace off the run's event loop, not on it ([c002422](https://github.com/runa-agent/runa/commit/c00242298202c12af539744960084f2ee89202ae))

- Import a provider's SDK where it's routed to, not at every entry point ([59b09c9](https://github.com/runa-agent/runa/commit/59b09c906764474c3a0d4b4b828a8c332db39e59))

- Run the local session backend off the event loop, and apply its schema once per file ([8a29a55](https://github.com/runa-agent/runa/commit/8a29a55556e7bc55a95ba9d15e1766f74531fed6))

- Build the dashboard's links one way, and serve the ids that way encodes ([ae73cb8](https://github.com/runa-agent/runa/commit/ae73cb8b2892212415284beb49355eb365e4d572))


### Refactor

- Hand out the tracing policy whole instead of one accessor per field ([05c89f2](https://github.com/runa-agent/runa/commit/05c89f25893cf538cf19e4785cb25c92011e4eb1))

- Give _items the two item shapes Runa constructs, not just the readers ([570eaa9](https://github.com/runa-agent/runa/commit/570eaa91ce5b99436e60d40b256552cc22c5e910))

- Give the framework logger its own leaf module, out of Hooks ([a5e8c04](https://github.com/runa-agent/runa/commit/a5e8c04317772179d3f49f7eed78cbd4b06327b0))

- Carry one record per tool call instead of an index-keyed side map ([dfa59ec](https://github.com/runa-agent/runa/commit/dfa59ec5935c3f3c50ba0c46aa5f5872b24ad238))

- Give _items every item shape Runa constructs, envelopes included ([ab2f9e9](https://github.com/runa-agent/runa/commit/ab2f9e900ea1d42487fd54cd48534bba12074088))

- Resolve every Agent setting in one pass over the signature ([4ab5d11](https://github.com/runa-agent/runa/commit/4ab5d11b200757ab5fff12a85d961dc59a07e215))

- Derive the tracing policy's override names from Policy itself ([d6128ad](https://github.com/runa-agent/runa/commit/d6128ad2814019291bce91c272291529d73f0cfa))

- Read the tool-call envelope through _items, as it is already written there ([1a31797](https://github.com/runa-agent/runa/commit/1a31797d0f99be3d890337ff7bada39c8009c3a5))

- Type the runtime-to-Agent seam with a protocol the boundary checks ([3f8cc53](https://github.com/runa-agent/runa/commit/3f8cc532f44f0f8c3ad7001e181bbb6dedb8097d))

- Hoist the query decisions both SQL adapters must share beside their tables ([8250251](https://github.com/runa-agent/runa/commit/82502516d1c1352f703885101598706984eb2385))

- Hoist the query decisions both SQL adapters must share beside their tables ([d8efb6b](https://github.com/runa-agent/runa/commit/d8efb6bad366da02c19e9e95dcd474c6b2f2cf5e))

- Carry a run error's message on RunaError, not in five copies ([4d531e5](https://github.com/runa-agent/runa/commit/4d531e5d36ce9bda5203dc90a423ae0f1e9e4437))

- Say "this process reads one app" once, through project.use ([147451c](https://github.com/runa-agent/runa/commit/147451c52d0125a7630e30c68ca21bdcecc352bd))

- Read the JSON object a model meant in one place, with one failure ([da7c6e6](https://github.com/runa-agent/runa/commit/da7c6e6409f37c0f934436c1680be512ff5daa6f))


### Testing

- Export ([42c6611](https://github.com/runa-agent/runa/commit/42c6611570f831fe28eeeff737f1dc90db32d825))


## [0.7.0] - 2026-10-11

### Features

- Give make release a recipe, stopping at the tag ([0f8dcc9](https://github.com/runa-agent/runa/commit/0f8dcc93aa0a11184e243b4ef89b72617ee2400d))

- Reconcile an existing database against its declaration on every connect ([a59333f](https://github.com/runa-agent/runa/commit/a59333f7044f95c5d3b45fab0ac45ad591164dea))


### Bug Fixes

- Version the knowledge corpus in its store instead of per Knowledge instance ([55fe23e](https://github.com/runa-agent/runa/commit/55fe23e09b91e012dfef1c5cb256e326afbf5b17))

- Resolve compact= at construction, so a bad value fails there instead of mid-run ([33a7b2d](https://github.com/runa-agent/runa/commit/33a7b2d95e8e2cc11a796552313f84c02594dc9e))

- Reject a bad model= at construction, so only resolution waits for the run ([dcd0e5d](https://github.com/runa-agent/runa/commit/dcd0e5de686ee13f8d579f34cdbf13cdce90c23b))

- Treat a user message at index 0 as a cut point, not a missing one ([95b27d5](https://github.com/runa-agent/runa/commit/95b27d5573034a47b973a62e23e20f1e93690b36))

- Grade evals on the run's own tool calls, not policy-filtered spans ([e5a33e8](https://github.com/runa-agent/runa/commit/e5a33e83cbffda8ab905062de855ecdd90492cac))

- Scope the tracing policy to the task, not the process ([db9d173](https://github.com/runa-agent/runa/commit/db9d17330d2a9df2a5642ab49c917ef4b93a1611))

- Let a misspelled MCP option fail at the call site ([e8de80b](https://github.com/runa-agent/runa/commit/e8de80bf0f476261e3920fb04640f330e4c9fa32))

- Refuse an Agent attribute that isn't one of its settings ([9d81135](https://github.com/runa-agent/runa/commit/9d81135f6ba3dcd638ba066391313abdb3d87477))

- Read the prompt file on construction instead of writing it ([4c0bd26](https://github.com/runa-agent/runa/commit/4c0bd26edc244ec85f47e482b53afac9f4cf3f7c))

- Own the Run wire shape on Run instead of re-deriving it in serve ([981add0](https://github.com/runa-agent/runa/commit/981add08997b73b613a7846ad3b5e2080929e447))

- Hold the project root in runa.db so all seven concerns share one file ([724c084](https://github.com/runa-agent/runa/commit/724c08423512070b688ece114a963dbce9c65f7b))

- Hold an MCP session per event loop, as every other client in Runa does ([1a5decd](https://github.com/runa-agent/runa/commit/1a5decde93bb0ec85cff9bc38a9877975804f954))

- Hold the embeddings client per event loop, and stop retrieval swallowing Runa's own bugs ([4dc39d3](https://github.com/runa-agent/runa/commit/4dc39d3647ca039282475946154ff9ccc78839c1))

- Let runa serve take a paused run's state back, on the route it paused on ([0a276a5](https://github.com/runa-agent/runa/commit/0a276a502859962c7e6d2bd302a737905789daed))

- Hash the knowledge corpus off the event loop, and only when a file changed ([e49474c](https://github.com/runa-agent/runa/commit/e49474c4fca1844efea3d51413c3f2a263fd22f3))

- Name a guardrails declaration in the module that owns its grammar ([8294bbb](https://github.com/runa-agent/runa/commit/8294bbb6beab3cf742b07f7fd5aec6ac9c10712d))

- Close the model-name prefix set, so a typo can't pick another provider ([fb9e56b](https://github.com/runa-agent/runa/commit/fb9e56b408189d60c49cd67fb101b79f262783ca))

- Resolve Knowledge's source directory against the project root ([66dff74](https://github.com/runa-agent/runa/commit/66dff74c3760136b718aa16e441de9fe01f7fc21))

- Answer a read's missing tables by testing for them, in one place ([2b7113b](https://github.com/runa-agent/runa/commit/2b7113bf21f83c7d1440cf300b87f39025090aa0))

- Remove .DS_Store ([83bfe69](https://github.com/runa-agent/runa/commit/83bfe69c0994ed810bc30e0d601e4740ebd6875d))

- Read embeddings' endpoint from the provider record, and let a deployment move it ([9a235cd](https://github.com/runa-agent/runa/commit/9a235cdeef378eacac26716e1dafc4d5e37092e8))


### Refactor

- Pass the turn loop its _Run instead of nine copies of its fields ([9438613](https://github.com/runa-agent/runa/commit/9438613382b06e671c3420097e9faf39e3dd4149))

- Resolve each agent's shape once instead of declaring it as a Protocol ([f39bd8b](https://github.com/runa-agent/runa/commit/f39bd8bb025086bfb27512ecedc6c255a2047c5a))

- Name the guardrail phase once instead of writing the 2x2 in nine modules ([90ee8a8](https://github.com/runa-agent/runa/commit/90ee8a8b293e31c1c8675ff3db8a8c4f7b4b3926))

- Declare which failures are the operator's at each raise instead of listing thirteen in the CLI ([b04cd85](https://github.com/runa-agent/runa/commit/b04cd85120dbcc0b128de7a6529f4830dde77956))

- Test the Anthropic backend through its two-method interface instead of its eight translators ([3a3ffb1](https://github.com/runa-agent/runa/commit/3a3ffb16c48c7ee65a9d4c8207e6ca3c65fee711))

- Return an agent filter's escape clause with its pattern, so an adapter can't take half ([3d93af1](https://github.com/runa-agent/runa/commit/3d93af13f4b9f5383de8d27865946ee1ef2ff4a4))

- One ModelRequest per turn, and file names that say what each backend speaks ([c8a4f52](https://github.com/runa-agent/runa/commit/c8a4f52cea1354f31a3d1e44c2c8fc92ff100816))

- Move RunConfig into run_internal, where nothing public takes one ([6c1a322](https://github.com/runa-agent/runa/commit/6c1a32285301d18fb8cd0c51ee2855c4ef07f3d6))

- Name the conversation item for what it is, and give it one reader ([f695120](https://github.com/runa-agent/runa/commit/f6951201794152db852287075f6d8194d43db4f8))

- Have CI run make audit instead of duplicating it ([35b2862](https://github.com/runa-agent/runa/commit/35b2862f035a729321666006320e4c6af2ca882d))

- Log a tool and model call once for both hook scopes ([acf679e](https://github.com/runa-agent/runa/commit/acf679e39d83c166bac055c21898f3b20473be02))

- Resolve a paused tool call by name, not by rebuilding its tool ([9ab2b6e](https://github.com/runa-agent/runa/commit/9ab2b6ecf6ca0a31e82e6c49ea3476d6e8394642))

- Have each generator report its own import line ([a57131f](https://github.com/runa-agent/runa/commit/a57131fcc82d5426df7a63b615e170fe586a183c))

- Name the session write seam Session, not SessionABC ([85fe635](https://github.com/runa-agent/runa/commit/85fe6355da288a331aa2545e91e6bb989229f0db))

- Give the approval protocol an owner in ApprovalLedger ([29aa3e4](https://github.com/runa-agent/runa/commit/29aa3e454c9a1073493da7fd16d97fdb1151472c))

- Grade evals against Run itself, dropping AgentRun and eval/tracing ([ca51b04](https://github.com/runa-agent/runa/commit/ca51b04026431c5d024f66fa9150a0831222a4d1))

- Carry a guardrail's verdict in one GuardrailVerdict, not two types ([043435c](https://github.com/runa-agent/runa/commit/043435c5597f0f36ff7baa3169c60f020ce9ff7c))

- Raise TraceNotFound in one require_trace, not three identical ifs ([57bc5b1](https://github.com/runa-agent/runa/commit/57bc5b1f4e2a4a347572f195771662c3bc5e8f79))

- Give the delegate-pause protocol an owner in PausedDelegates ([3885812](https://github.com/runa-agent/runa/commit/38858124fbdc749e82bc10df6a8e3c2d38b06763))

- Give every resolved agent setting one home, checked by a test ([e7ec9c8](https://github.com/runa-agent/runa/commit/e7ec9c8ad15396fa32385ee1ba74ebc1b1476631))

- Fold _Pending into the RunState it was copied from, and require every schema field ([bc0ec72](https://github.com/runa-agent/runa/commit/bc0ec7239b041ead4404da5317f5cf33677f951d))

- Let Handoff own keying a handoffs list, so run_state stops importing run_internal ([20c7900](https://github.com/runa-agent/runa/commit/20c79005b9c45d1e66350e918381b96b693decf2))

- Collapse RunHooks and AgentHooks into one Hooks class with one set of names ([1c3b97f](https://github.com/runa-agent/runa/commit/1c3b97f9e183b867a38f055e209185a162f85fb2))

- Declare each eval metric once, as data the runner and the report both read ([e439be7](https://github.com/runa-agent/runa/commit/e439be7bd4ce49dd69ad3325d0a09c5bc08d8304))

- State Agent's settings once, as its constructor's signature ([ab1d65e](https://github.com/runa-agent/runa/commit/ab1d65e3e01187dfbd001cc2727152e07be37fa8))

- Walk the agent graph once, where its edges are defined ([ac285fa](https://github.com/runa-agent/runa/commit/ac285fa978eb9429fbc9130294fb7f457eae0426))

- Resolve the package facade one name at a time ([c3d579e](https://github.com/runa-agent/runa/commit/c3d579e528b704a0d6c81e8573c32457abdf4755))

- Name observe's two scopes instead of inferring them from a ([b8e7312](https://github.com/runa-agent/runa/commit/b8e7312c3675ba75ff1e728e4b867cf0b7f79bc1))

- Name observe's two scopes instead of inferring them from a ([73f7a26](https://github.com/runa-agent/runa/commit/73f7a26fd4e5dc3700b3692c8623f3d1bc36eeb1))

- State the prompt-file convention once, in the module that owns it ([6cac605](https://github.com/runa-agent/runa/commit/6cac605b626cebd85c0cb4baadc14d48deb14967))


### Documentation

- Point tracing/config at its real callers instead of the removed SDK's processor ([0d4f70e](https://github.com/runa-agent/runa/commit/0d4f70ef701c17774a45704a89e7ea3f6b5cc26e))

- Record why runa.db repeats its branch per concern ([cba794c](https://github.com/runa-agent/runa/commit/cba794c539c04d4494952dca7f7fc326cd98ba4d))

- Document Agent's nested-run seam by contract, not by caller ([51d1539](https://github.com/runa-agent/runa/commit/51d153988e132389c5499157b0580753b89e71a3))


### Other

- *(deps)* Bump mcp from 2.2.0 to 2.3.0 ([a952dc4](https://github.com/runa-agent/runa/commit/a952dc46ca330c17e9b5d76c5213248fb0d63987))

- *(deps)* Bump httpx2 from 2.12.0 to 2.13.1 ([b55efb9](https://github.com/runa-agent/runa/commit/b55efb93e765b7803707e5ed64e1e1f92fbc3d61))

- *(deps)* Bump fastapi from 0.141.1 to 0.142.2 ([98000bf](https://github.com/runa-agent/runa/commit/98000bf37d925098cad047ea66d0ad86c6cf07d5))

- *(deps)* Bump opentelemetry-sdk from 1.44.0 to 1.45.0 ([17d2c7c](https://github.com/runa-agent/runa/commit/17d2c7cb6a2451c68d9bf61989d7220d66c5651f))

- *(deps-dev)* Bump the dev-dependencies group across 1 directory with 2 updates ([074e950](https://github.com/runa-agent/runa/commit/074e950ec47bae145e20cf857fc57c0b53afd063))


## [0.5.0] - 2026-10-06

### Bug Fixes

- A user's message is never a path to read off disk ([3093d17](https://github.com/runa-agent/runa/commit/3093d1767af50abc20c5cb3b174da5af6da6069d))

- Keep the add-to-evals redirect on this app's own path (CWE-601) ([9cc665b](https://github.com/runa-agent/runa/commit/9cc665bba02c907a2e26bf1f9ef3d3fddaaba79e))

- Send add-to-evals back to its trace, not the sessions list ([8350927](https://github.com/runa-agent/runa/commit/83509276940627c5d4f8d90e164541ec1a9cf973))

- Keep exception detail out of the streaming error event (CWE-209) ([ef5aeb7](https://github.com/runa-agent/runa/commit/ef5aeb724da717e48da4143c0518041229ca7e6d))

- Fire AgentHooks through one dispatch that pairs both hook scopes ([ac06a7e](https://github.com/runa-agent/runa/commit/ac06a7ea68a1bd347ccb6db9f35f490949d408c2))

- Hold the one-conversation latch across run_streamed too ([d3008bf](https://github.com/runa-agent/runa/commit/d3008bf3769ea28c4081db1a83f5bdae2490d2a4))


### Refactor

- Require python 3.14+ and drop the __future__ annotations import ([77d6852](https://github.com/runa-agent/runa/commit/77d6852d20225ef1549d5c02d0536dea3a5c798f))

- Resolve every storage backend from one RUNA_DATABASE_URL ([853d39c](https://github.com/runa-agent/runa/commit/853d39c8d374a68f812feee555ca699a6b61e994))

- Delete runa prune in favor of ON DELETE CASCADE and a documented SQL pass ([dcdafab](https://github.com/runa-agent/runa/commit/dcdafabf4b3ba654ed4c11f4eb15d0db541180d0))

- Delete runa prune in favor of ON DELETE CASCADE and a documented SQL pass ([aac9f6d](https://github.com/runa-agent/runa/commit/aac9f6d7e2908035e813c88156193a72b0f47851))

- Resolve every store once in runa.db, behind one interface per concern ([dc34e36](https://github.com/runa-agent/runa/commit/dc34e36455734ae92f40bee8a299b1e2c2dd626d))

- Delete Runner and RunResult, leaving Agent.run the only seam and Run the only result ([8c6f0ea](https://github.com/runa-agent/runa/commit/8c6f0ea7820b56afcede660baebbfe14fe48e454))

- Hold every loop-bound resource in one LoopCache, keyed per event loop ([91b62b5](https://github.com/runa-agent/runa/commit/91b62b5c720c1fdf06743f73c64f5833900fd7ef))

- Collapse the six vector adapters onto one VectorStore seam ([e1340dd](https://github.com/runa-agent/runa/commit/e1340dd2abd9ef5b3d6c35bd339020873bfcaf9f))

- Vectrostore ([40c45d4](https://github.com/runa-agent/runa/commit/40c45d487c4d5498f0ef189fadd47ce02cd65b81))

- Lift the project module out of cli/ into runa.project ([50690ec](https://github.com/runa-agent/runa/commit/50690ecf44b6a6771981efac8a1a322e03707b85))

- Declare each table once and render it per dialect ([8ad96ad](https://github.com/runa-agent/runa/commit/8ad96ad9d77ba8e2d3138ed7f348888110528374))

- Give Trace one walk() and render both trace views from it ([a135a8f](https://github.com/runa-agent/runa/commit/a135a8fcbdd3ec56b90829a89b0a3ba3d72daaeb))

- Dissolve agent_runner_helpers into its callers, behind one AgentLike ([9330906](https://github.com/runa-agent/runa/commit/93309060813fc5d49f896ff4c71877cc249be8d2))

- Run every guardrail through one runner, and name the tool it's checking ([36efc28](https://github.com/runa-agent/runa/commit/36efc28c72e496d1a05a23ad2fa9d9050bdd97f5))

- Give every Postgres adapter one (url, ddl) base and one @sync pairing ([d6e14a4](https://github.com/runa-agent/runa/commit/d6e14a457b09bb0bc065c57319116a3a5d08fb5f))

- Put the five generators behind one scaffold() and one ScaffoldExists ([a0994ed](https://github.com/runa-agent/runa/commit/a0994ed5553402391c869c93acf1763ed10791b0))

- Pair the in-process memory and knowledge stores in runa.db ([199cc8c](https://github.com/runa-agent/runa/commit/199cc8c9add26463d84b130cd56d137b39ddbda0))


### Documentation

- Scaffold agent skill config for issue tracker, triage labels, and domain docs ([7eee26f](https://github.com/runa-agent/runa/commit/7eee26ff3b04d28a85f00f52378313eea5e70a2c))

- Fill the override example in CLAUDE.md and record the rules its hooks assume ([166db11](https://github.com/runa-agent/runa/commit/166db1157d2f84271464127e9902afc0c7268c1d))

- Point the engineering skills at docs/agents via CLAUDE.md ([1f0e8f2](https://github.com/runa-agent/runa/commit/1f0e8f25b5050b8b37e32d1bbdbdfe6875a61496))


### Testing

- Hold every Cache backend to one contract, and collapse the vec0 pairing into db/vectors.py ([42a9c17](https://github.com/runa-agent/runa/commit/42a9c17c5965687a80298937d87a49f493c632a5))


### Other

- *(deps)* Bump pyjwt from 2.13.0 to 2.15.0 ([2dad2dc](https://github.com/runa-agent/runa/commit/2dad2dc468927fac79fe7e2309028d2e866a1dc4))

- *(deps-dev)* Bump the dev-dependencies group across 1 directory with 3 updates ([27d0651](https://github.com/runa-agent/runa/commit/27d0651a0dcb558ae132e45c940154c3edfa83ef))

- *(deps)* Bump pydantic from 2.13.4 to 2.13.5 ([f74c4fe](https://github.com/runa-agent/runa/commit/f74c4feb95e2870d4de44e5ae6399d86d3696afa))

- *(deps)* Bump anthropic from 1.4.0 to 1.7.0 ([d07b666](https://github.com/runa-agent/runa/commit/d07b6660052cfdeb04ce86d26171ce0d03af7895))

- *(deps)* Bump pypdf from 6.18.0 to 6.19.0 ([5196419](https://github.com/runa-agent/runa/commit/5196419b9b09ba7125979bc5fb4fc89a976c133f))

- *(deps)* Bump mcp from 2.1.1 to 2.2.0 ([8cb7e46](https://github.com/runa-agent/runa/commit/8cb7e468a005026a9d23783a7b70b4cc0a449b53))

- *(deps)* Bump the actions group across 1 directory with 4 updates ([3769472](https://github.com/runa-agent/runa/commit/37694725aba59474f8623e5511c0ce6c33eaaae0))


## [0.4.0] - 2026-09-24

### Features

- Agent.run inside tracing.trace is grouped under it by group_id ([2b91683](https://github.com/runa-agent/runa/commit/2b9168307b5cad49fee20a2a1fef0121110a7e00))

- Production readiness (runa serve, run budgets, shared Postgres, retention) ([4d1c670](https://github.com/runa-agent/runa/commit/4d1c67044869f1405e39643a62c25292039081d1))

- Production readiness (runa serve, run budgets, shared Postgres, retention) ([a6146eb](https://github.com/runa-agent/runa/commit/a6146ebffde9f734f236c4541e928e37289eec28))


### Bug Fixes

- Skip release commits in changelog and drop duplicate entries ([c595938](https://github.com/runa-agent/runa/commit/c595938062445fd83d5af0385982915aed2debc8))

- Message is one user turn, not a transcript ([7244d58](https://github.com/runa-agent/runa/commit/7244d58967bdf9935b939a511d08deefb99ead23))

- Agent.graph no longer needs the Graphviz binary to build ([38f08ee](https://github.com/runa-agent/runa/commit/38f08ee33629abd239008136170cfaf15b685b28))

- A missing optional extra is one clean line, not a traceback ([8305a2f](https://github.com/runa-agent/runa/commit/8305a2f7a301ca308366ba12b4f67e5bd0c81cfd))


## [0.3.0] - 2026-09-24

### Features

- Automate changelog generation with git-cliff ([bdf2df7](https://github.com/runa-agent/runa/commit/bdf2df79e29d2a8bf0d02b6cd31e4d38fbd7829f))

- Support image input in messages, auto-detected from a plain string list ([f6f50eb](https://github.com/runa-agent/runa/commit/f6f50ebd6d9019978a91a76ae7e878a2b5a82cd8))

- Add langfuse integration ([9c40b32](https://github.com/runa-agent/runa/commit/9c40b3248027372519e0ec4642bba40fd0eb16ca))

- Update traces in ui ([3912821](https://github.com/runa-agent/runa/commit/39128210bd354beae19b06602fa6b79ee6f97055))

- Evals/<agent_name>.jsonl is a complete eval, agent resolved from filename ([b35d034](https://github.com/runa-agent/runa/commit/b35d03492f9b7b5e36cd4896badde8a0df0f0c01))

- Evals/<agent_name>.jsonl is a complete eval, agent resolved from filename ([5a2b955](https://github.com/runa-agent/runa/commit/5a2b955aed48ec727b61f76fcfe0cb3a00323ab0))

- Generate agent also creates its evals/<name>.jsonl ([f55955a](https://github.com/runa-agent/runa/commit/f55955a8be261c8b57961a77d9a96af737e15b0e))

- Add traces to evals, flag regressions against the last run, bare-string cases ([dd1ddb5](https://github.com/runa-agent/runa/commit/dd1ddb5fb521d5889c28058a65ce023407b93b12))

- Parallel eval cases, case-to-trace links, regressions in ui, dedupe --add ([f4df943](https://github.com/runa-agent/runa/commit/f4df943e347febce7f744a7e28797ca96d8c9560))

- Run a message's tool calls concurrently, honor parallel_tool_calls=False ([cd65cfd](https://github.com/runa-agent/runa/commit/cd65cfdddb79d4174bf029f57e2c60c590569d9e))

- Run_streamed pauses for approval and resumes from a RunState ([45a16da](https://github.com/runa-agent/runa/commit/45a16da51b7bb7fab210eb56320ace972d01a7dd))

- Agent is the single run API: paused Runs resume, delegates pause their caller, output_type parses, model calls retry ([b1b5ca0](https://github.com/runa-agent/runa/commit/b1b5ca0e6face808783a97b2b0ff7812405c3b5b))

- Run carries the guardrail audit trail for every status ([80e1892](https://github.com/runa-agent/runa/commit/80e1892d44f82467225d735d3e23b7b24e0a21e6))


### Bug Fixes

- Read llm span usage from output, not unused attributes field ([17cb27b](https://github.com/runa-agent/runa/commit/17cb27b5d9153b3cb47d9d90488aef384303d7c6))

- Rebuild RedisCache's client when the event loop changes ([c5f356a](https://github.com/runa-agent/runa/commit/c5f356a30cb9a73ab2d84be1ba0aa3c891921608))

- Async tool call ([63b08aa](https://github.com/runa-agent/runa/commit/63b08aadc11afee87b84c84944448982354ecc45))

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

- Fix links ([68eed32](https://github.com/runa-agent/runa/commit/68eed32d92031bdedb90cc8489666b33198ada67))

- Add examples ([e87e4fe](https://github.com/runa-agent/runa/commit/e87e4fe23e902f7a321bc297a6794813ea94e25f))

- Remove redundancy ([3dae8ab](https://github.com/runa-agent/runa/commit/3dae8abb11b0c7ff7504e05912d0cbf45477f82d))

- Add cache example ([ab417ab](https://github.com/runa-agent/runa/commit/ab417abc3dfac7d6d0c57960daae6cd389ae92aa))

- Mention runa ui at end of tracing section ([3dd8b71](https://github.com/runa-agent/runa/commit/3dd8b711b572b506ceba51278d532c4b55fea167))

- Document loading eval datasets from jsonl ([7afb5c3](https://github.com/runa-agent/runa/commit/7afb5c3f1be76cfa83294770cce64592282b8e42))


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


