# Revision

# Going over each LLM API, calling the correct method according to the documentation,
# implementing caching, error handling, and logging.

# Finished Documents

api_utils/
├── __init__.py
├── config.py                   # Finished
├── anthropic_api.py            # Finished
├── gemini_api.py               # Finished
├── openai_api.py               # Finished
├── logging_utils.py            # Logging utilities
├── pdf_utils.py                # Finished
├── schema_manager.py           # Schema management for API responses
└── token_counter.py            # Token counting utilities

# Todo

Optional
- Fix token_counter.py to take results dictionary as input instead of logging for individual requests.

# Current issues:
- gemini-2.5-flash-lite-preview is using "thinking" when instructed not to.