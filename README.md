# civil-complaint-ai

민원-ai-agent/
├── .github/
│   └── workflows/
│       └── ci.yml
├── src/
│   ├── nodes/
│   │   ├── __init__.py
│   │   ├── civi_complaint_node.py
│   │   ├── refine_query_node.py
│   │   ├── generate_strategy_node.py
│   │   ├── draft_reply_node.py
│   │   ├── verify_reply_node.py
│   │   └── final_answer_node.py
│   ├── graph/
│   │   ├── __init__.py
│   │   └── workflow.py
│   ├── utils/
│   │   ├── __init__.py
│   │   └── rag_helper.py
│   └── models/
│       ├── __init__.py
│       └── state.py
├── tests/
│   ├── __init__.py
│   ├── test_civi_complaint_node.py
│   ├── test_refine_query_node.py
│   ├── test_generate_strategy_node.py
│   ├── test_draft_reply_node.py
│   ├── test_verify_reply_node.py
│   └── test_final_answer_node.py
├── .gitignore
├── README.md
├── requirements.txt
└── pyproject.toml
