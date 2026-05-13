### Log SEARCH_CANDIDATE intent
- Query: Tìm giúp tôi 5 ứng viên phù hợp với vị trí này nhất

- Log trong chatbot service. time: 53s

[HR Graph] INTERNAL: 10 SQL record(s) loaded (for metadata display — Qdrant filter is position-based).
[Router] P3b → RANK (default)
[HR Expansion] strategy=RANK | keywords=[] → calling LLM
[Expansion] TIMEOUT after 4.0s — using fallback
[HR Retrieve] strategy=RANK, mode=INTERNAL, top_n=5 | expanded='Tìm giúp tôi 5 ứng viên phù hợp với vị trí này nhất...' | variants=0 | exclude=0 id(s)
[HybridRetrieve] dense=20, keyword=0 | top_n=5, skills=[]...
[Reranker] 20 chunks → 8 unique IDs → top 5 IDs selected (17 total chunks, field='cvId')
[HybridRetrieve] After rerank: 17 chunks from unique CVs
[HR Retrieve] active_cv_ids set to top-5: [12, 23, 14, 15, 24]
[Retrieval] CV chunk: cvId=12, section=SUMMARY, score=0.55
[Retrieval] CV chunk: cvId=12, section=SKILLS, score=0.53
[Retrieval] CV chunk: cvId=12, section=PROJECTS_SMART_ATTENDANCE, score=0.57
[Retrieval] CV chunk: cvId=12, section=PROJECTS_SMART_ATTENDANCE, score=0.52
[Retrieval] CV chunk: cvId=12, section=EDUCATION, score=0.56
[Retrieval] CV chunk: cvId=12, section=EXECUTIVE_SUMMARY, score=0.54
[Retrieval] CV chunk: cvId=23, section=EXPERIENCE_IT_TECHNICIAN_INTERN, score=0.51
[Retrieval] CV chunk: cvId=23, section=EXECUTIVE_SUMMARY, score=0.50
[Retrieval] CV chunk: cvId=14, section=SUMMARY, score=0.52
[Retrieval] CV chunk: cvId=14, section=PROJECTS, score=0.50
[Retrieval] CV chunk: cvId=14, section=SKILLS, score=0.54
[Retrieval] CV chunk: cvId=14, section=EDUCATION, score=0.53
[Retrieval] CV chunk: cvId=14, section=EXECUTIVE_SUMMARY, score=0.54
[Retrieval] CV chunk: cvId=15, section=EXECUTIVE_SUMMARY, score=0.50
[Retrieval] CV chunk: cvId=15, section=EDUCATION, score=0.52
[Retrieval] CV chunk: cvId=24, section=EXECUTIVE_SUMMARY, score=0.50
[Retrieval] CV chunk: cvId=24, section=LANGUAGES, score=0.51
[Reasoning] Auto-triggering scoring for 5 candidates (strategy=RANK)