### Log SEARCH_CANDIDATE intent

## Vấn đề 1: Chưa bypass node không cần thiết, response thừa
- Query: Tìm giúp tôi 5 ứng viên phù hợp với vị trí này nhất

- Log trong chatbot service. time: 40s

INFO:     127.0.0.1:62081 - "POST /chatbot/hr/chat/stream HTTP/1.1" 200 OK
[TIMER][HR] load_hr_session_history: 0.70s
[HR Graph] INTERNAL: 10 SQL record(s) loaded (for metadata display — Qdrant filter is position-based).
[TIMER][HR] load_candidate_scope: 2.50s
[Router] P3b → RANK (default)
[TIMER][HR] route_hr_intent: 0.00s
[HR Expansion] strategy=RANK | keywords=[] → skipped (no keyword leg benefit)
[TIMER][HR] query_expansion: 0.00s
[HR Retrieve] strategy=RANK, mode=INTERNAL, top_n=5 | expanded='Tìm giúp tôi 5 ứng viên phù hợp với vị trí này nhất...' | variants=0 | exclude=0 id(s)
[HybridRetrieve] dense=20, keyword=0 | top_n=5, skills=[]...
[Reranker] 20 chunks → 8 unique IDs → top 5 IDs selected (17 total chunks, field='cvId')
[HybridRetrieve] After rerank: 17 chunks from unique CVs
[HR Retrieve] active_cv_ids set to top-5: [12, 23, 14, 15, 24]
[TIMER][HR] retrieve_hr_context: 22.75s
[Retrieval] CV chunk: cvId=12, section=SUMMARY, score=0.55
[Retrieval] CV chunk: cvId=12, section=SKILLS, score=0.53
[Retrieval] CV chunk: cvId=12, section=PROJECTS_SMART_ATTENDANCE, score=0.57
[Retrieval] CV chunk: cvId=12, section=EDUCATION, score=0.56
[Retrieval] CV chunk: cvId=12, section=EXECUTIVE_SUMMARY, score=0.54
[Retrieval] CV chunk: cvId=12, section=PROJECTS_SMART_ATTENDANCE, score=0.52
[Retrieval] CV chunk: cvId=23, section=EXPERIENCE_IT_TECHNICIAN_INTERN, score=0.51
[Retrieval] CV chunk: cvId=23, section=EXECUTIVE_SUMMARY, score=0.50
[Retrieval] CV chunk: cvId=14, section=SUMMARY, score=0.52
[Retrieval] CV chunk: cvId=14, section=SKILLS, score=0.54
[Retrieval] CV chunk: cvId=14, section=PROJECTS, score=0.50
[Retrieval] CV chunk: cvId=14, section=EDUCATION, score=0.53
[Retrieval] CV chunk: cvId=14, section=EXECUTIVE_SUMMARY, score=0.54
[Retrieval] CV chunk: cvId=15, section=EXECUTIVE_SUMMARY, score=0.50
[Retrieval] CV chunk: cvId=15, section=EDUCATION, score=0.52
[Retrieval] CV chunk: cvId=24, section=EXECUTIVE_SUMMARY, score=0.50
[Retrieval] CV chunk: cvId=24, section=LANGUAGES, score=0.51
[TIMER][HR] build_hr_prompts: 0.00s
[Reasoning] Auto-triggering scoring for 5 candidates (strategy=RANK)
[TIMER][HR] llm_hr_reasoning: 5.54s
[TIMER][HR] hr_scoring: 13.33s
[TIMER][HR] save_hr_turn: 0.70s
[TIMER][HR] format_hr_response: 0.00s


- Query: Trong số này, ai có kinh nghiệm với Spring Boot và Docker không?

INFO:     127.0.0.1:54950 - "POST /chatbot/hr/chat/stream HTTP/1.1" 200 OK
[Cache Hit] Restored active_cv_ids: [12, 23, 14, 15, 24]
[Cache Hit] Restored ranked_cv_list: 5 entries
[TIMER][HR] load_hr_session_history: 0.57s
[HR Graph] INTERNAL: 10 SQL record(s) loaded (for metadata display — Qdrant filter is position-based).
[TIMER][HR] load_candidate_scope: 1.18s
[Router] P3a → FILTER
[TIMER][HR] route_hr_intent: 0.00s
[HR Expansion] strategy=FILTER | keywords=['spring', 'docker', 'spring boot'] → calling LLM
[Expansion] TIMEOUT after 4.0s — using fallback
[TIMER][HR] query_expansion: 4.06s
[HR Retrieve] strategy=FILTER, mode=INTERNAL, top_n=5 | expanded='Trong số này, ai có kinh nghiệm với Spring Boot và Docker kh...' | variants=3 | exclude=0 id(s)
[HybridRetrieve] dense=20, keyword=0 | top_n=5, skills=['spring', 'docker', 'spring boot']...
[Reranker] 19 chunks → 6 unique IDs → top 5 IDs selected (18 total chunks, field='cvId')
[HybridRetrieve] After rerank: 18 chunks from unique CVs
[HR Retrieve] active_cv_ids set to top-5: [12, 14, 24, 15, 23]
[TIMER][HR] retrieve_hr_context: 18.58s
[Retrieval] CV chunk: cvId=12, section=SKILLS, score=0.65
[Retrieval] CV chunk: cvId=12, section=SUMMARY, score=0.64
[Retrieval] CV chunk: cvId=12, section=EXECUTIVE_SUMMARY, score=0.65
[Retrieval] CV chunk: cvId=12, section=CAREER_OBJECTIVE, score=0.62
[Retrieval] CV chunk: cvId=12, section=PROJECTS_SMART_ATTENDANCE, score=0.70
[Retrieval] CV chunk: cvId=12, section=EDUCATION, score=0.63
[Retrieval] CV chunk: cvId=14, section=SUMMARY, score=0.67
[Retrieval] CV chunk: cvId=14, section=EXECUTIVE_SUMMARY, score=0.66
[Retrieval] CV chunk: cvId=14, section=CAREER_OBJECTIVE, score=0.63
[Retrieval] CV chunk: cvId=14, section=SKILLS, score=0.68
[Retrieval] CV chunk: cvId=14, section=PROJECTS, score=0.63
[Retrieval] CV chunk: cvId=14, section=EDUCATION, score=0.59
[Retrieval] CV chunk: cvId=24, section=LANGUAGES, score=0.54
[Retrieval] CV chunk: cvId=24, section=EXECUTIVE_SUMMARY, score=0.54
[Retrieval] CV chunk: cvId=24, section=EXPERIENCE_2015_01, score=0.53
[Retrieval] CV chunk: cvId=15, section=EXECUTIVE_SUMMARY, score=0.54
[Retrieval] CV chunk: cvId=15, section=EXPERIENCE_2015_01, score=0.53
[Retrieval] CV chunk: cvId=23, section=EXECUTIVE_SUMMARY, score=0.54
[TIMER][HR] build_hr_prompts: 0.00s
[Reasoning] Auto-triggering scoring for 5 candidates (strategy=FILTER)
[TIMER][HR] llm_hr_reasoning: 7.73s
[TIMER][HR] hr_scoring: 20.46s
[TIMER][HR] save_hr_turn: 0.94s
[TIMER][HR] format_hr_response: 0.00s

## Vấn đề 2: Thiếu thông tin và không confirm được email
Tôi: Gửi mail phỏng vấn cho Chí và A. Thời gian:
Chí: 9 giờ sáng ngày 20 tháng 5 năm 2026
A: 11 giờ sáng ngày 20 tháng 5 năm 2026 
Lời nhắc: Please bring your personal laptop for the test

Chatbot: Để gửi email phỏng vấn, tôi cần xác nhận lại thông tin của hai ứng viên:
1. Pham Minh Chi: chi12345pham@gmail.com
2. (Bị thiếu)

Tôi: Đồng ý, gửi mail đi

Chatbot:
Bạn có chắc muốn gửi email tới các ứng viên sau không?

Pham Minh Chi (chi12345pham@gmail.com)
Nguyen Van A (chipm9101@ut.edu.vn)
Gõ 'Đồng ý' để xác nhận hoặc 'Huỷ' để bỏ qua. (Lại có email của người tôi cần gửi)

Tôi: Đồng ý

Chatbot: Đã gửi email mời phỏng vấn thành công tới Pham Minh Chi (chi12345pham@gmail.com). Đã gửi email mời phỏng vấn thành công tới Nguyen Van A (chipm9101@ut.edu.vn).

Tôi kiểm tra thì đúng là gửi được rồi.
