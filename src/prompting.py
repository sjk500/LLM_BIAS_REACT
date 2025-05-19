import ast
import re

def make_prompt(context, question, choices):
    choices = ast.literal_eval(choices)
    return f"""문맥을 읽고, 질문에 가장 적절한 답 하나를 고르세요.  
- 문맥에 나온 정보만 기준으로 판단하세요.   
- 반드시 선택지 중 하나를 선택하세요.
- 문맥에 근거가 충분하지 않은 경우에는 '**알 수 없음**'을 선택하세요.  
- 마지막 줄에 정답을 다음 형식으로 작성하세요:
정답: [선택지 번호]. [선택지 내용]

문맥:  
{context}

질문:  
{question}

선택지:  
1. {choices[0]}  
2. {choices[1]}  
3. {choices[2]}

생각:
"""

def extract_answer(text):
    raw_answer = text.split("생각:")[1].strip()
    match = re.search(r"정답:\s*\d+\.\s*([^.]+)\.", raw_answer)
    if match:
        answer = match.group(1).strip()
    else:
        print("⚠️ 정규식 매칭 실패! raw_answer:", raw_answer)
        answer = "오류"
    return raw_answer, answer
