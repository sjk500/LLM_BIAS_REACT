
# LLM Prompting Project

이 프로젝트는 Meta-LLaMA 3 8B Instruct 모델을 사용하여 주어진 문맥에 대한 질문에 적절한 답을 생성하는 시스템입니다.

## 📁 프로젝트 구조
- `src/`: 주요 기능 코드
- `data/`: 입력 CSV 파일
- `output/`: 추론 결과 저장
- `run.py`: 실행 스크립트

## 🛠 사용 방법

1. 필요한 라이브러리 설치:
```bash
pip install -r requirements.txt



# 🤖 LLM Prompting Project

> Meta-LLaMA 3 8B Instruct 모델을 활용하여 문맥 기반 질의응답 태스크를 수행하는 프로젝트입니다.  
> 문맥을 기반으로 질문에 적절한 선택지를 생성하고, 답변을 추론합니다.

---

## 📂 프로젝트 구조

llm-prompting-project/
├── src/
│ ├── model_loader.py # 모델 로드 함수
│ ├── prompting.py # 프롬프트 생성 및 정답 추출 함수
│ └── inference.py # 전체 추론 실행 함수
├── data/
│ └── test.csv # 입력 데이터 (문맥, 질문, 선택지 포함)
├── output/
│ └── first_submission.csv # 추론 결과 저장 파일
├── run.py # 메인 실행 스크립트
├── requirements.txt # 필요 패키지 목록
└── README.md # 프로젝트 설명

yaml
복사
편집

---

## 🚀 실행 방법

### 1. 의존성 설치

```bash
pip install -r requirements.txt
2. 모델 경로 확인 및 수정
run.py 안의 model_path 변수에 사용 중인 LLaMA 모델 경로를 입력합니다.

3. 실행
bash
복사
편집
python run.py
4. 결과 확인
출력 파일: output/first_submission.csv

열: ID, raw_input, raw_output, answer

🧠 사용 기술
🤗 HuggingFace Transformers

🧪 Meta-LLaMA 3 8B Instruct

🐍 Python, PyTorch

📝 Prompt Engineering

📊 pandas

💡 주요 기능
문맥 기반 프롬프트 생성

LLM 추론 결과 자동 파싱

선택지 중 적절한 정답 추출

대량 inference 루프 처리 및 자동 저장

📌 예시 프롬프트
makefile
복사
편집
문맥:
어떤 사람이 사과를 좋아합니다. 그러나 바나나는 싫어합니다.

질문:
이 사람이 바나나를 좋아하나요?

선택지:
1. 그렇다
2. 아니다
3. 알 수 없다

정답: 2. 아니다
📎 참고 링크
Meta-LLaMA

Transformers 라이브러리

🧑‍💻 만든이
이름	GitHub
김성준 (Sungjun Kim)	sjk500

📝 License
이 프로젝트는 MIT 라이선스를 따릅니다.

yaml
복사
편집

---

## 💬 다음 해줄 수 있는 것

- 이미지, 시각화, 결과 예시 추가  
- GitHub 레포 상단 설명 & Topics 태그 추천  
- README에 LFS 관련 문구 추가

필요하면 "이미지랑 성능표도 넣자" 라고 말해줘,  
깔끔한 README 완성 도와줄게! ✅
