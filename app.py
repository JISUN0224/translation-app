from flask import Flask, render_template, request, jsonify
from google.generativeai import GenerativeModel
import google.generativeai as genai
from dotenv import load_dotenv
import os
import logging
import concurrent.futures

# 로깅 설정
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# 환경 변수 로드
load_dotenv()
api_key = os.getenv('GOOGLE_API_KEY')
logger.debug(f"API Key loaded: {'Yes' if api_key else 'No'}")

if not api_key:
    raise ValueError("GOOGLE_API_KEY environment variable is not set!")

# API 키 설정
genai.configure(api_key=api_key)

# Gemini 모델 초기화
model = GenerativeModel('models/gemini-2.0-flash')

app = Flask(__name__, static_folder='static')

# 번역 유형별 프롬프트 템플릿
PROMPT_TEMPLATES = {
    'daily': """
다음 일상 대화 텍스트를 {target_language}로 번역해주세요. 출발 언어는 {source_language}입니다.

**전반적인 지침:**
* 당신은 경력이 풍부한 번역가입니다. 사용자의 니즈와 요청에 맞추어 도착 언어에 적합한 번역문을 제공합니다.
* 번역할 때, 문맥, 어조, 스타일을 고려하여 자연스럽고 정확한 번역을 제공합니다.
* 원문에 없는 내용은 절대로 임의로 추가하지 않습니다. 
* 번역 결과만 생성하세요. 추가적인 설명이나 프롬프트 지침은 포함하지 마세요.

**번역 지침:**
* 당신은 숙련된 번역가입니다. 원문의 텍스트 성격을 고려하여 그에 맞춘 번역을 제공합니다.
* 자연스럽고, 구어체적인 표현을 사용하여 대화의 흐름과 뉘앙스를 최대한 살려주세요.
* 대화 참여자 간의 관계 (친구, 가족, 동료 등)와 상황을 고려하여 적절한 어조와 존댓말/반말을 사용하세요.
* 속어나 유행어는 문맥에 맞게 적절히 사용하거나, 괄호 안에 설명을 추가하여 이해를 돕습니다.
* 가능하면 원문의 길이를 유지하며, 너무 길거나 짧아지지 않도록 주의하세요.
* 원문에 없는 내용은 절대로 임의로 추가하지 않습니다. 번역 결과만보여주세요.
* **대상 독자:** {target_audience} ({target_audience_instructions})

**추가 요청:** {additional_request}

**원문:** {text}
""",
    'article': """
다음 기사를 {target_language}로 번역해주세요. 출발 언어는 {source_language}입니다.

**전반적인 지침:**
* 당신은 경력이 풍부한 번역가입니다. 사용자의 니즈와 요청에 맞추어 도착 언어에 적합한 번역문을 제공합니다.
* 번역할 때, 문맥, 어조, 스타일을 고려하여 자연스럽고 정확한 번역을 제공합니다.
* 원문에 없는 내용은 절대로 임의로 추가하지 않습니다. 
* 번역 결과만 생성하세요. 추가적인 설명이나 프롬프트 지침은 포함하지 마세요.

**번역 지침:**
1. 최신 전문 용어를 정확하게 사용하고, 필요한 경우 괄호 안에 추가 설명을 넣어주세요.
2. 도착 언어의 기사 문체를 사용하세요. 예를 들어, 한국어는 평서체 (~이다, ~했다, ~다, ~한다)를 사용합니다.
3. 전체적인 문맥을 고려하여, 각 문장의 의미가 명확하게 전달되도록 번역합니다.
4. 지명, 인명, 기관명 등 고유명사는 표준적인 음역을 사용하고, 다른 음역이 있다면 괄호 안에 표기합니다.
5. 숫자, 날짜, 화폐 단위 등은 원문과 동일하게 표기하고, 필요에 따라 괄호 안에 추가 정보를 제공합니다.
6. 원문에 없는 내용은 절대로 임의로 추가하지 않습니다. 
7. 번역 결과만 생성하세요. 추가적인 설명이나 프롬프트 지침은 포함하지 마세요.
8. 경어체를 사용하지 않고, '~다' '~했다'와 같은 평서체를 사용한다.
* **대상 독자:** {target_audience} ({target_audience_instructions})

**추가 요청:** {additional_request}

**원문:** {text}
""",
    'legal': """
다음 법률 텍스트를 {target_language}로 번역해주세요. 출발 언어는 {source_language}입니다.

**전반적인 지침:**
* 당신은 경력이 풍부한 번역가입니다. 사용자의 니즈와 요청에 맞추어 도착 언어에 적합한 번역문을 제공합니다.
* 번역할 때, 문맥, 어조, 스타일을 고려하여 자연스럽고 정확한 번역을 제공합니다.
* 원문에 없는 내용은 절대로 임의로 추가하지 않습니다. 
* 번역 결과만 생성하세요. 추가적인 설명이나 프롬프트 지침은 포함하지 마세요.

**번역 지침:**
1. 법률 용어는 가능한 한 일관된 용어를 사용하고, 다른 용어가 사용될 경우 괄호 안에 원문 용어와 함께 표기합니다.
2. 법률적 정확성을 최우선으로 하며, 모호하거나 애매한 표현은 피합니다.
3. 법률 문서에 적합한 공식적이고 정중한 어조를 유지합니다. 비공식적인 표현이나 약어는 사용하지 않습니다. 
4. 원문의 문장 구조와 논리적 흐름을 최대한 유지합니다.
5. 원문에 없는 내용은 절대로 임의로 추가하지 않습니다. 
6. 번역 결과만 생성하세요. 추가적인 설명이나 프롬프트 지침은 포함하지 마세요.
7. '~다'와 같은 평서체를 사용한다. 
 **대상 독자:** {target_audience} ({target_audience_instructions})

**추가 요청:** {additional_request}

**원문:** {text}
""",
    'literature': """
다음 문학 텍스트를 {target_language}로 번역해주세요. 출발 언어는 {source_language}입니다.

**전반적인 지침:**
* 당신은 경력이 풍부한 번역가입니다. 사용자의 니즈와 요청에 맞추어 도착 언어에 적합한 번역문을 제공합니다.
* 번역할 때, 문맥, 어조, 스타일을 고려하여 자연스럽고 정확한 번역을 제공합니다.
* 원문에 없는 내용은 절대로 임의로 추가하지 않습니다. 
* 번역 결과만 생성하세요. 추가적인 설명이나 프롬프트 지침은 포함하지 마세요.

**번역 지침:**
1. 작가의 의도, 등장인물의 성격, 배경 설정 등을 고려하여 문맥에 맞는 번역을 제공합니다.
2. 원문의 문체와 어조를 최대한 유지하면서, 자연스럽고 읽기 쉬운 문장으로 번역합니다.
3. 비유, 은유, 상징 등 문학적 표현 기법은 가능한 한 원문의 의미와 효과를 유지하도록 번역합니다.
4. 작품의 전체적인 분위기와 인물 관계, 성격 등을 파악하며 그에 맞는 말투를 사용합니다.
5. 원문에 없는 내용은 절대로 임의로 추가하지 않습니다. 
6. 번역 결과만 생성하세요. 추가적인 설명이나 프롬프트 지침은 포함하지 마세요.
7. 시제 사용에 주의하고, '~는', '~가', '~이'를 올바르게 사용합니다.
 **대상 독자:** {target_audience} ({target_audience_instructions})

**추가 요청:** {additional_request}

**원문:** {text}
"""
}

TARGET_AUDIENCES = {
    '어린아이': "최대한 쉬운 어휘로, 친절하게 말하고, 문장을 짧게 끊어서 번역할 것",
    '청소년': "청소년이 이해할 수 있도록 친근한 어휘와 문체로 번역할 것",
    '성인': "자연스럽고, 정확하고 명료한 말투를 사용하고 기본적으로 평서체를 (~다, ~한다)를 사용할 것",
    '전문가': "전문적인 용어와 표현을 정확하게 사용하고, 학술적 또는 전문적인 맥락을 충실히 반영할 것"
}

def call_generate_content_with_timeout(prompt, generation_config, timeout):
    """
    타임아웃을 설정하여 Google Gemini API 호출을 수행합니다.
    """
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(
            lambda: model.generate_content(
                prompt,
                generation_config=generation_config
            )
        )
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            raise Exception("API 요청이 시간 초과되었습니다. 다시 시도해주세요.")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/translate', methods=['POST'])
def translate():
    try:
        data = request.get_json()
        logger.debug(f"Received data: {data}")
        
        text = data.get('text')
        text_type = data.get('type')
        target_language = data.get('target_language', '한국어')
        source_language = data.get('source_language', '자동 감지')
        target_audience = data.get('target_audience', '성인')
        additional_request = data.get('additional_request', '').strip()
        audience_instructions = TARGET_AUDIENCES.get(target_audience, '')

        if not text or not text_type:
            return jsonify({'error': 'Missing text or type'}), 400
            
        logger.debug(f"Translating text type: {text_type}")
        logger.debug(f"Text to translate: {text}")
        
        if text_type not in PROMPT_TEMPLATES:
            raise KeyError(f"Invalid text type: {text_type}")
        
        prompt = PROMPT_TEMPLATES[text_type].format(
            text=text,
            target_language=target_language,
            source_language=source_language,
            target_audience=target_audience,
            target_audience_instructions=audience_instructions,
            additional_request=additional_request
        )
        logger.debug(f"Generated prompt: {prompt}")
        
        # 타임아웃 처리 (30초)
        generation_config = {"temperature": 0.5}
        timeout = 30
        response = call_generate_content_with_timeout(prompt, generation_config, timeout)
        
        if response and hasattr(response, 'text'):
            translated_text = response.text
            logger.debug(f"Received response: {translated_text}")
            return jsonify({'translation': translated_text})
        else:
            raise Exception("번역 결과를 받아올 수 없습니다.")
            
    except KeyError as e:
        logger.error(f"Invalid text type: {e}")
        return jsonify({'error': f"Invalid text type: {str(e)}"}), 400
    except Exception as e:
        logger.error(f"Error occurred: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 10000)))