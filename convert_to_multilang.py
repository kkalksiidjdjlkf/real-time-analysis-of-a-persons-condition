#!/usr/bin/env python3
"""
Multilingual Dashboard Generator
Converts app_dashboard.html to support English, Russian, and Korean
"""

import json
import re
import sys

# Russian text mappings to be replaced
RUSSIAN_TEXTS = {
    # Headers & Navigation
    'Мониторинг': {'en': 'Monitor', 'ko': '모니터링'},
    'Дашборд': {'en': 'Dashboard', 'ko': '대시보드'},
    'Сессии': {'en': 'Sessions', 'ko': '세션'},
    'Люди': {'en': 'People', 'ko': '사람들'},
    'Настройки': {'en': 'Settings', 'ko': '설정'},
    
    # Monitor section
    'Начать мониторинг': {'en': 'Start Monitoring', 'ko': '모니터링 시작'},
    'Остановить': {'en': 'Stop', 'ko': '중지'},
    'Время:': {'en': 'Time:', 'ko': '시간:'},
    'Уровень звука': {'en': 'Sound Level', 'ko': '음량'},
    'Тихо': {'en': 'Quiet', 'ko': '조용함'},
    'Громко': {'en': 'Loud', 'ko': '큼'},
    
    # Real-time metrics
    'Анализ в реальном времени': {'en': 'Real-Time Analysis', 'ko': '실시간 분석'},
    'Лицо': {'en': 'Face', 'ko': '얼굴'},
    'Людей': {'en': 'People', 'ko': '사람'},
    'Кто это': {'en': 'Who is', 'ko': '인식됨'},
    'Усталость': {'en': 'Fatigue', 'ko': '피로도'},
    'Глаза': {'en': 'Eyes', 'ko': '눈'},
    'Моргания': {'en': 'Blinks', 'ko': '눈 깜빡임'},
    'Голос': {'en': 'Voice', 'ko': '음성'},
    'Стресс': {'en': 'Stress', 'ko': '스트레스'},
    'Контакт глаз': {'en': 'Eye Contact', 'ko': '눈 맞춤'},
    'Улыбка': {'en': 'Smile', 'ko': '미소'},
    'Грусть': {'en': 'Sad', 'ko': '슬픔'},
    'Злость': {'en': 'Anger', 'ko': '분노'},
    'Чёткость речи': {'en': 'Voice Clarity', 'ko': '음성 명확도'},
    'Темп речи': {'en': 'Speech Rate', 'ko': '말하기 속도'},
    'Пульс': {'en': 'Pulse', 'ko': '맥박'},
    'Дыхание': {'en': 'Breathing', 'ko': '호흡'},
    'Кожа': {'en': 'Skin', 'ko': '피부'},
    
    # Dashboard
    'Всего сессий': {'en': 'Total sessions', 'ko': '전체 세션'},
    'Среднее за 10 сессий': {'en': 'Avg last 10', 'ko': '평균(최근 10개)'},
    'Дата и время': {'en': 'Date & time', 'ko': '날짜 & 시간'},
    'Результаты лица': {'en': 'Face Results', 'ko': '얼굴 분석 결과'},
    'Результаты голоса': {'en': 'Voice Results', 'ko': '음성 분석 결과'},
    'Здоровье (DOC.FAI.ME)': {'en': 'Health (DOC.FAI.ME)', 'ko': '건강 (DOC.FAI.ME)'},
    'Сессии за последние 7 дней': {'en': 'Sessions last 7 days', 'ko': '최근 7일 세션'},
    
    # Sessions
    'История сессий': {'en': 'Session History', 'ko': '세션 기록'},
    'Обновить': {'en': 'Refresh', 'ko': '새로고침'},
    'Нет сессий. Начните мониторинг.': {'en': 'No sessions. Start monitoring.', 'ko': '세션이 없습니다. 모니터링을 시작하세요.'},
    
    # People
    'Распознавание лиц': {'en': 'Face Recognition', 'ko': '얼굴 인식'},
    'Добавить человека': {'en': 'Add Person', 'ko': '사람 추가'},
    'Как это работает': {'en': 'How it works', 'ko': '작동 방식'},
    'Нет зарегистрированных лиц. Нажмите «Добавить человека».': {'en': 'No registered faces. Click "Add Person".', 'ko': '등록된 얼굴이 없습니다. "사람 추가"를 클릭하세요.'},
    'Расположите лицо в центре круга': {'en': 'Position face in circle center', 'ko': '원 중심에 얼굴을 위치시키세요'},
    'Сканировать лицо': {'en': 'Scan Face', 'ko': '얼굴 스캔'},
    'Отмена': {'en': 'Cancel', 'ko': '취소'},
    
    # Settings
    'API Сервер': {'en': 'API Server', 'ko': 'API 서버'},
    'URL API': {'en': 'API URL', 'ko': 'API URL'},
    'Сохранить': {'en': 'Save', 'ko': '저장'},
    'Проверить': {'en': 'Test', 'ko': '테스트'},
    'Камера и микрофон': {'en': 'Camera & Microphone', 'ko': '카메라 및 마이크'},
    'Перезапросить разрешения': {'en': 'Re-request Permissions', 'ko': '권한 다시 요청'},
    'Данные': {'en': 'Data', 'ko': '데이터'},
    'Экспорт данных': {'en': 'Export Data', 'ko': '데이터 내보내기'},
    'Очистить данные': {'en': 'Clear Data', 'ko': '데이터 지우기'},
    
    # Video states
    'Камера не активна': {'en': 'Camera inactive', 'ko': '카메라가 비활성상태입니다'},
    'Нажмите «Начать мониторинг»': {'en': 'Click "Start Monitoring"', 'ko': '"모니터링 시작"을 클릭하세요'},
    'Метрики по людям': {'en': 'Person Metrics', 'ko': '개인별 메트릭'},
    'Анализ кожи': {'en': 'Skin Analysis', 'ko': '피부 분석'},
    'Обнаруженные состояния': {'en': 'Detected Conditions', 'ko': '감지된 상태'},
    'Лог событий': {'en': 'Event Log', 'ko': '이벤트 로그'},
    'Система готова. Нажмите «Начать мониторинг».': {'en': 'System ready. Click "Start Monitoring".', 'ko': '시스템 준비 완료. "모니터링 시작"을 클릭하세요.'},
    'Ожидание данных...': {'en': 'Waiting for data...', 'ko': '데이터 대기 중...'},
    'Нет данных': {'en': 'No data', 'ko': '데이터 없음'},
    'Обнаружено': {'en': 'Detected', 'ko': '감지됨'},
    'Неизвестный': {'en': 'Unknown', 'ko': '알 수 없음'},
}

def inject_lang_selector():
    """Create language selector HTML"""
    return '''
    <style>
        .lang-selector-wrapper {
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .lang-selector {
            padding: 8px 12px;
            border: 1px solid var(--border);
            border-radius: 8px;
            background: white;
            cursor: pointer;
            font-size: 13px;
            font-weight: 600;
            color: var(--text);
            font-family: inherit;
        }
        .lang-selector:hover {
            background: var(--light);
        }
    </style>
    <div class="lang-selector-wrapper">
        <select class="lang-selector" id="langSelector" onchange="changeLanguage(this.value)">
            <option value="en">🇺🇸 English</option>
            <option value="ru" selected>🇷🇺 Русский</option>
            <option value="ko">🇰🇷 한국어</option>
        </select>
    </div>
    '''

def inject_translation_system():
    """Create JavaScript translation system"""
    return '''
    <script>
        // Translation system
        let currentLang = localStorage.getItem('selectedLanguage') || 'ru';
        
        const translations = %s;
        
        function t(key) {
            if (translations[currentLang] && translations[currentLang][key]) {
                return translations[currentLang][key];
            }
            if (translations['en'] && translations['en'][key]) {
                return translations['en'][key];
            }
            return key;
        }
        
        function changeLanguage(lang) {
            currentLang = lang;
            localStorage.setItem('selectedLanguage', lang);
            document.documentElement.lang = lang;
            location.reload();
        }
        
        window.addEventListener('DOMContentLoaded', () => {
            document.getElementById('langSelector').value = currentLang;
        });
    </script>
    ''' % json.dumps(translations)

def process_html(html_content):
    """Process HTML to add translation support"""
    
    # Add language selector style
    head_end = html_content.find('</head>')
    if head_end != -1:
        lang_style = '<style>.lang-selector-wrapper{display:flex;align-items:center;gap:8px;}.lang-selector{padding:8px 12px;border:1px solid var(--border);border-radius:8px;background:white;cursor:pointer;font-size:13px;font-weight:600;color:var(--text);font-family:inherit;}.lang-selector:hover{background:var(--light);}</style>'
        html_content = html_content[:head_end] + lang_style + html_content[head_end:]
    
    # Find header and add language selector
    # This needs careful placement - look for status-indicator div
    status_indicator_pos = html_content.find('class="status-indicator"')
    if status_indicator_pos != -1:
        # Find the closing div of status-indicator
        closing_pos = html_content.find('</div>', status_indicator_pos)
        if closing_pos != -1:
            lang_selector_html = '<div class="lang-selector-wrapper"><select class="lang-selector" id="langSelector" onchange="changeLanguage(this.value)"><option value="en">🇺🇸 English</option><option value="ru" selected>🇷🇺 Русский</option><option value="ko">🇰🇷 한국어</option></select></div>'
            html_content = html_content[:closing_pos+6] + lang_selector_html + html_content[closing_pos+6:]
    
    # Add translation system before closing body
    body_end = html_content.rfind('</body>')
    if body_end != -1:
        trans_script = f'''<script>
let currentLang = localStorage.getItem("selectedLanguage") || "ru";
const i18nData = {json.dumps(RUSSIAN_TEXTS)};

function t(key) {{
    if (i18nData[key] && i18nData[key][currentLang]) {{
        return i18nData[key][currentLang];
    }}
    return key;
}}

function changeLanguage(lang) {{
    currentLang = lang;
    localStorage.setItem("selectedLanguage", lang);
    document.documentElement.lang = lang;
    location.reload();
}}

window.addEventListener("DOMContentLoaded", () => {{
    document.getElementById("langSelector").value = currentLang;
}});
</script>'''
        html_content = html_content[:body_end] + trans_script + html_content[body_end:]
    
    return html_content

if __name__ == '__main__':
    # Read original HTML
    with open('/Users/maks/Desktop/макс проекты/монитоинг/app_dashboard.html', 'r', encoding='utf-8') as f:
        html = f.read()
    
    # Process HTML
    html = process_html(html)
    
    # Write to new file (will replace original)
    output_path = '/Users/maks/Desktop/макс проекты/монитоинг/app_dashboard_new.html'
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"✅ Multilingual dashboard created: {output_path}")
    print(f"📊 Supported languages: English, Русский, 한국어")
