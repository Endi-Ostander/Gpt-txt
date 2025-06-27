

===== FILE: C:\Users\Brazers\Desktop\1. стадия1\project_root\config.py =====

# config.py

MEMORY_PATH = "data/memory.json"
KNOWLEDGE_PATH = "data/knowledge_base.json"
LOG_PATH = "data/logs/ai.log"
LOG_LEVEL = "DEBUG"


===== FILE: C:\Users\Brazers\Desktop\1. стадия1\project_root\README.md =====



===== FILE: C:\Users\Brazers\Desktop\1. стадия1\project_root\requirements.txt =====

pyspellchecker
pytest
beautifulsoup4


===== FILE: C:\Users\Brazers\Desktop\1. стадия1\project_root\run.py =====

from core.learner.trainer import Trainer
from tools.dev_console_ext import print_facts, print_knowledge, export_md, export_csv
import sys

def main():
    trainer = Trainer()
    print("Введите текст для обучения ИИ (команды начинаются с /, exit — выход):")

    while True:
        user_input = input("> ").strip()
        if not user_input:
            continue

        if user_input.startswith("/"):
            parts = user_input[1:].split()
            cmd = parts[0].lower()

            if cmd == "exit":
                break
            elif cmd == "mem":
                if len(parts) == 1:
                    print("❗ Уточните: facts, knowledge или export")
                    continue
                subcmd = parts[1].lower()

                if subcmd == "facts":
                    subj = parts[2] if len(parts) > 2 else None
                    print_facts(subj)
                elif subcmd == "knowledge":
                    tag = parts[2] if len(parts) > 2 else None
                    print_knowledge(tag)
                elif subcmd == "export":
                    if len(parts) < 3:
                        print("❗ Укажите формат: md или csv")
                    elif parts[2] == "md":
                        export_md()
                    elif parts[2] == "csv":
                        export_csv()
                    else:
                        print("❗ Неизвестный формат. Используйте md или csv.")
                else:
                    print(f"❗ Неизвестная команда: {subcmd}")
            else:
                print(f"❗ Неизвестная команда: /{cmd}")
        else:
            trainer.process_text(user_input)

if __name__ == "__main__":
    main()


===== FILE: C:\Users\Brazers\Desktop\1. стадия1\project_root\.pytest_cache\README.md =====

# pytest cache directory #

This directory contains data from the pytest's cache plugin,
which provides the `--lf` and `--ff` options, as well as the `cache` fixture.

**Do not** commit this to version control.

See [the docs](https://docs.pytest.org/en/stable/how-to/cache.html) for more information.


===== FILE: C:\Users\Brazers\Desktop\1. стадия1\project_root\core\common\log.py =====

import logging
import os

LOG_DIR = "data/logs"
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(LOG_DIR, "ai.log"),
    filemode='a',
    format='%(asctime)s | %(levelname)s | %(message)s',
    level=logging.DEBUG
)

def log_info(message: str):
    logging.info(message)

def log_error(message: str):
    logging.error(message)

def log_debug(message: str):
    logging.debug(message)


===== FILE: C:\Users\Brazers\Desktop\1. стадия1\project_root\core\common\types.py =====

from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Any, Optional


class PhraseType(Enum):
    """Типы фраз, которые может обрабатывать ИИ."""
    STATEMENT = "statement"    # Утверждение
    QUESTION = "question"      # Вопрос
    COMMAND = "command"        # Команда
    UNKNOWN = "unknown"        # Неопределено


class KnowledgeType(Enum):
    """Типы знаний, сохраняемых в память."""
    FACT = "fact"
    CONCEPT = "concept"
    DEFINITION = "definition"
    RULE = "rule"
    EVENT = "event"


@dataclass
class Phrase:
    """Структура входной фразы."""
    text: str
    tokens: List[str]
    phrase_type: PhraseType
    intent: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class Fact:
    """Факт, который можно сохранить в память."""
    id: str
    subject: str
    predicate: str
    obj: str
    source: Optional[str]
    timestamp: str


@dataclass
class KnowledgeEntry:
    """Хранилище произвольной информации."""
    id: str
    title: str
    content: str
    type: KnowledgeType
    tags: List[str]
    created: str


===== FILE: C:\Users\Brazers\Desktop\1. стадия1\project_root\core\common\utils.py =====

# core/common/utils.py

import json
import os
import re
import uuid
from datetime import datetime, timezone

def resource_path(relative_path: str) -> str:
    """
    Абсолютный путь относительно project_root (корня проекта).
    Работает, даже если скрипт запущен из поддиректории (например, tools/).
    """
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))  # из core/common → project_root
    return os.path.join(base_path, relative_path)

def clean_text(text: str) -> str:
    """Удаляет лишние пробелы, спецсимволы и приводит к нижнему регистру."""
    text = re.sub(r'[^\w\s]', '', text)  # Удалить пунктуацию
    return re.sub(r'\s+', ' ', text).strip().lower()

def normalize_whitespace(text: str) -> str:
    """Приводит все пробелы к одному."""
    return re.sub(r'\s+', ' ', text).strip()

def generate_id() -> str:
    """Создаёт уникальный идентификатор."""
    return str(uuid.uuid4())

def timestamp() -> str:
    """Возвращает текущую дату и время в ISO формате."""
    return datetime.now(timezone.utc).isoformat()

def load_json(filepath: str) -> dict:
    """Загружает JSON-файл в словарь."""
    if not os.path.exists(filepath):
        return {}
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(filepath: str, data: dict) -> None:
    """Сохраняет словарь в JSON-файл."""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def pretty_print(obj):
    """Удобный вывод словаря."""
    print(json.dumps(obj, indent=4, ensure_ascii=False))



===== FILE: C:\Users\Brazers\Desktop\1. стадия1\project_root\core\inputs\html_scraper.py =====

import requests
from bs4 import BeautifulSoup

class HtmlScraper:
    """
    Модуль для сбора текстов с HTML-страниц.
    Минимальная реализация для извлечения текста из тега <body>.
    """
    def fetch_text(self, url: str) -> str:
        try:
            response = requests.get(url)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, "html.parser")
                body = soup.body
                return body.get_text(separator=' ', strip=True) if body else ""
            else:
                print(f"[HtmlScraper] Ошибка загрузки страницы: {response.status_code}")
                return ""
        except Exception as e:
            print(f"[HtmlScraper] Исключение: {e}")
            return ""


===== FILE: C:\Users\Brazers\Desktop\1. стадия1\project_root\core\inputs\text_input.py =====

class TextInput:
    """
    Модуль для получения текстового ввода от пользователя.
    Пока просто заглушка для имитации ввода.
    """
    def get_input(self) -> str:
        return input("Введите текст для ИИ: ")


===== FILE: C:\Users\Brazers\Desktop\1. стадия1\project_root\core\learner\curiosity.py =====

class Curiosity:
    def __init__(self):
        self.unknown_phrases = []

    def add_unknown(self, phrase: str):
        self.unknown_phrases.append(phrase)
        print(f"[Curiosity] Новая непонятная фраза: '{phrase}'")

    def get_questions(self):
        return [f"Что значит: '{p}'" for p in self.unknown_phrases]


    def clear(self):
        self.unknown_phrases.clear()


===== FILE: C:\Users\Brazers\Desktop\1. стадия1\project_root\core\learner\reinforce.py =====

class Reinforce:
    """
    Модуль для подкрепления обучения ИИ.
    Пока заглушка — сюда можно добавить логику усиления изученных знаний,
    например, усиление весов правил или фактов по частоте использования.
    """
    def __init__(self):
        self.rewards = {}

    def reinforce_fact(self, fact_id: str):
        self.rewards[fact_id] = self.rewards.get(fact_id, 0) + 1
        print(f"[Reinforce] Усиление факта {fact_id}: {self.rewards[fact_id]}")

    def get_reward(self, fact_id: str) -> int:
        return self.rewards.get(fact_id, 0)


===== FILE: C:\Users\Brazers\Desktop\1. стадия1\project_root\core\learner\rules.py =====

from typing import List, Optional, Dict, Any
from core.common.types import PhraseType

class Rule:
    """
    Правило для интерпретации фразы и преобразования в знание или действие.
    """
    def __init__(self, name: str, condition: callable, action: callable):
        self.name = name
        self.condition = condition  # Функция: принимает (phrase_type, tokens, raw_text), возвращает bool
        self.action = action        # Функция: принимает (phrase_type, tokens, raw_text), возвращает результат


class RulesEngine:
    def __init__(self):
        self.rules: List[Rule] = []

    def add_rule(self, rule: Rule):
        self.rules.append(rule)

    def apply_rules(self, phrase_type: PhraseType, tokens: List[str], raw_text: str) -> Optional[Dict[str, Any]]:
        """
        Применяет правила к фразе. Возвращает результат первого применимого правила.
        """
        for rule in self.rules:
            if rule.condition(phrase_type, tokens, raw_text):
                return rule.action(phrase_type, tokens, raw_text)
        return None


# Условие и действие для факта с 3 и более токенами
def condition_fact_long(phrase_type: PhraseType, tokens: List[str], raw_text: str) -> bool:
    return phrase_type == PhraseType.STATEMENT and len(tokens) >= 3

def action_fact_long(phrase_type: PhraseType, tokens: List[str], raw_text: str) -> Dict[str, Any]:
    subject = tokens[0]
    predicate = tokens[1]
    obj = " ".join(tokens[2:])
    return {"type": "fact", "subject": subject, "predicate": predicate, "object": obj, "source": raw_text}

# Условие и действие для факта с ровно 2 токенами
def condition_fact_short(phrase_type: PhraseType, tokens: List[str], raw_text: str) -> bool:
    return phrase_type == PhraseType.STATEMENT and len(tokens) == 2

def action_fact_short(phrase_type: PhraseType, tokens: List[str], raw_text: str) -> Dict[str, Any]:
    subject = tokens[0]
    predicate = tokens[1]
    obj = ""  # Объекта нет
    return {"type": "fact", "subject": subject, "predicate": predicate, "object": obj, "source": raw_text}


# Инициализация движка и добавление правил в порядке приоритета
rules_engine = RulesEngine()
rules_engine.add_rule(Rule("fact_rule_long", condition_fact_long, action_fact_long))
rules_engine.add_rule(Rule("fact_rule_short", condition_fact_short, action_fact_short))
# ...внизу rules.py

def condition_question(phrase_type, tokens, raw_text):
    return phrase_type == PhraseType.QUESTION

def action_question(phrase_type, tokens, raw_text):
    return {
        "type": "question",
        "text": raw_text,
        "tokens": tokens
    }

# Добавляем правило (после фактов)
rules_engine.add_rule(Rule("question_rule", condition_question, action_question))


===== FILE: C:\Users\Brazers\Desktop\1. стадия1\project_root\core\learner\trainer.py =====

# core/learner/trainer.py
from core.processor.tokenizer import Tokenizer
from core.processor.classifier import Classifier
from core.memory.memory import Memory
from core.common.types import PhraseType
from core.learner.rules import rules_engine
from core.learner.curiosity import Curiosity
from core.processor.spellchecker import SpellCorrector
from core.common.log import log_info, log_error
from core.common.utils import resource_path



class Trainer:
    def __init__(self):
        self.tokenizer = Tokenizer()
        self.classifier = Classifier()
        self.memory = Memory(
            memory_path=resource_path("data/memory.json"),
            knowledge_path=resource_path("data/knowledge_base.json")
        )
        self.curiosity = Curiosity()
        self.spellchecker = SpellCorrector()

    def process_text(self, text: str):
        """Главная точка обучения: обработка входного текста"""
        phrase_type = self.classifier.classify(text)
        tokens = self.tokenizer.tokenize(text)

        print(f"[Trainer] Тип фразы: {phrase_type.name}")
        print(f"[Trainer] Токены: {tokens}")

        corrected = self.spellchecker.correct_tokens(tokens)

        if corrected != tokens:
            print(f"[Spell] Исправлено: {tokens} → {corrected}")
            tokens = corrected

        result = rules_engine.apply_rules(phrase_type, tokens, text)
        if result:
            if result.get("type") == "fact":
                added = self.memory.add_fact(
                    result["subject"],
                    result["predicate"],
                    result["object"],
                    source=result.get("source")
                )
                if added:
                    print(f"[Trainer] Запомнено: {result['subject']} — {result['predicate']} — {result['object']}")
                    log_info(f"Факт добавлен: {result['subject']} — {result['predicate']} — {result['object']}")
            elif result.get("type") == "question":
                print(f"[Trainer] Вопрос: {result['text']}")
                log_info(f"Вопрос: {result['text']}")
            elif result.get("type") == "concept":
                self.memory.add_knowledge(
                    title=result["title"],
                    content=result["content"],
                    k_type=result["k_type"],
                    tags=result.get("tags", [])
                )
                print(f"[Trainer] Сохранено как знание: {result['title']}")
        else:
            self.curiosity.add_unknown(text)
            print("[Trainer] Непонятная фраза — добавлена в список вопросов.")
            log_error(f"Не удалось обработать: {text}")


===== FILE: C:\Users\Brazers\Desktop\1. стадия1\project_root\core\memory\memory.py =====

import os
from typing import List, Optional
from core.common.utils import load_json, save_json, generate_id, timestamp
from core.common.types import Fact, KnowledgeEntry, KnowledgeType
from core.common.utils import resource_path

MEMORY_FILE = resource_path("data/memory.json")
KNOWLEDGE_FILE = resource_path("data/knowledge_base.json")


class Memory:
    def __init__(self, memory_path: Optional[str] = None, knowledge_path: Optional[str] = None):
        self.memory_path = memory_path or MEMORY_FILE
        self.knowledge_path = knowledge_path or KNOWLEDGE_FILE

        self.facts = self._load_facts()
        self.knowledge = self._load_knowledge()

    def _load_facts(self) -> List[Fact]:
        if not os.path.exists(self.memory_path):
            # Если файл не существует, возвращаем пустой список
            return []
        
        data = load_json(self.memory_path)
        return [Fact(**f) for f in data.get("facts", [])]

    def _load_knowledge(self) -> List[KnowledgeEntry]:
        if not os.path.exists(self.knowledge_path):
            # Если файл не существует, возвращаем пустой список
            return []
        
        data = load_json(self.knowledge_path)
        return [KnowledgeEntry(**k) for k in data.get("entries", [])]

    def _save_facts(self):
        data = {"facts": [f.__dict__ for f in self.facts]}
        save_json(self.memory_path, data)

    def _save_knowledge(self):
        data = {
            "entries": [
                {
                    **k.__dict__,
                    "type": k.type.value if hasattr(k.type, "value") else k.type
                }
                for k in self.knowledge
            ]
        }
        save_json(self.knowledge_path, data)

    def add_fact(self, subject: str, predicate: str, obj: str, source: Optional[str] = None) -> bool:
        for f in self.facts:
            if f.subject == subject and f.predicate == predicate and f.obj == obj:
                print(f"[Memory] Факт уже существует: {subject} — {predicate} — {obj}")
                return False

        fact = Fact(
            id=generate_id(),
            subject=subject,
            predicate=predicate,
            obj=obj,
            source=source,
            timestamp=timestamp()
        )
        self.facts.append(fact)
        self._save_facts()
        print(f"[Memory] Новый факт добавлен: {subject} — {predicate} — {obj}")
        return True

    def add_knowledge(self, title: str, content: str, k_type: KnowledgeType, tags: List[str]):
        entry = KnowledgeEntry(
            id=generate_id(),
            title=title,
            content=content,
            type=k_type,
            tags=tags,
            created=timestamp()
        )
        self.knowledge.append(entry)
        self._save_knowledge()

    def find_facts_by_subject(self, subject: str) -> List[Fact]:
        return [f for f in self.facts if f.subject == subject]

    def find_knowledge_by_tag(self, tag: str) -> List[KnowledgeEntry]:
        return [k for k in self.knowledge if tag in k.tags]


===== FILE: C:\Users\Brazers\Desktop\1. стадия1\project_root\core\memory\recall.py =====

from core.memory.memory import Memory

class Recall:
    """
    Модуль для извлечения и поиска информации из памяти.
    Делегирует функции объекту Memory.
    """
    def __init__(self):
        self.memory = Memory()

    def recall_facts(self, subject: str):
        return self.memory.find_facts_by_subject(subject)

    def recall_knowledge(self, tag: str):
        return self.memory.find_knowledge_by_tag(tag)


===== FILE: C:\Users\Brazers\Desktop\1. стадия1\project_root\core\memory\update.py =====

from core.memory.memory import Memory
from core.common.types import Fact

class UpdateMemory:
    """
    Модуль обновления существующих фактов в памяти.
    Пока реализована простая замена факта по ID.
    """
    def __init__(self):
        self.memory = Memory()

    def update_fact(self, fact_id: str, new_subject: str, new_predicate: str, new_obj: str):
        updated = False
        for i, fact in enumerate(self.memory.facts):
            if fact.id == fact_id:
                self.memory.facts[i] = Fact(fact_id, new_subject, new_predicate, new_obj, fact.source, fact.timestamp)
                updated = True
                break
        if updated:
            self.memory._save_facts()
            print(f"[UpdateMemory] Факт {fact_id} обновлен.")
        else:
            print(f"[UpdateMemory] Факт {fact_id} не найден.")


===== FILE: C:\Users\Brazers\Desktop\1. стадия1\project_root\core\processor\classifier.py =====

# core/processor/classifier.py

from core.common.types import PhraseType


class Classifier:
    def __init__(self):
        self.question_starters = {"что", "кто", "где", "почему", "зачем", "как", "когда", "сколько", "можно", "ли"}
        self.command_verbs = {"расскажи", "покажи", "скажи", "объясни", "запомни", "ответь", "найди"}

    def classify(self, text: str) -> PhraseType:
        lowered = text.strip().lower()
        if not text:
            return PhraseType.UNKNOWN
        if text.startswith("/"):
            return PhraseType.COMMAND
        if all(c == '?' for c in lowered):
            return PhraseType.UNKNOWN

        if lowered.endswith("?"):
            return PhraseType.QUESTION

        if any(lowered.startswith(q) for q in self.question_starters):
            return PhraseType.QUESTION

        if any(lowered.startswith(cmd) for cmd in self.command_verbs):
            return PhraseType.COMMAND

        if lowered.endswith(".") or " " in lowered:
            return PhraseType.STATEMENT

        return PhraseType.UNKNOWN


===== FILE: C:\Users\Brazers\Desktop\1. стадия1\project_root\core\processor\extractor.py =====

from typing import Dict, Any, Optional

class Extractor:
    """
    Модуль для извлечения информации из токенов и текста.
    На этом этапе — простой интерфейс.
    """
    def extract_fact(self, tokens: list, raw_text: str) -> Optional[Dict[str, Any]]:
        if len(tokens) < 3:
            return None
        return {
            "subject": tokens[0],
            "predicate": tokens[1],
            "object": " ".join(tokens[2:]),
            "source": raw_text
        }


===== FILE: C:\Users\Brazers\Desktop\1. стадия1\project_root\core\processor\parser.py =====

# core/processor/parser.py

from core.common.types import Phrase, PhraseType
from core.processor.tokenizer import Tokenizer

class Parser:
    def __init__(self):
        self.tokenizer = Tokenizer()

    def parse(self, text: str) -> Phrase:
        tokens = self.tokenizer.tokenize(text)
        phrase_type = PhraseType.STATEMENT  # временно по умолчанию
        return Phrase(text=text, tokens=tokens, phrase_type=phrase_type)


===== FILE: C:\Users\Brazers\Desktop\1. стадия1\project_root\core\processor\spellchecker.py =====

from spellchecker import SpellChecker

class SpellCorrector:
    def __init__(self, language="ru"):
        self.spell = SpellChecker(language=language)

    def correct_tokens(self, tokens: list) -> list:
        corrected = []
        for word in tokens:
            correction = self.spell.correction(word)
            corrected.append(correction if correction else word)
        return corrected


===== FILE: C:\Users\Brazers\Desktop\1. стадия1\project_root\core\processor\tokenizer.py =====

import re
from typing import List
from core.common.utils import clean_text


class Tokenizer:
    def __init__(self):
        # Можно позже добавить список стоп-слов
        self.delimiters = r"[ \t\n\r\f\v.,!?;:\"()\-—]+"  # Разделители слов

    def tokenize(self, text: str) -> List[str]:
        """Разбивает строку на токены."""
        cleaned = clean_text(text)
        tokens = re.split(self.delimiters, cleaned)
        return [t for t in tokens if t]  # Убираем пустые строки

    def count_tokens(self, text: str) -> int:
        """Подсчитывает количество токенов."""
        return len(self.tokenize(text))

    def preview_tokens(self, text: str) -> None:
        """Выводит токены в консоль для отладки."""
        tokens = self.tokenize(text)
        print(f"[Tokenizer] → {tokens}")


===== FILE: C:\Users\Brazers\Desktop\1. стадия1\project_root\data\knowledge_base.json =====

{
    "entries": [
        {
            "id": "6a2b92d5-003d-4677-9ac8-6dcfa911ca99",
            "title": "Тест",
            "content": "Описание",
            "type": "fact",
            "tags": [
                "тест"
            ],
            "created": "2025-06-25T19:47:05.917364"
        },
        {
            "id": "e41b0ce8-843b-4715-a75e-1fde8280b642",
            "title": "Тест",
            "content": "Описание",
            "type": "fact",
            "tags": [
                "тест"
            ],
            "created": "2025-06-25T19:50:27.315605+00:00"
        },
        {
            "id": "7d57d090-0c66-4c30-8e6a-c9d19044255a",
            "title": "Тест",
            "content": "Описание",
            "type": "fact",
            "tags": [
                "тест"
            ],
            "created": "2025-06-25T19:53:26.690630+00:00"
        }
    ]
}

===== FILE: C:\Users\Brazers\Desktop\1. стадия1\project_root\data\memory.json =====

{
    "facts": [
        {
            "id": "22222222-2222-2222-2222-222222222222",
            "subject": "вода",
            "predicate": "состоит из",
            "obj": "водорода и кислорода",
            "source": "начальные знания",
            "timestamp": "2025-06-25T12:00:00"
        },
        {
            "id": "5d113417-ab1b-42d5-93a5-c1e703651742",
            "subject": "деревья",
            "predicate": "могут",
            "obj": "гореть",
            "source": "Деревья могут гореть",
            "timestamp": "2025-06-25T18:18:49.854477"
        },
        {
            "id": "67bced0c-d0ce-47cb-86ef-089d35bceeea",
            "subject": "дерево",
            "predicate": "горит",
            "obj": "",
            "source": "Дерево горит",
            "timestamp": "2025-06-25T18:25:34.438213"
        },
        {
            "id": "4e447987-563c-4607-99b2-0db4dec3e842",
            "subject": "Тест",
            "predicate": "есть",
            "obj": "данные",
            "source": "тест",
            "timestamp": "2025-06-25T19:26:35.762239"
        },
        {
            "id": "a8c0063d-1108-4741-af2a-000df871027f",
            "subject": "солнце",
            "predicate": "это",
            "obj": "звезда",
            "source": "Солнце это звезда",
            "timestamp": "2025-06-25T19:58:12.328126+00:00"
        },
        {
            "id": "61331c5d-071a-4b9d-829c-cadbd8d5ca76",
            "subject": "\"Солнце\"",
            "predicate": "\"Солнце",
            "obj": "- это звезда, которая является источником света и тепла для Земли.\" [\"астрономия\", \"солнце\",",
            "source": "\"звезды\"]",
            "timestamp": "2025-06-25T20:23:55.058045+00:00"
        },
        {
            "id": "e8a2b6ad-3560-47ce-8d66-7dc8287d89ec",
            "subject": "ты",
            "predicate": "мой",
            "obj": "помощьнык",
            "source": "Ты мой помощьнык",
            "timestamp": "2025-06-26T12:57:27.870724+00:00"
        },
        {
            "id": "8aa2c1c2-2c51-4e2d-894c-4a5e91ae4888",
            "subject": "mem",
            "predicate": "facts",
            "obj": "",
            "source": "mem facts",
            "timestamp": "2025-06-26T13:03:21.926253+00:00"
        },
        {
            "id": "ae477ccc-7500-4cb2-b11b-0974e0e80b9d",
            "subject": "мы",
            "predicate": "лучшые",
            "obj": "друзья",
            "source": "Мы лучшые друзья",
            "timestamp": "2025-06-26T17:15:18.049711+00:00"
        },
        {
            "id": "c18fec1d-5717-48c5-9eaa-e69a09d69d78",
            "subject": "мы",
            "predicate": "лучшие",
            "obj": "друзья",
            "source": "Мы лучшие друзья",
            "timestamp": "2025-06-26T17:15:34.986747+00:00"
        },
        {
            "id": "55608bf1-9e15-49b4-b84c-7171951340de",
            "subject": "мы",
            "predicate": "лучше",
            "obj": "друзья",
            "source": "Мы лучшые друзья",
            "timestamp": "2025-06-26T17:33:53.085594+00:00"
        }
    ]
}

===== FILE: C:\Users\Brazers\Desktop\1. стадия1\project_root\tests\test_classifier.py =====

import pytest
from core.processor.classifier import Classifier
from core.common.types import PhraseType

classifier = Classifier()

def test_classify_command():
    c = Classifier()
    assert c.classify("/run") == PhraseType.COMMAND

def test_classify_unknown():
    c = Classifier()
    assert c.classify("😳?🐍") == PhraseType.UNKNOWN

@pytest.mark.parametrize("text,expected", [
    ("Что ты знаешь?", PhraseType.QUESTION),
    ("Сколько времени?", PhraseType.QUESTION),
    ("Расскажи про солнце", PhraseType.COMMAND),
    ("Солнце — это звезда.", PhraseType.STATEMENT),
    ("???", PhraseType.UNKNOWN),
])
def test_classify(text, expected):
    assert classifier.classify(text) == expected


===== FILE: C:\Users\Brazers\Desktop\1. стадия1\project_root\tests\test_curiosity.py =====

from core.learner.curiosity import Curiosity

def test_add_and_get_questions():
    c = Curiosity()
    c.add_unknown("Квантовая запутанность?")
    questions = c.get_questions()
    assert questions == ["Что значит: 'Квантовая запутанность?'"]

def test_clear():
    c = Curiosity()
    c.add_unknown("abc")
    c.clear()
    assert c.get_questions() == []


===== FILE: C:\Users\Brazers\Desktop\1. стадия1\project_root\tests\test_html_scraper.py =====

from core.inputs.html_scraper import HtmlScraper

def test_fetch_empty_html(monkeypatch):
    class FakeResponse:
        text = ""
    def fake_get(*args, **kwargs):
        return FakeResponse()
    monkeypatch.setattr("requests.get", fake_get)
    scraper = HtmlScraper()
    result = scraper.fetch_text("http://test")
    assert result == ""

def test_fetch_text_invalid_url():
    scraper = HtmlScraper()
    result = scraper.fetch_text("http://invalid.url")
    assert isinstance(result, str)


===== FILE: C:\Users\Brazers\Desktop\1. стадия1\project_root\tests\test_memory.py =====

import tempfile
import shutil
import os
from core.memory.memory import Memory
from core.common.types import KnowledgeType
from core.common.utils import generate_id, timestamp

def test_find_facts_by_subject(tmp_path):
    memory_file = tmp_path / "memory.json"
    knowledge_file = tmp_path / "knowledge.json"
    memory = Memory(memory_path=str(memory_file), knowledge_path=str(knowledge_file))

    # Добавляем тестовый факт
    memory.add_fact("вода", "является", "жидкостью")
    
    results = memory.find_facts_by_subject("вода")
    assert len(results) == 1
    assert results[0].subject == "вода"


def test_find_facts_by_subject(tmp_path):
    mem = Memory(
        memory_path=str(tmp_path / "memory.json"),
        knowledge_path=str(tmp_path / "knowledge.json")
    )
    mem.add_fact("вода", "является", "жидкостью")
    results = mem.find_facts_by_subject("вода")
    assert len(results) == 1

def test_memory_fact_storage():
    tmp_dir = tempfile.mkdtemp()
    memory_file = os.path.join(tmp_dir, "memory.json")
    knowledge_file = os.path.join(tmp_dir, "knowledge_base.json")

    # Создаём пустые файлы для памяти
    with open(memory_file, "w", encoding="utf-8") as f:
        f.write('{"facts": []}')
    with open(knowledge_file, "w", encoding="utf-8") as f:
        f.write('{"entries": []}')

    # Переопределяем пути
    memory = Memory(memory_path=memory_file, knowledge_path=knowledge_file)

    # Добавляем факт
    memory.add_fact("Тест", "есть", "данные", "тест")
    
    # Проверяем, что факт был добавлен
    facts = memory.find_facts_by_subject("Тест")
    assert len(facts) == 1
    assert facts[0].predicate == "есть"

    shutil.rmtree(tmp_dir)

def test_memory_knowledge_storage():
    tmp_dir = tempfile.mkdtemp()
    memory_file = os.path.join(tmp_dir, "memory.json")
    knowledge_file = os.path.join(tmp_dir, "knowledge_base.json")

    # Очистка временных файлов
    with open(memory_file, "w", encoding="utf-8") as f:
        f.write('{"facts": []}')
    with open(knowledge_file, "w", encoding="utf-8") as f:
        f.write('{"entries": []}')

    # Переопределяем пути
    memory = Memory(memory_path=memory_file, knowledge_path=knowledge_file)

    # Добавляем знания
    memory.add_knowledge("Тест", "Описание", KnowledgeType.FACT, ["тест"])
    
    # Проверяем, что знания добавлены
    found = memory.find_knowledge_by_tag("тест")
    assert len(found) == 1

    shutil.rmtree(tmp_dir)


===== FILE: C:\Users\Brazers\Desktop\1. стадия1\project_root\tests\test_parser.py =====

from core.processor.parser import Parser
from core.common.types import Phrase

def test_parse_structure():
    parser = Parser()
    phrase = parser.parse("Солнце это звезда")
    assert isinstance(phrase, Phrase)
    assert phrase.tokens == ["солнце", "это", "звезда"]  # ⬅ заменили на нижний регистр


===== FILE: C:\Users\Brazers\Desktop\1. стадия1\project_root\tests\test_reinforce.py =====

from core.learner.reinforce import Reinforce

def test_reinforce_fact():
    r = Reinforce()
    r.reinforce_fact("f123")
    assert r.get_reward("f123") == 1
    r.reinforce_fact("f123")
    assert r.get_reward("f123") == 2

def test_no_reward():
    r = Reinforce()
    assert r.get_reward("unknown") == 0


===== FILE: C:\Users\Brazers\Desktop\1. стадия1\project_root\tests\test_rules.py =====

from core.learner.rules import rules_engine
from core.common.types import PhraseType

def test_long_fact_rule():
    result = rules_engine.apply_rules(PhraseType.STATEMENT, ["солнце", "является", "звездой"], "Солнце является звездой")
    assert result["type"] == "fact"
    assert result["subject"] == "солнце"

def test_short_fact_rule():
    result = rules_engine.apply_rules(PhraseType.STATEMENT, ["вода", "мокрая"], "Вода мокрая")
    assert result["type"] == "fact"
    assert result["subject"] == "вода"

def test_question_rule():
    result = rules_engine.apply_rules(PhraseType.QUESTION, ["что", "такое", "вода"], "Что такое вода?")
    assert result["type"] == "question"
    assert result["text"] == "Что такое вода?"

def test_no_match_rule():
    result = rules_engine.apply_rules(PhraseType.UNKNOWN, ["..."], "???")
    assert result is None


===== FILE: C:\Users\Brazers\Desktop\1. стадия1\project_root\tests\test_spellchecker.py =====

from core.processor.spellchecker import SpellCorrector

def test_correct_known_word():
    sc = SpellCorrector()
    assert sc.correct_tokens(["помощнык"]) == ["помощник"]

def test_does_not_change_correct():
    sc = SpellCorrector()
    assert sc.correct_tokens(["друзья"]) == ["друзья"]

def test_mixed_batch():
    sc = SpellCorrector()
    assert sc.correct_tokens(["лучшые", "помощнык", "мы"]) == ["лучше", "помощник", "мы"]


===== FILE: C:\Users\Brazers\Desktop\1. стадия1\project_root\tests\test_text_input.py =====

from core.inputs.text_input import TextInput

def test_text_input_stub():
    ti = TextInput()
    assert hasattr(ti, "get_input")


===== FILE: C:\Users\Brazers\Desktop\1. стадия1\project_root\tests\test_tokenizer.py =====

import pytest
from core.processor.tokenizer import Tokenizer

tokenizer = Tokenizer()

def test_clean_text():
    from core.common.utils import clean_text
    assert clean_text("Привет, мир!!!") == "привет мир"

def test_split_symbols():
    t = Tokenizer()
    result = t.tokenize("Ты — мой друг.")
    assert "друг" in result

def test_tokenize_basic():
    text = "Привет, мир!"
    tokens = tokenizer.tokenize(text)
    assert tokens == ["привет", "мир"]

def test_tokenize_empty():
    assert tokenizer.tokenize("") == []

def test_tokenize_whitespace():
    assert tokenizer.tokenize("   \n\t") == []


===== FILE: C:\Users\Brazers\Desktop\1. стадия1\project_root\tests\test_utils.py =====

from core.common.utils import clean_text, normalize_whitespace, generate_id, timestamp
import os
from core.common.utils import save_json, load_json

def test_save_and_load_json(tmp_path):
    filepath = tmp_path / "test.json"
    data = {"a": 1}
    save_json(str(filepath), data)
    loaded = load_json(str(filepath))
    assert loaded == data

def test_clean_text():
    text = " Привет, мир!!! "
    assert clean_text(text) == "привет мир"

def test_normalize_whitespace():
    text = "это   тест\n\nстроки"
    assert normalize_whitespace(text) == "это тест строки"

def test_generate_id_unique():
    id1 = generate_id()
    id2 = generate_id()
    assert id1 != id2 and isinstance(id1, str)

def test_timestamp_format():
    ts = timestamp()
    assert "T" in ts  # ISO формат


===== FILE: C:\Users\Brazers\Desktop\1. стадия1\project_root\tests\__init__.py =====



===== FILE: C:\Users\Brazers\Desktop\1. стадия1\project_root\tools\debug_console.py =====

import sys
from core.memory.memory import Memory
from core.processor.tokenizer import Tokenizer
from core.processor.classifier import Classifier

memory = Memory()
tokenizer = Tokenizer()
classifier = Classifier()

print("🔧 Debug Console — type 'exit' to quit.\n")

while True:
    text = input("> ")
    if text.strip().lower() == "exit":
        break

    tokens = tokenizer.tokenize(text)
    phrase_type = classifier.classify(text)

    print(f"[Debug] Тип: {phrase_type.name}")
    print(f"[Debug] Токены: {tokens}")

    facts = memory.find_facts_by_subject(tokens[0]) if tokens else []
    if facts:
        print(f"[Debug] Найдено фактов по '{tokens[0]}':")
        for f in facts:
            print(f"    - {f.subject} — {f.predicate} — {f.obj}")
    else:
        print("[Debug] Факты по субъекту не найдены.")


===== FILE: C:\Users\Brazers\Desktop\1. стадия1\project_root\tools\dev_console.py =====

import sys
from core.memory.memory import Memory
from core.processor.tokenizer import Tokenizer
from core.processor.classifier import Classifier
from core.common.types import KnowledgeType

memory = Memory()
tokenizer = Tokenizer()
classifier = Classifier()

HELP = """
🧪 Dev Console — команды:
  > help               — показать список команд
  > exit               — выйти
  > mem facts          — показать все факты
  > mem knowledge      — показать все знания
  > mem export         — сохранить отчет в memory_report.md
  > test <фраза>       — токенизация и классификация
  > <фраза>            — обычная фраза для анализа
"""

def print_facts():
    if not memory.facts:
        print("❌ Факты не найдены.")
        return
    for f in memory.facts:
        print(f"- {f.subject} — {f.predicate} — {f.obj}  | from: {f.source or 'n/a'}")

def print_knowledge():
    if not memory.knowledge:
        print("❌ Знания не найдены.")
        return
    for k in memory.knowledge:
        print(f"- {k.title} [{k.type}] — {', '.join(k.tags)}")
        print(f"  > {k.content}")

def export_to_md():
    with open("memory_report.md", "w", encoding="utf-8") as f:
        f.write("# 🧠 Память ИИ\n\n")
        f.write("## 📌 Факты\n\n")
        for fact in memory.facts:
            f.write(f"- **{fact.subject}** — *{fact.predicate}* — {fact.obj}\n")
            f.write(f"  Источник: `{fact.source}`\n")
            f.write(f"  Время: `{fact.timestamp}`\n\n")
        f.write("## 📚 Знания\n\n")
        for entry in memory.knowledge:
            f.write(f"- **{entry.title}** ({entry.type})\n")
            f.write(f"  > {entry.content}\n")
            f.write(f"  Теги: {', '.join(entry.tags)}\n")
            f.write(f"  Время: `{entry.created}`\n\n")
    print("✅ Отчёт сохранён: memory_report.md")

print(HELP)
while True:
    try:
        text = input("> ").strip()
        if text.lower() == "exit":
            break
        elif text.lower() == "help":
            print(HELP)
        elif text.lower() == "mem facts":
            print_facts()
        elif text.lower() == "mem knowledge":
            print_knowledge()
        elif text.lower() == "mem export":
            export_to_md()
        elif text.lower().startswith("test "):
            phrase = text[5:]
            tokens = tokenizer.tokenize(phrase)
            p_type = classifier.classify(phrase)
            print(f"[Test] Тип: {p_type.name}")
            print(f"[Test] Токены: {tokens}")
        else:
            tokens = tokenizer.tokenize(text)
            p_type = classifier.classify(text)
            print(f"[Debug] Тип: {p_type.name}")
            print(f"[Debug] Токены: {tokens}")
            facts = memory.find_facts_by_subject(tokens[0]) if tokens else []
            if facts:
                print(f"[Debug] Факты по '{tokens[0]}':")
                for f in facts:
                    print(f"  - {f.subject} — {f.predicate} — {f.obj}")
            else:
                print("[Debug] Факты не найдены.")

    except KeyboardInterrupt:
        print("\n🛑 Выход.")
        break
    except Exception as e:
        print(f"⚠️ Ошибка: {e}")


===== FILE: C:\Users\Brazers\Desktop\1. стадия1\project_root\tools\dev_console_ext.py =====

import sys
import os
import csv
import subprocess

# Подключение core-модулей
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.memory.memory import Memory
from core.processor.tokenizer import Tokenizer
from core.processor.classifier import Classifier
from core.common.types import KnowledgeType


def resource_path(relative_path):
    """Абсолютный путь к ресурсу относительно project_root/."""
    script_dir = os.path.abspath(os.path.dirname(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))
    return os.path.join(project_root, relative_path)


# Инициализация
memory = Memory(
    memory_path=resource_path("data/memory.json"),
    knowledge_path=resource_path("data/knowledge_base.json")
)

tokenizer = Tokenizer()
classifier = Classifier()

HELP = """
🧪 Расширенная Dev Console (с командами через /):

  📄 Общие команды:
    /help                        — показать эту справку
    /exit                        — выйти из консоли
    /run                         — запустить run.py

  📦 Работа с памятью:
    /mem facts                   — показать все факты
    /mem facts <субъект>         — найти факты по субъекту
    /mem knowledge               — показать все знания
    /mem knowledge <тег>         — найти знания по тегу
    /mem export md|csv           — экспорт памяти

  ➕ Добавление и удаление:
    /add fact <subj> <pred> <obj> [source] — добавить факт
    /del fact <id>                         — удалить факт по ID

  🧪 Тестирование:
    /test <фраза>              — токенизация и классификация

  🔍 Пример:
    /add fact солнце есть звезда Википедия
"""

def print_facts(filter_subj=None):
    filtered = memory.facts
    if filter_subj:
        filtered = [f for f in filtered if filter_subj.lower() in f.subject.lower()]
    if not filtered:
        print("❌ Факты не найдены.")
        return
    for f in filtered:
        print(f"- ID:{f.id} | {f.subject} — {f.predicate} — {f.obj} | from: {f.source or 'n/a'}")

def print_knowledge(filter_tag=None):
    filtered = memory.knowledge
    if filter_tag:
        filtered = [k for k in filtered if filter_tag.lower() in map(str.lower, k.tags)]
    if not filtered:
        print("❌ Знания не найдены.")
        return
    for k in filtered:
        print(f"- ID:{k.id} | {k.title} [{k.type.value}] — {', '.join(k.tags)}")
        print(f"  > {k.content}")

def export_md():
    with open("memory_report.md", "w", encoding="utf-8") as f:
        f.write("# 🧠 Память ИИ\n\n## 📌 Факты\n\n")
        for fact in memory.facts:
            f.write(f"- **{fact.subject}** — *{fact.predicate}* — {fact.obj}\n")
            f.write(f"  Источник: `{fact.source}`\n")
            f.write(f"  Время: `{fact.timestamp}`\n\n")
        f.write("## 📚 Знания\n\n")
        for entry in memory.knowledge:
            f.write(f"- **{entry.title}** ({entry.type.value})\n")
            f.write(f"  > {entry.content}\n")
            f.write(f"  Теги: {', '.join(entry.tags)}\n")
            f.write(f"  Время: `{entry.created}`\n\n")
    print("✅ Отчёт сохранён в memory_report.md")

def export_csv():
    with open("memory_report.csv", "w", encoding="utf-8", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Type", "ID", "Subject/Title", "Predicate", "Object/Content", "Source/Tags", "Timestamp/Created"])
        for fact in memory.facts:
            writer.writerow(["Fact", fact.id, fact.subject, fact.predicate, fact.obj, fact.source or "", fact.timestamp])
        for entry in memory.knowledge:
            tags = ";".join(entry.tags)
            writer.writerow(["Knowledge", entry.id, entry.title, "", entry.content, tags, entry.created])
    print("✅ Отчёт сохранён в memory_report.csv")

def add_fact(args):
    if len(args) < 3:
        print("⚠️ Нужно минимум 3 аргумента: subject predicate object [source]")
        return
    subject = args[0]
    predicate = args[1]
    rest = args[2:]
    obj = " ".join(rest[:-1]) if len(rest) > 1 else rest[0]
    source = rest[-1] if len(rest) > 1 else None
    memory.add_fact(subject, predicate, obj, source)
    print(f"✅ Факт добавлен: {subject} — {predicate} — {obj}")

def del_fact(args):
    if len(args) != 1:
        print("⚠️ Нужно указать ID факта для удаления.")
        return
    fid = args[0]
    before = len(memory.facts)
    memory.facts = [f for f in memory.facts if f.id != fid]
    memory._save_facts()
    if len(memory.facts) < before:
        print(f"✅ Факт с ID {fid} удалён.")
    else:
        print(f"❌ Факт с ID {fid} не найден.")

def main():
    print(HELP)
    while True:
        try:
            line = input("> ").strip()
            if not line:
                continue

            if line.startswith("/"):
                parts = line[1:].split()
                cmd = parts[0].lower()

                if cmd == "exit":
                    break
                elif cmd == "help":
                    print(HELP)
                elif cmd == "run":
                    run_path = resource_path("run.py")
                    if os.path.exists(run_path):
                        subprocess.run([sys.executable, run_path])
                        # После выхода перезагрузить память
                        memory.facts = memory._load_facts()
                        memory.knowledge = memory._load_knowledge()
                    else:
                        print(f"⚠️ Файл run.py не найден по пути: {run_path}")
                elif cmd == "mem":
                    if len(parts) < 2:
                        print("⚠️ Укажите аргумент: facts, knowledge или export")
                        continue
                    subcmd = parts[1].lower()
                    if subcmd == "facts":
                        subj = parts[2] if len(parts) > 2 else None
                        print_facts(subj)
                    elif subcmd == "knowledge":
                        tag = parts[2] if len(parts) > 2 else None
                        print_knowledge(tag)
                    elif subcmd == "export":
                        if len(parts) < 3:
                            print("⚠️ Укажите формат: md или csv")
                        elif parts[2].lower() == "md":
                            export_md()
                        elif parts[2].lower() == "csv":
                            export_csv()
                        else:
                            print("⚠️ Неизвестный формат. Используйте md или csv.")
                    else:
                        print("⚠️ Неизвестная команда mem.")
                elif cmd == "add" and len(parts) > 1 and parts[1].lower() == "fact":
                    add_fact(parts[2:])
                elif cmd == "del" and len(parts) > 1 and parts[1].lower() == "fact":
                    del_fact(parts[2:])
                elif cmd == "test":
                    phrase = " ".join(parts[1:])
                    tokens = tokenizer.tokenize(phrase)
                    p_type = classifier.classify(phrase)
                    print(f"[Test] Тип: {p_type.name}")
                    print(f"[Test] Токены: {tokens}")
                else:
                    print("⚠️ Неизвестная команда. Введите /help для справки.")
            else:
                # Тестовое обучение напрямую (альтернатива run.py)
                tokens = tokenizer.tokenize(line)
                p_type = classifier.classify(line)
                print(f"[Debug] Тип: {p_type.name}")
                print(f"[Debug] Токены: {tokens}")
                facts = memory.find_facts_by_subject(tokens[0]) if tokens else []
                if facts:
                    print(f"[Debug] Факты по '{tokens[0]}':")
                    for f in facts:
                        print(f"  - {f.subject} — {f.predicate} — {f.obj}")
                else:
                    print("[Debug] Факты не найдены.")
        except KeyboardInterrupt:
            print("\n🛑 Прерывание.")
            break
        except Exception as e:
            print(f"⚠️ Ошибка: {e}")

if __name__ == "__main__":
    main()


===== FILE: C:\Users\Brazers\Desktop\1. стадия1\project_root\tools\generate_memory_doc.py =====

import os
from core.memory.memory import Memory

memory = Memory()

with open("memory_report.md", "w", encoding="utf-8") as f:
    f.write("# 🧠 Память ИИ\n\n")

    f.write("## 📌 Факты\n\n")
    for fact in memory.facts:
        f.write(f"- **{fact.subject}** — *{fact.predicate}* — {fact.obj}  \n")
        f.write(f"  Источник: `{fact.source}`  \n")
        f.write(f"  Время: `{fact.timestamp}`\n\n")

    f.write("## 📚 Знания\n\n")
    for entry in memory.knowledge:
        f.write(f"- **{entry.title}** ({entry.type})\n")
        f.write(f"  > {entry.content}  \n")
        f.write(f"  Теги: {', '.join(entry.tags)}  \n")
        f.write(f"  Время: `{entry.created}`\n\n")
