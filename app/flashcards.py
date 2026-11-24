"""Генератор карток для повторення з нотаток.
Інструкція запуску: README.md
"""
import json
import sys

from openai import APIConnectionError, APITimeoutError, OpenAI

MODEL = "qwen3:1.7b"
BASE_URL = "http://localhost:11434/v1"
MAX_TOKENS = 700
TEMPERATURE = 0.3

SYSTEM = (
    "Ти допомагаєш студентові повторювати матеріал. З наданих нотаток зроби рівно 5 карток "
    "«питання-відповідь» українською мовою. Відповідай ТІЛЬКИ JSON-масивом виду "
    '[{"question": "...", "answer": "..."}] без жодного іншого тексту. '
    "Використовуй лише інформацію з нотаток, нічого не вигадуй. "
    "Навіть якщо нотаток мало, поверни JSON-масив у квадратних дужках."
)

client = OpenAI(base_url=BASE_URL, api_key="local", timeout=300)


def parse_cards(raw):
    """Розбирає відповідь моделі: приймає і масив карток, і одиночний об'єкт."""
    try:
        data = json.loads(raw[raw.index("["): raw.rindex("]") + 1])
    except ValueError:
        data = json.loads(raw[raw.index("{"): raw.rindex("}") + 1])
    if isinstance(data, dict):
        data = [data]
    return [(c["question"], c["answer"]) for c in data]


def main(path):
    try:
        with open(path, encoding="utf-8-sig") as f:
            notes = f.read().strip()
    except FileNotFoundError:
        sys.exit(f"Помилка: файл {path} не знайдено.")
    if not notes:
        sys.exit("Помилка: файл з нотатками порожній.")

    print("Генерую картки (на слабкому комп'ютері це може зайняти до пари хвилин)…", flush=True)
    text = ""
    try:
        stream = client.chat.completions.create(
            model=MODEL, stream=True, temperature=TEMPERATURE, max_tokens=MAX_TOKENS,
            messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": notes}],
            extra_body={"reasoning_effort": "none"},
        )
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                piece = chunk.choices[0].delta.content
                text += piece
                print(piece, end="", flush=True)
        print("\n")
    except APIConnectionError:
        sys.exit("Помилка: сервер не запущено. Запусти Ollama і повтори.")
    except APITimeoutError:
        sys.exit("Помилка: таймаут. Спробуй коротші нотатки або меншу модель.")
    except Exception as error:
        sys.exit(f"Помилка сервера: {error}")

    try:
        cards = parse_cards(text)
    except (ValueError, KeyError, TypeError):
        sys.exit("Помилка: модель повернула некоректну відповідь (не JSON). Спробуй ще раз.")

    print("=" * 40)
    for i, (q, a) in enumerate(cards, 1):
        print(f"Картка {i}\n  Питання:   {q}\n  Відповідь: {a}\n")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "notes.txt")