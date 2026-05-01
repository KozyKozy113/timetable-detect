"""
チケットサイトから出演者リストを取得するモジュール
対応サイト: TicketDive, LivePocket, tiget
"""

import os
from functools import lru_cache
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse

from dotenv import load_dotenv
load_dotenv()
from openai import OpenAI

openai_api_key = os.getenv('OPENAI_API_KEY')
if openai_api_key:
    OpenAI.api_key = openai_api_key
client = OpenAI()

# テキスト抽出用モデル
LLM_MODEL_NAME = "gpt-5"


@lru_cache(maxsize=128)
def get_performers_from_ticket_url(url: str) -> str | None:
    """
    チケットサイトURLから出演者リストを取得する

    Args:
        url: チケットサイトのURL

    Returns:
        出演者リストの箇条書き文字列（取得失敗時はNone）
    """
    if not url:
        return None

    try:
        # URLからドメインを判定
        parsed_url = urlparse(url)
        domain = parsed_url.netloc.lower()

        # サイトごとのテキスト取得
        page_text = _fetch_page_text(url)
        if not page_text:
            return None

        # LLMで出演者リストを抽出
        performers = _extract_performers_with_llm(page_text, domain)
        return performers

    except Exception as e:
        print(f"[ticket_scraper] Error fetching performers from {url}: {e}")
        return None


def _fetch_page_text(url: str) -> str | None:
    """
    URLからページのテキストコンテンツを取得する

    Args:
        url: ページのURL

    Returns:
        ページのテキストコンテンツ（取得失敗時はNone）
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()

        # エンコーディング対応
        response.encoding = response.apparent_encoding

        soup = BeautifulSoup(response.text, 'html.parser')

        # scriptタグとstyleタグを除去
        for tag in soup(['script', 'style', 'nav', 'footer', 'header']):
            tag.decompose()

        # テキストを取得
        text = soup.get_text(separator='\n', strip=True)

        # 空行を整理
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        text = '\n'.join(lines)

        # 長すぎる場合は切り詰め（トークン制限対策）
        max_chars = 15000
        if len(text) > max_chars:
            text = text[:max_chars]

        return text

    except Exception as e:
        print(f"[ticket_scraper] Error fetching page: {e}")
        return None


def _extract_performers_with_llm(page_text: str, domain: str) -> str | None:
    """
    LLMを使ってページテキストから出演者リストを抽出する

    Args:
        page_text: ページのテキストコンテンツ
        domain: サイトのドメイン

    Returns:
        出演者リストの箇条書き文字列（抽出失敗時はNone）
    """
    try:
        prompt_system = """あなたはイベント情報から出演者リストを抽出するアシスタントです。

与えられたテキストから、イベントに出演するアーティスト・アイドル・グループの名前を全て抽出してください。

出力形式:
- 箇条書きで1行に1組ずつ出力
- グループ名のみを抽出（「出演」「ゲスト」などの修飾語は除く）
- 重複は除去
- 出演者が見つからない場合は「出演者情報なし」と出力

注意:
- 会場名、主催者名、スタッフ名は出演者ではありません
- 「and more」「他」などの表記は出力しないでください
- 日程や時間情報は出力しないでください"""

        prompt_user = f"""以下はチケットサイト（{domain}）から取得したイベントページのテキストです。
このイベントの出演者リストを箇条書きで抽出してください。

---
{page_text}
---"""

        response = client.chat.completions.create(
            model=LLM_MODEL_NAME,
            messages=[
                {"role": "system", "content": prompt_system},
                {"role": "user", "content": prompt_user}
            ],
            max_tokens=2048,
            temperature=0
        )

        result = response.choices[0].message.content.strip()

        # 「出演者情報なし」の場合はNoneを返す
        if "出演者情報なし" in result:
            return None

        return result

    except Exception as e:
        print(f"[ticket_scraper] Error extracting performers with LLM: {e}")
        return None


def get_performers_list_from_ticket_urls(ticket_urls: list[str] | str | None) -> list[str]:
    """
    チケットURLから出演者名のリスト（重複なし）を返す。

    Args:
        ticket_urls: チケットサイトURL（文字列または文字列の配列、またはNone）

    Returns:
        出演者名のリスト（重複なし）。取得できない場合は空リスト。
    """
    if not ticket_urls:
        return []

    if isinstance(ticket_urls, str):
        ticket_urls = [ticket_urls]

    all_performers = []
    for url in ticket_urls:
        performers_text = get_performers_from_ticket_url(url)
        if not performers_text:
            continue
        for line in performers_text.split("\n"):
            # 箇条書き記号を除去
            name = line.strip().lstrip("-・●◆■▶▷※").strip()
            if name:
                all_performers.append(name)

    # 重複除去（順序保持）
    return list(dict.fromkeys(all_performers))


# テスト用
if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding='utf-8')

    test_urls = [
        "https://ticketdive.com/event/ssuggoi_0118",
        "https://ticketdive.com/event/apparechristmas2025",
        "https://livepocket.jp/e/aikou_0104e",
        "https://tiget.net/events/445684",
    ]

    for url in test_urls:
        print(f"\n{'='*60}")
        print(f"URL: {url}")
        print(f"{'='*60}")
        result = get_performers_from_ticket_url(url)
        if result:
            print(result)
        else:
            print("出演者情報を取得できませんでした")
