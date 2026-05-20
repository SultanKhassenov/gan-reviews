"""
Scraper для отзывов Kaspi.kz.

Адаптирован из github.com/SultanKhassenov/kaspi-parser/ai-service/scraper.py.
Изменения:
- убраны mock-данные и Docker-специфика
- возвращает сырые отзывы с URL для дальнейшей обработки в collect.py
- логирование без префикса "[Scraper]" (его добавит оркестратор)
"""
import asyncio
import re
import hashlib
from datetime import datetime
from pathlib import Path
from playwright.async_api import async_playwright


def review_id(url: str, author: str, text: str) -> str:
    """Стабильный ID отзыва для дедупликации."""
    raw = f"{url}|{author}|{text[:80]}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


async def scrape_product_reviews_async(
    url: str,
    max_reviews: int = 300,
    debug_screenshot_dir: Path | None = None,
) -> tuple[str, list[dict]]:
    """
    Парсит отзывы одного товара Kaspi.

    Returns: (title, reviews) где reviews — список dict с полями
             id, author, rating, text, date, url
    """
    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True,
            args=["--no-sandbox", "--disable-setuid-sandbox", "--disable-dev-shm-usage"],
        )
        context = await browser.new_context(
            viewport={"width": 1280, "height": 800},
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/122.0.0.0 Safari/537.36"
            ),
        )
        page = await context.new_page()

        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=25000)
            await page.wait_for_selector("h1", timeout=10000)

            title_element = await page.query_selector("h1")
            title = (await title_element.inner_text()).strip() if title_element else "Неизвестный товар"

            # JS hydration
            await page.wait_for_timeout(4000)

            # City modal (Almaty)
            city_element = None
            potential_cities = (
                await page.query_selector_all("text=Алматы")
                or await page.query_selector_all("a:has-text('Алматы')")
                or await page.query_selector_all("span:has-text('Алматы')")
            )
            if potential_cities:
                city_element = potential_cities[0]
            else:
                city_element = (
                    await page.query_selector(".dialog__close")
                    or await page.query_selector(".dialog-close")
                    or await page.query_selector(".close")
                )
            if city_element:
                try:
                    await page.evaluate("el => el.click()", city_element)
                    await page.wait_for_timeout(2000)
                except Exception:
                    pass

            # Reviews tab
            reviews_tab = (
                await page.query_selector("li[data-tab='reviews']")
                or await page.query_selector("text=Отзывы")
                or await page.query_selector(".tabs-content__tab:has-text('Отзывы')")
            )
            if reviews_tab:
                try:
                    await page.evaluate("el => el.click()", reviews_tab)
                    await page.wait_for_timeout(2000)
                except Exception:
                    try:
                        await reviews_tab.click(force=True)
                    except Exception:
                        pass

            # Gradual scroll for lazy-load
            for i in range(1, 6):
                await page.evaluate(
                    f"window.scrollTo(0, (document.body.scrollHeight / 5) * {i})"
                )
                await page.wait_for_timeout(800)

            # Wait for review cards
            try:
                await page.wait_for_selector(".reviews__review", timeout=5000)
            except Exception:
                await browser.close()
                return title, []

            review_elements = await page.query_selector_all(".reviews__review")

            # Click "Show more" until enough loaded
            click_count = 0
            while len(review_elements) < max_reviews and click_count < 120:
                more_btn = (
                    await page.query_selector(".reviews__more")
                    or await page.query_selector("text='Показать ещё'")
                )
                if not more_btn:
                    break
                try:
                    await more_btn.scroll_into_view_if_needed()
                    await page.wait_for_timeout(500)
                    await page.evaluate("el => el.click()", more_btn)
                    await page.wait_for_timeout(1500)
                    click_count += 1
                    review_elements = await page.query_selector_all(".reviews__review")
                except Exception:
                    break

            # Extract review data
            reviews = []
            for idx, el in enumerate(review_elements[:max_reviews]):
                try:
                    author_el = await el.query_selector(".reviews__author")
                    author = (await author_el.inner_text()).strip() if author_el else "Аноним"

                    date_el = await el.query_selector(".reviews__date")
                    date = (
                        (await date_el.inner_text()).strip()
                        if date_el
                        else datetime.now().strftime("%d.%m.%Y")
                    )

                    text_el = (
                        await el.query_selector(".reviews__review-text p")
                        or await el.query_selector(".reviews__review-text")
                    )
                    text = await text_el.inner_text() if text_el else ""
                    text = re.sub(
                        r"^(Комментарий|Достоинства|Недостатки|Описание):\s*",
                        "",
                        text,
                        flags=re.IGNORECASE,
                    ).strip()
                    text = re.sub(
                        r"\s*\d+\s*человек\(а\)\s*посчитал\(и\)\s*отзыв\s*полезным.*",
                        "",
                        text,
                        flags=re.IGNORECASE,
                    ).strip()

                    rating = 5
                    rating_el = await el.query_selector(".rating")
                    if rating_el:
                        rating_classes = await rating_el.evaluate("el => el.className")
                        match = re.search(r"_([1-5])0", rating_classes)
                        if match:
                            rating = int(match.group(1))

                    reviews.append(
                        {
                            "id": review_id(url, author, text),
                            "author": author,
                            "rating": rating,
                            "text": text,
                            "date": date,
                            "url": url,
                        }
                    )
                except Exception:
                    continue

            await browser.close()
            return title, reviews

        except Exception as e:
            if debug_screenshot_dir:
                try:
                    debug_screenshot_dir.mkdir(parents=True, exist_ok=True)
                    safe_name = re.sub(r"[^\w-]", "_", url)[-60:]
                    await page.screenshot(
                        path=str(debug_screenshot_dir / f"err_{safe_name}.png")
                    )
                except Exception:
                    pass
            await browser.close()
            raise e


def scrape_product_reviews(url: str, max_reviews: int = 300) -> tuple[str, list[dict]]:
    """Синхронная обёртка для удобства вызова из collect.py."""
    return asyncio.run(scrape_product_reviews_async(url, max_reviews))
