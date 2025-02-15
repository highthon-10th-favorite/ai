import asyncio
import aiohttp
import time

# 동시 요청 제한 설정 (예: 최대 20개 동시 요청)
CONCURRENT_REQUESTS = 20

# aiohttp 세마포어 (동시 요청 수 제한)
wiki_semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)
wikidata_semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)

# ----- 비동기 Wikipedia 카테고리 크롤링 -----
async def async_get_category_members(category: str, cmtype: str = "page", session: aiohttp.ClientSession = None) -> list:
    """
    주어진 카테고리에서 cmtype("page": 문서, "subcat": 하위 카테고리)에 해당하는 페이지들을 비동기적으로 가져옵니다.
    """
    url = "https://ko.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "list": "categorymembers",
        "cmtitle": category,
        "cmlimit": "max",
        "cmtype": cmtype,
        "format": "json"
    }
    collected = []
    while True:
        async with wiki_semaphore:
            async with session.get(url, params=params, timeout=10) as response:
                data = await response.json()
        if "query" in data and "categorymembers" in data["query"]:
            collected.extend(data["query"]["categorymembers"])
        else:
            break

        if "continue" in data:
            params["cmcontinue"] = data["continue"]["cmcontinue"]
            # API 부담 최소화를 위해 짧은 딜레이 적용 (필요시 조정)
            await asyncio.sleep(0.05)
        else:
            break
    return collected

async def async_get_all_celebrities(category: str, session: aiohttp.ClientSession) -> list:
    """
    주어진 카테고리와 하위 카테고리의 모든 페이지를 재귀적으로 비동기 방식으로 수집합니다.
    """
    pages = await async_get_category_members(category, cmtype="page", session=session)
    subcats = await async_get_category_members(category, cmtype="subcat", session=session)
    
    tasks = []
    for subcat in subcats:
        subcat_title = subcat["title"]
        print(f"{subcat_title} 카테고리 탐색 중...")
        tasks.append(async_get_all_celebrities(subcat_title, session))
    
    if tasks:
        results = await asyncio.gather(*tasks)
        for subpages in results:
            pages.extend(subpages)
    return pages

# ----- 페이지의 pageprops 정보를 배치로 가져오기 (재시도 로직 포함) -----
async def fetch_pageprops(pageids: list, session: aiohttp.ClientSession, retries=3) -> dict:
    """
    주어진 pageids에 대해 pageprops 정보를 가져옵니다.
    타임아웃 또는 기타 예외 발생 시 재시도합니다.
    """
    url = "https://ko.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "pageids": "|".join(pageids),
        "prop": "pageprops",
        "format": "json"
    }
    for attempt in range(retries):
        try:
            async with wiki_semaphore:
                async with session.get(url, params=params, timeout=20) as response:
                    return await response.json()
        except asyncio.TimeoutError:
            print(f"TimeoutError: 재시도 {attempt + 1}/{retries} 중...")
            await asyncio.sleep(0.5)
        except Exception as e:
            print(f"fetch_pageprops 에서 예외 발생: {e}")
            break
    return {}  # 실패 시 빈 딕셔너리 반환

async def add_pageprops_to_pages(pages: list, session: aiohttp.ClientSession, batch_size: int = 50) -> list:
    """
    크롤링한 각 페이지에 대해 pageprops 정보를 추가합니다.
    """
    pages_dict = {str(page["pageid"]): page for page in pages}
    pageids = list(pages_dict.keys())
    
    for i in range(0, len(pageids), batch_size):
        batch_ids = pageids[i:i+batch_size]
        data = await fetch_pageprops(batch_ids, session)
        pages_data = data.get("query", {}).get("pages", {})
        for pid, pdata in pages_data.items():
            if "pageprops" in pdata:
                pages_dict[pid]["pageprops"] = pdata["pageprops"]
        await asyncio.sleep(0.05)
    return list(pages_dict.values())

# ----- 비동기 Wikidata 필터링 -----
async def fetch_wikidata_entities(wikibase_ids: list, session: aiohttp.ClientSession) -> dict:
    """
    wikibase_ids 리스트(예: ['Q5', 'Q123', ...])에 대해 Wikidata 엔터티 정보를 비동기적으로 가져옵니다.
    """
    url = "https://www.wikidata.org/w/api.php"
    params = {
        "action": "wbgetentities",
        "ids": "|".join(wikibase_ids),
        "props": "claims",
        "format": "json"
    }
    async with wikidata_semaphore:
        async with session.get(url, params=params, timeout=10) as response:
            return await response.json()

async def filter_batch_humans(batch: list, session: aiohttp.ClientSession, cache: dict) -> list:
    """
    배치 내 페이지들에 대해 Wikidata 정보를 확인하여 인간(Q5) 여부를 판별합니다.
    이미 확인한 wikibase_id는 캐싱합니다.
    """
    human_pages = []
    ids_to_query = []
    id_to_pages = {}
    for page in batch:
        if "pageprops" not in page or "wikibase_item" not in page["pageprops"]:
            continue
        wikibase_id = page["pageprops"]["wikibase_item"]
        if wikibase_id in cache:
            if cache[wikibase_id]:
                human_pages.append(page)
        else:
            ids_to_query.append(wikibase_id)
            id_to_pages.setdefault(wikibase_id, []).append(page)

    if ids_to_query:
        data = await fetch_wikidata_entities(ids_to_query, session)
        entities = data.get("entities", {})
        for wid in ids_to_query:
            is_human = False
            entity = entities.get(wid, {})
            claims = entity.get("claims", {})
            for claim in claims.get("P31", []):
                mainsnak = claim.get("mainsnak", {})
                datavalue = mainsnak.get("datavalue", {})
                if datavalue.get("value", {}).get("id") == "Q5":
                    is_human = True
                    break
            cache[wid] = is_human
            if is_human:
                human_pages.extend(id_to_pages.get(wid, []))
    return human_pages

async def filter_human_pages_async(pages: list) -> list:
    """
    전체 페이지 리스트에 대해 배치로 비동기 방식으로 인간(Q5) 여부를 필터링합니다.
    """
    human_pages = []
    BATCH_SIZE = 50
    tasks = []
    cache = {}
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=CONCURRENT_REQUESTS)) as session:
        # 먼저 각 페이지에 대해 pageprops 정보를 추가합니다.
        pages = await add_pageprops_to_pages(pages, session, batch_size=BATCH_SIZE)
        for i in range(0, len(pages), BATCH_SIZE):
            batch = pages[i:i+BATCH_SIZE]
            tasks.append(filter_batch_humans(batch, session, cache))
        results = await asyncio.gather(*tasks)
        for res in results:
            human_pages.extend(res)
    return human_pages

# ----- 메인 실행 함수 -----
async def main():
    category_name = "Category:대한민국의_연예인"
    
    # Wikipedia 카테고리 크롤링 (동시 요청 제한 적용)
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=CONCURRENT_REQUESTS)) as session:
        print("카테고리 내 페이지 및 하위 카테고리 탐색 중...")
        all_pages = await async_get_all_celebrities(category_name, session)
        print(f"전체 {len(all_pages)}개의 페이지를 찾았습니다.")
    
    print("인간(연예인) 필터링 중 (비동기 배치 처리)...")
    start_time = time.time()
    human_pages = await filter_human_pages_async(all_pages)
    elapsed = time.time() - start_time
    print(f"인간으로 판별된 페이지: {len(human_pages)}개, 필터링 시간: {elapsed:.2f}초")

    # 중복 제거 후 페이지 제목 정렬
    celebrity_names = sorted({page["title"] for page in human_pages})
    print(f"총 {len(celebrity_names)}명의 대한민국 연예인 이름을 찾았습니다.")

    output_filename = "korean_celebrities.txt"
    with open(output_filename, "w", encoding="utf-8") as f:
        for name in celebrity_names:
            f.write(name + "\n")
    print(f"연예인 이름 리스트가 '{output_filename}' 파일에 저장되었습니다.")

if __name__ == "__main__":
    asyncio.run(main())
