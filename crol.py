import os
import cv2
import asyncio
import aiohttp
import aiofiles
import numpy as np
from bs4 import BeautifulSoup
from insightface.app import FaceAnalysis

# ==============================
# 설정
# ==============================
INPUT_FILE = "celebrity_names.txt"   # 연예인 목록 파일
OUTPUT_DIR = "images"                   # 저장 폴더
MAX_IMAGES = 4                          # 인포박스에서 최대 몇 장을 저장할지 (0~3 번 index)
MAX_RETRIES = 3                         # 요청 최대 재시도 횟수
CONCURRENT_REQUESTS = 10               # 동시 요청 제한

# 폴더 생성
os.makedirs(OUTPUT_DIR, exist_ok=True)
for sub in ["male", "female", "unknown"]:
    os.makedirs(os.path.join(OUTPUT_DIR, sub), exist_ok=True)

# ==============================
# 얼굴 검출 모델 (Detection만)
# ==============================
face_app = FaceAnalysis(allowed_modules=["detection"])
face_app.prepare(ctx_id=0, det_size=(640, 640))
print("FaceAnalysis: 얼굴 검출 모델 로드 완료")

# 동시 요청 제한 세마포어
semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)

# ==============================
# 비동기 HTTP 요청 함수
# ==============================
async def fetch_text(url, session):
    """URL로부터 HTML 텍스트를 가져옴 (재시도 포함)"""
    for attempt in range(MAX_RETRIES):
        try:
            async with semaphore, session.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10) as resp:
                if resp.status == 200:
                    return await resp.text()
        except asyncio.TimeoutError:
            print(f"[Timeout] {url}, 재시도 {attempt+1}/{MAX_RETRIES}")
            await asyncio.sleep(2)
    return None

async def fetch_bytes(url, session):
    """URL로부터 바이너리(이미지) 데이터를 가져옴 (재시도 포함)"""
    headers = {"User-Agent": "Mozilla/5.0", "Referer": "https://ko.wikipedia.org/"}
    for attempt in range(MAX_RETRIES):
        try:
            async with semaphore, session.get(url, headers=headers, timeout=10) as resp:
                if resp.status == 200:
                    return await resp.read()
        except asyncio.TimeoutError:
            print(f"[Timeout] {url}, 재시도 {attempt+1}/{MAX_RETRIES}")
            await asyncio.sleep(2)
    return None

async def save_file(filename, data):
    """비동기 파일 저장"""
    async with aiofiles.open(filename, "wb") as f:
        await f.write(data)

# ==============================
# 성별 판별 (문서 하단 카테고리)
# ==============================
def extract_gender_from_categories(soup) -> str:
    """
    문서 하단의 카테고리 영역(#mw-normal-catlinks)에서
    '남자', '여자' 키워드가 있는지 확인 → 남/여/unknown
    """
    catlinks = soup.find("div", id="mw-normal-catlinks")
    if not catlinks:
        return "unknown"

    lis = catlinks.select("ul li")
    if not lis:
        return "unknown"

    for li in lis:
        a = li.find("a")
        if not a:
            continue
        text = a.text.strip()
        if "남자" in text or "남성" in text:
            return "male"
        if "여자" in text or "여성" in text:
            return "female"
    return "unknown"

# ==============================
# 위키백과 인포박스 파싱
# ==============================
async def get_profile_image_urls_and_gender(name, session):
    """
    (이름) → (위키백과 문서) → (인포박스 이미지 URL 리스트, 성별)
    """
    wiki_url = f"https://ko.wikipedia.org/wiki/{name}"
    html_text = await fetch_text(wiki_url, session)
    if not html_text:
        return [], "unknown"

    soup = BeautifulSoup(html_text, "html.parser")
    gender = extract_gender_from_categories(soup)

    infobox = soup.find("table", class_="infobox")
    if not infobox:
        return [], gender

    images = infobox.find_all("img")
    img_urls = []
    for img in images:
        src = img.get("src")
        if not src:
            continue
        if "Picto_infobox_music" in src:
            continue
        if src.startswith("//"):
            src = "https:" + src
        elif src.startswith("/"):
            src = "https://ko.wikipedia.org" + src
        img_urls.append(src)

    return img_urls, gender

# ==============================
# 얼굴 검출
# ==============================
async def is_face_image_async(image_bytes):
    """비동기 얼굴 검출"""
    return await asyncio.to_thread(is_face_image_sync, image_bytes)

def is_face_image_sync(image_bytes):
    """동기 얼굴 검출 (임베딩 X)"""
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None:
        return False
    faces = face_app.get(img)
    return bool(faces)

async def download_and_save_image(url, filepath, session):
    """이미지 다운로드 후 얼굴 검출 → 있으면 저장"""
    data = await fetch_bytes(url, session)
    if not data:
        print(f"[Fail] 다운로드 실패: {url}")
        return False

    face_found = await is_face_image_async(data)
    if not face_found:
        print(f"[NoFace] 얼굴 미검출: {url}")
        return False

    await save_file(filepath, data)
    print(f"[SAVE] {filepath}")
    return True

# ==============================
# 연예인 처리
# ==============================
async def process_celebrity(name, session):
    """
    1) 위키백과 페이지 파싱 → 이미지 URL들 + 성별
    2) 이미지 최대 MAX_IMAGES 장 다운로드
       (각각 <이름>_profile_0.jpg, _profile_1.jpg 등)
    3) 남/여/unknown 폴더에 저장
    """
    img_urls, gender = await get_profile_image_urls_and_gender(name, session)
    if not img_urls:
        print(f"[{name}] 인포박스 이미지가 없습니다.")
        return

    # 폴더 결정
    base_dir = os.path.join(OUTPUT_DIR, gender)
    if not os.path.exists(base_dir):
        base_dir = os.path.join(OUTPUT_DIR, "unknown")

    # 안전한 이름
    safe_name = "".join(c for c in name if c.isalnum() or c in " _-").rstrip()

    print(f"[{name}] → 성별: {gender}, 인포박스 이미지 {len(img_urls)}개")

    # 최대 MAX_IMAGES 장만 저장
    count = 0
    for idx, url in enumerate(img_urls):
        if count >= MAX_IMAGES:
            break
        filename = os.path.join(base_dir, f"{safe_name}_profile_{idx}.jpg")

        # 중복 파일 있으면 스킵
        if os.path.exists(filename):
            print(f"[{name}] '{filename}' 이미 존재, 스킵.")
            continue

        success = await download_and_save_image(url, filename, session)
        if success:
            count += 1

    print(f"[{name}] 총 {count}개 저장 완료.")

# ==============================
# 메인
# ==============================
async def main():
    if not os.path.exists(INPUT_FILE):
        print(f"입력 파일 '{INPUT_FILE}'을 찾을 수 없습니다.")
        return

    # 파일에서 연예인 목록 읽기
    async with aiofiles.open(INPUT_FILE, "r", encoding="utf-8") as f:
        content = await f.read()
    names = [line.strip() for line in content.splitlines() if line.strip()]

    connector = aiohttp.TCPConnector(limit=CONCURRENT_REQUESTS)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [process_celebrity(name, session) for name in names]
        await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(main())
