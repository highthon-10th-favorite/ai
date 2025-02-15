import requests
import os
from bs4 import BeautifulSoup

def get_wikipedia_page_title(name):
    """
    입력한 이름으로 위키피디아 페이지를 직접 가져옵니다.
    페이지가 정상적으로 로드되면 제목(name)을 반환하고,
    disambiguation(동음이의) 페이지일 경우 페이지 HTML을 함께 반환합니다.
    """
    headers = {"User-Agent": "Mozilla/5.0"}
    url = f"https://ko.wikipedia.org/wiki/{name}"
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, "html.parser")

        if soup.find("table", class_="ambox") or "동일명" in soup.text or "동음이의" in soup.text:
            return None, response.text
        return name, None
    else:
        return None, None

def search_wikipedia_api(query):
    """
    위키피디아 검색 API를 사용하여 query와 관련된 페이지 제목을 반환합니다.
    """
    search_url = "https://ko.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "format": "json"
    }
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(search_url, params=params, headers=headers)
    if response.status_code == 200:
        data = response.json()
        results = data.get("query", {}).get("search", [])
        if results:
            return results[0].get("title")
    return None

def extract_candidate_from_disambiguation(content, keywords):
    """
    동음이의 페이지 HTML(content)에서 목록(li 항목) 중 키워드가 포함된 후보 제목을 추출합니다.
    """
    soup = BeautifulSoup(content, "html.parser")
    candidates = []
    for li in soup.select("div.mw-parser-output ul li"):
        a = li.find("a")
        if a and a.get("title"):
            title = a.get("title")
            for kw in keywords:
                if kw in title:
                    candidates.append(title)
                    break
    if candidates:
        return candidates[0]
    return None

def get_final_page_title(name):
    """
    입력한 이름에 대해 유효한 위키피디아 페이지 제목을 얻습니다.
      ① 직접 접근 시 정상 페이지면 그대로 사용,
      ② 동음이의 페이지인 경우 후보 키워드(예: "(배우)", "(가수)", "(연예인)", "(모델)")로 적합한 항목을 선택,
      ③ 없으면 검색 API를 사용합니다.
    """
    candidate_keywords = ["(배우)", "(가수)", "(연예인)", "(모델)"]
    title, disamb_content = get_wikipedia_page_title(name)
    if title:
        return title
    else:
        if disamb_content:
            candidate = extract_candidate_from_disambiguation(disamb_content, candidate_keywords)
            if candidate:
                return candidate
        api_title = search_wikipedia_api(name)
        return api_title

def get_profile_image_url_from_title(title):
    """
    주어진 위키피디아 페이지 제목(title)에서 인포박스 내 유효한 프로필 이미지 URL을 추출합니다.
    """
    headers = {"User-Agent": "Mozilla/5.0"}
    url = f"https://ko.wikipedia.org/wiki/{title}"
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        print("대체 페이지를 불러오는데 실패했습니다.")
        return None
    soup = BeautifulSoup(response.text, "html.parser")
    infobox = soup.find("table", class_="infobox")
    if not infobox:
        print("검색 결과 페이지에서도 인포박스를 찾을 수 없습니다.")
        return None
    images = infobox.find_all("img")
    selected_img_src = None
    for img in images:
        src = img.get("src")
        if not src:
            continue
        if "Picto_infobox_music" in src:
            continue
        selected_img_src = src
        break
    if not selected_img_src:
        print("유효한 프로필 이미지를 찾을 수 없습니다.")
        return None
    if selected_img_src.startswith("//"):
        selected_img_src = "https:" + selected_img_src
    elif selected_img_src.startswith("/"):
        selected_img_src = "https://ko.wikipedia.org" + selected_img_src
    return selected_img_src

def get_profile_image_url(celebrity_name):
    """
    전체 흐름:
      입력한 이름으로 최종 페이지 제목을 결정한 후, 해당 페이지에서 프로필 이미지 URL을 가져옵니다.
    """
    final_title = get_final_page_title(celebrity_name)
    if not final_title:
        print("적합한 페이지를 찾지 못했습니다.")
        return None
    print(f"대신 '{final_title}' 페이지를 사용합니다.")
    return get_profile_image_url_from_title(final_title)

def save_image(image_url, filename):
    """
    이미지 URL로부터 이미지를 다운로드하여 filename으로 저장합니다.
    다운로드 요청 시 User-Agent와 Referer 헤더를 추가합니다.
    """
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Referer": "https://ko.wikipedia.org/"
    }
    response = requests.get(image_url, headers=headers)
    if response.status_code == 200:
        if not response.content:
            print("다운로드된 이미지 데이터가 비어 있습니다.")
            return False
        try:
            with open(filename, "wb") as f:
                f.write(response.content)
            if os.path.exists(filename):
                abs_path = os.path.abspath(filename)
                print(f"이미지가 '{abs_path}'로 저장되었습니다.")
                return True
            else:
                print("파일 저장에 실패했습니다: 파일이 존재하지 않습니다.")
                return False
        except Exception as e:
            print("파일 저장 중 오류 발생:", e)
            return False
    else:
        print(f"이미지를 다운로드하는데 실패했습니다. 상태 코드: {response.status_code}")
        return False

if __name__ == "__main__":
    celebrity = input("연예인 이름을 입력하세요: ").strip()
    print("현재 작업 디렉토리:", os.getcwd())
    
    alias_mapping = {
        "해린": "강해린"
    }
    
    img_url = get_profile_image_url(celebrity)
    if not img_url and celebrity in alias_mapping:
        print(f"'{celebrity}'의 프로필 이미지를 찾지 못했습니다. 대신 '{alias_mapping[celebrity]}' 페이지를 시도합니다.")
        img_url = get_profile_image_url(alias_mapping[celebrity])
    
    if img_url:
        print("프로필 이미지 URL:", img_url)
        if not save_image(img_url, f"{celebrity}_profile.jpg"):
            print("이미지를 저장하지 못했습니다.")
    else:
        print("프로필 이미지를 불러오는데 실패했습니다.")
