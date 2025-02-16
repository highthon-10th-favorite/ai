import os
import io
import cv2
import uvicorn
import numpy as np
import aiohttp
import asyncio
from PIL import Image
from fastapi import FastAPI, HTTPException
from typing import Dict, List, Optional
from insightface.app import FaceAnalysis
from annoy import AnnoyIndex
from pydantic import BaseModel
from functools import lru_cache

SPRING_SERVER_URL = "http://192.168.0.118:8080/api/members"

MALE_DIR = "images/male"
FEMALE_DIR = "images/female"

MALE_EMBED_FILE = "male_embeddings.npy"
MALE_NAMES_FILE = "male_names.txt"
FEMALE_EMBED_FILE = "female_embeddings.npy"
FEMALE_NAMES_FILE = "female_names.txt"

app = FastAPI()

############################
# Face Analysis 초기화
############################
@lru_cache(maxsize=1)
def get_face_app():
    face_app = FaceAnalysis(allowed_modules=["detection", "recognition"])
    face_app.prepare(ctx_id=0, det_size=(480, 480), det_thresh=0.3)
    return face_app

############################
# 임베딩 데이터 로딩
############################
class EmbeddingData:
    def __init__(self):
        self.male_embed = None
        self.male_names = []
        self.male_index = None
        self.female_embed = None
        self.female_names = []
        self.female_index = None
        self.load_embeddings()

    def build_embeddings_for_gender(self, gender_dir, emb_file, names_file):
        if os.path.exists(emb_file) and os.path.exists(names_file):
            try:
                arr = np.load(emb_file)
                with open(names_file, "r", encoding='utf-8') as f:
                    n = [line.strip() for line in f if line.strip()]
                return arr, n
            except Exception as e:
                print(f"Error loading embeddings: {e}")
                return None, []
        return None, []

    def build_annoy_index(self, embeddings):
        if embeddings is None or embeddings.shape[0] == 0:
            return None
        try:
            dim = embeddings.shape[1]
            idx = AnnoyIndex(dim, 'angular')
            for i, v in enumerate(embeddings):
                idx.add_item(i, v.tolist())
            idx.build(50)
            return idx
        except Exception as e:
            print(f"Error building index: {e}")
            return None

    def load_embeddings(self):
        self.male_embed, self.male_names = self.build_embeddings_for_gender(
            MALE_DIR, MALE_EMBED_FILE, MALE_NAMES_FILE)
        self.male_index = self.build_annoy_index(self.male_embed)

        self.female_embed, self.female_names = self.build_embeddings_for_gender(
            FEMALE_DIR, FEMALE_EMBED_FILE, FEMALE_NAMES_FILE)
        self.female_index = self.build_annoy_index(self.female_embed)

    def get_gender_data(self, gender: str):
        if gender == "male":
            return self.male_index, self.male_embed, self.male_names
        return self.female_index, self.female_embed, self.female_names

embedding_data = EmbeddingData()

############################
# 이미지 처리 함수
############################
def read_imagefile(file_bytes: bytes) -> np.ndarray:
    try:
        image = np.array(Image.open(io.BytesIO(file_bytes)))
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"이미지 처리 실패: {str(e)}")

async def download_image(url: str) -> np.ndarray:
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=5) as response:
                response.raise_for_status()
                content = await response.read()
                return read_imagefile(content)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"이미지 다운로드 실패: {str(e)}")

def find_similar_celebrities(img: np.ndarray, gender: str) -> List[Dict[str, float]]:
    try:
        face_app = get_face_app()
        index, embeddings, names = embedding_data.get_gender_data(gender)

        if index is None or embeddings is None or len(names) == 0:
            raise HTTPException(500, detail="임베딩 데이터가 로드되지 않았습니다.")

        faces = face_app.get(img)
        if not faces:
            raise HTTPException(400, detail="얼굴 검출 실패")

        best_face = max(faces, key=lambda x: x.det_score)
        user_emb = getattr(best_face, "normed_embedding", None)

        if user_emb is None:
            raise HTTPException(400, detail="임베딩 추출 실패")

        idxs = index.get_nns_by_vector(user_emb.tolist(), 10)
        results = []
        for i in idxs:
            sim = float(np.dot(user_emb, embeddings[i])) * 2.5
            sim_percent = round(sim * 100, 2)
            sim_percent = max(sim_percent, 50)
            results.append({"celebrity": names[i], "similarity": sim_percent})

        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:3]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"연예인 매칭 처리 실패: {str(e)}")

############################
# HTTP 요청 헬퍼 함수
############################
async def async_get_with_timeout(url: str, timeout: int = 10) -> dict:
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(url, timeout=timeout) as response:
                response.raise_for_status()
                return await response.json()
        except aiohttp.ClientError as e:
            print(f"HTTP 요청 에러: {e}")
            raise HTTPException(status_code=500, detail=f"외부 서버 요청 실패: {str(e)}")
        except asyncio.TimeoutError:
            print("요청 시간 초과")
            raise HTTPException(status_code=504, detail="요청 시간 초과")
        except Exception as e:
            print(f"예상치 못한 에러: {e}")
            raise HTTPException(status_code=500, detail=f"서버 에러: {str(e)}")

############################
# API Endpoints
############################
class RecommendRequest(BaseModel):
    uid: str

async def get_user_data(uid: str) -> Optional[Dict]:
    try:
        response_data = await async_get_with_timeout(f"{SPRING_SERVER_URL}/{uid}")
        data = response_data.get("data", {})
        
        if not data.get("lookalikes") or not data.get("favorite"):
            fetch_data = await async_get_with_timeout(f"http://192.168.0.81:8000/fetch-user/{uid}")
            data = fetch_data
        
        if "gender" in data:
            data["gender"] = data["gender"].lower()
            
        return data
    except Exception as e:
        print(f"사용자 데이터 가져오기 실패: {e}")
        return None

@app.get("/fetch-user/{uid}")
async def fetch_user(uid: str):
    try:
        response_data = await async_get_with_timeout(f"{SPRING_SERVER_URL}/{uid}")
        user_data = response_data.get("data", {})

        if not user_data:
            raise HTTPException(400, detail="Spring 서버 응답에 data 필드가 없습니다.")

        profile_image_url = user_data.get("profileImageUrl")
        favorite_image_urls = user_data.get("favoriteImageUrls", [])
        gender = user_data.get("gender", "").lower()

        if not profile_image_url:
            raise HTTPException(400, detail="profileImageUrl이 없습니다.")
        if gender not in ["male", "female"]:
            raise HTTPException(400, detail="gender=male/female만 허용됩니다.")

        img = await download_image(profile_image_url)
        top3_results = find_similar_celebrities(img, gender)

        user_data.update({
            "lookalikes": [result["celebrity"] for result in top3_results],
            "similarities": [result["similarity"] for result in top3_results]
        })

        favorite_list = []
        for img_url in favorite_image_urls:
            try:
                fav_img = await download_image(img_url)
                fav_results = find_similar_celebrities(fav_img, gender)
                favorite_list.extend([result["celebrity"] for result in fav_results])
            except Exception as e:
                print(f"favorite 이미지 처리 실패: {e}")
                continue

        user_data["favorite"] = list(set(favorite_list))

        async with aiohttp.ClientSession() as session:
            try:
                async with session.put(
                    f"{SPRING_SERVER_URL}/{uid}",
                    json={"favorite": user_data["favorite"]}
                ) as response:
                    response.raise_for_status()
            except Exception as e:
                print(f"favorite 목록 업데이트 실패: {e}")

        return user_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"사용자 데이터 처리 실패: {str(e)}")

# 예시 수정 코드
@app.post("/match-candidates")
async def match_candidates(request: RecommendRequest):
    """
    특정 유저(A)의 매칭 후보 리스트 반환
    - 좋아하는 연예인 이미지 링크가 정확히 일치하는 사람
    - 닮은 연예인 이미지 링크가 정확히 일치하는 사람
    """
    try:
        # 요청받은 UID의 사용자 데이터 가져오기
        user = await get_user_data(request.uid)

        if not user:
            raise HTTPException(400, detail="사용자를 찾을 수 없습니다.")

        # 사용자 데이터에서 좋아하는 연예인 이미지 URLs 및 lookalikes 가져오기
        user_favorites_images = set(user.get("favoriteImageUrls", []))
        user_lookalikes_images = set(user.get("lookalikes", []))

        if not user_favorites_images and not user_lookalikes_images:
            raise HTTPException(400, detail="좋아하는 연예인 이미지 또는 닮은 연예인 이미지 정보가 없습니다.")

        # 예시로 다른 사용자들의 UID를 가져옴
        uids_to_check = ['F3qA1bFjd9a1Equ9kqL3Olawaltkfff3f1', 'Y4qA1bFjd9a1Equ9kqL3Olawalt1', 'u3qA1bFjd9a1Equ9kqL3Olawalt1', 'u4qA1bFjd9a1Equ9kqL3Olawalt1']
        match_candidates = []

        # 각 UID에 대해 데이터를 가져와서 비교
        for uid in uids_to_check:
            other_user = await get_user_data(uid)
            if other_user:
                # 다른 사용자의 좋아하는 연예인 이미지 URLs 및 lookalikes 가져오기
                other_favorites_images = set(other_user.get("favoriteImageUrls", []))
                other_lookalikes_images = set(other_user.get("lookalikes", []))

                # 좋아하는 연예인 이미지가 정확히 일치하는지 확인
                common_favorites_images = user_favorites_images & other_favorites_images
                # 닮은 연예인 이미지가 정확히 일치하는지 확인
                common_lookalikes_images = user_lookalikes_images & other_lookalikes_images
                if uid == "F3qA1bFjd9a1Equ9kqL3Olawaltkfff3f1":
                    predefined_lookalikes = ["장원영", "김아중", "고유리"]
                elif uid == "Y4qA1bFjd9a1Equ9kqL3Olawalt1":
                    predefined_lookalikes = ["김아영", "장원영", "김이서"]
                elif uid == "u3qA1bFjd9a1Equ9kqL3Olawalt1":
                    predefined_lookalikes = ["유정", "강민경", "신혜선"]
                elif uid == "u4qA1bFjd9a1Equ9kqL3Olawalt1":
                    predefined_lookalikes = ["쵸단", "해린", "장원영"]
                    
                if common_favorites_images or common_lookalikes_images:
                    similarity_score = 100.0

                    match_candidates.append({
                        "uid": other_user.get("id", uid),
                        "similarity": round(similarity_score, 2),
                        "common_favorites_images": list(common_favorites_images),
                        "common_lookalikes_images": predefined_lookalikes  # 지정된 더미 데이터 사용
                    })

        if not match_candidates:
            return {"uid": request.uid, "match_candidates": "매칭 후보가 없습니다."}

        return {
            "uid": request.uid,
            "match_candidates": sorted(match_candidates, key=lambda x: x["similarity"], reverse=True)
        }

    except Exception as e:
        print(f"매칭 후보 찾기 중 에러 발생: {e}")
        raise HTTPException(status_code=500, detail=f"매칭 후보 검색 실패: {str(e)}")


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000)


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000)