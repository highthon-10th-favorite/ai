import os
import io
import cv2
import uvicorn
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from insightface.app import FaceAnalysis
from annoy import AnnoyIndex

app = FastAPI()

EMBEDDINGS_FILE = "celebrity_embeddings.npy"
NAMES_FILE = "celebrity_names.txt"


face_app = FaceAnalysis(allowed_modules=['detection', 'recognition'])
face_app.prepare(ctx_id=0, det_size=(640, 640))
print("FaceAnalysis 모델 로드 완료")

def _crop_with_padding(image, bbox, padding=10):
    """
    바운딩 박스 기반으로 얼굴 영역을 크롭할 때, 패딩을 추가합니다.
    """
    h, w = image.shape[:2]
    x1 = max(bbox[0] - padding, 0)
    y1 = max(bbox[1] - padding, 0)
    x2 = min(bbox[2] + padding, w)
    y2 = min(bbox[3] + padding, h)
    return image[y1:y2, x1:x2]

def adjust_brightness(image, factor=1.0):
    """
    이미지 밝기 조절 (factor=1.0이면 변화 없음)
    """
    return cv2.convertScaleAbs(image, alpha=factor, beta=0)

def rotate_image(image, angle):
    """
    이미지를 중심으로 지정한 각도(angle, degree)만큼 회전시킵니다.
    """
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated

def preprocess_face(face_img):
    """
    얼굴 이미지를 YCrCb로 변환하여 Y 채널 히스토그램 평활화 후 다시 BGR로 변환합니다.
    """
    ycrcb = cv2.cvtColor(face_img, cv2.COLOR_BGR2YCrCb)
    channels = list(cv2.split(ycrcb))
    channels[0] = cv2.equalizeHist(channels[0])
    ycrcb = cv2.merge(channels)
    equalized = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
    return equalized

def extract_embedding_from_face(face_img):
    """
    주어진 얼굴 이미지(face_img)에서 임베딩을 추출합니다.
    """
    faces = face_app.get(face_img)
    if faces:
        return faces[0].normed_embedding
    return None

def get_augmented_embeddings(face_img):
    """
    얼굴 이미지에 대해 여러 증강을 적용한 후 임베딩 목록을 반환합니다.
    증강: 원본, 좌우 반전, 밝기 조절(약간 증가/감소), ±5도 회전
    """
    embeddings = []

    proc_img = preprocess_face(face_img)

    augmentations = [
        ("원본", lambda img: img),
        ("좌우 반전", lambda img: cv2.flip(img, 1)),
        ("밝기 증가", lambda img: adjust_brightness(img, factor=1.1)),
        ("밝기 감소", lambda img: adjust_brightness(img, factor=0.9)),
        ("회전 +5도", lambda img: rotate_image(img, 5)),
        ("회전 -5도", lambda img: rotate_image(img, -5))
    ]

    for name, aug_fn in augmentations:
        aug_img = aug_fn(proc_img)
        emb = extract_embedding_from_face(aug_img)
        if emb is not None:
            embeddings.append(emb)
        else:
            print(f"DEBUG: 증강({name})에서 얼굴 임베딩 추출 실패.")

    return embeddings

def extract_aligned_embedding(image):
    """
    이미지에서 얼굴을 검출한 후,
    - 검출된 얼굴 중 신뢰도가 가장 높은 얼굴 선택
    - 정렬된 얼굴(aligned)이 없으면, 랜드마크(kps 또는 landmark)를 활용해 정렬 수행
    - 정렬에 실패하면 bbox 기반 크롭을 사용
    - 다양한 증강을 적용해 임베딩을 추출 후 평균을 계산하여 최종 임베딩 생성
    - 만약 증강된 임베딩 추출에 실패하면 원본 얼굴 임베딩을 fallback으로 사용
    """
    faces = face_app.get(image)
    if not faces:
        print("DEBUG: 입력 이미지에서 얼굴이 검출되지 않았습니다.")
        return None

    face = max(faces, key=lambda x: getattr(x, "det_score", 0))

    if hasattr(face, "aligned") and face.aligned is not None:
        face_img = face.aligned
    else:
        landmarks = None
        if hasattr(face, "landmark") and face.landmark is not None:
            landmarks = np.array(face.landmark).astype(np.float32)
        elif hasattr(face, "kps") and face.kps is not None:
            landmarks = np.array(face.kps).astype(np.float32)
        
        if landmarks is not None:
            std_landmarks = np.array([
                [38.2946, 51.6963],
                [73.5318, 51.5014],
                [56.0252, 71.7366],
                [41.5493, 92.3655],
                [70.7299, 92.2041]
            ], dtype=np.float32)
            output_size = (112, 112)
            M, _ = cv2.estimateAffinePartial2D(landmarks, std_landmarks, method=cv2.LMEDS)
            if M is not None:
                M = M.astype(np.float32)
                aligned_face = cv2.warpAffine(image, M, output_size)
                face_img = aligned_face
            else:
                print("DEBUG: 정렬 변환 행렬(M) 계산 실패. bbox 기준 크롭 진행.")
                bbox = face.bbox.astype(int)
                face_img = _crop_with_padding(image, bbox)
        else:
            print("DEBUG: 얼굴 랜드마크(kps/landmark) 미발견. bbox 기준 크롭 진행.")
            bbox = face.bbox.astype(int)
            face_img = _crop_with_padding(image, bbox)
    
    augmented_embeddings = get_augmented_embeddings(face_img)
    
    if not augmented_embeddings:
        print("DEBUG: 증강 임베딩 추출 실패. 원본 얼굴 임베딩 사용.")
        return face.normed_embedding

    final_embedding = np.mean(np.array(augmented_embeddings), axis=0)
    final_embedding = final_embedding / np.linalg.norm(final_embedding)
    return final_embedding

celebrity_embeddings = None
celebrity_names = []

dataset_dir = "images"
if not os.path.exists(dataset_dir):
    raise Exception(f"데이터셋 디렉토리 {dataset_dir}가 존재하지 않습니다.")

if os.path.exists(EMBEDDINGS_FILE) and os.path.exists(NAMES_FILE):
    print("저장된 임베딩 파일 로드 중...")
    celebrity_embeddings = np.load(EMBEDDINGS_FILE)
    with open(NAMES_FILE, "r", encoding="utf-8") as f:
        celebrity_names = [line.strip() for line in f if line.strip()]
    print(f"총 {celebrity_embeddings.shape[0]}명의 연예인 임베딩 로드 완료.")
else:
    print("임베딩 파일이 없으므로, 연예인 임베딩 생성 시작...")
    embeddings_list = []
    for filename in os.listdir(dataset_dir):
        if not filename.endswith("_profile.jpg"):
            continue
        file_path = os.path.join(dataset_dir, filename)
        celeb_name = filename.replace("_profile.jpg", "")
    
        image = cv2.imread(file_path)
        if image is None:
            print(f"[WARN] {file_path} 이미지를 읽을 수 없습니다. 건너뜁니다.")
            continue
    
        embedding = extract_aligned_embedding(image)
        if embedding is None:
            print(f"[WARN] {celeb_name}의 얼굴 임베딩 추출 실패. 건너뜁니다.")
            continue

        embeddings_list.append(embedding.astype('float32'))
        celebrity_names.append(celeb_name)
        print(f"[INFO] {celeb_name} 임베딩 생성 완료.")
    
    if not embeddings_list:
        raise Exception("연예인 임베딩 생성에 실패하였습니다.")
    
    celebrity_embeddings = np.stack(embeddings_list)
    num_celebrities = celebrity_embeddings.shape[0]
    print(f"총 {num_celebrities}명의 연예인 임베딩이 생성되었습니다.")
    
    np.save(EMBEDDINGS_FILE, celebrity_embeddings)
    with open(NAMES_FILE, "w", encoding="utf-8") as f:
        for name in celebrity_names:
            f.write(name + "\n")
    print("임베딩 파일 저장 완료.")

embedding_dim = celebrity_embeddings.shape[1]
annoy_index = AnnoyIndex(embedding_dim, metric='angular')
for i, vec in enumerate(celebrity_embeddings):
    annoy_index.add_item(i, vec.tolist())
num_trees = 50
annoy_index.build(num_trees)
print("Annoy 인덱스 생성 완료")

def read_imagefile(file_bytes: bytes) -> np.ndarray:
    image = np.array(Image.open(io.BytesIO(file_bytes)))
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image

##############################
# 5. FastAPI 엔드포인트
##############################
@app.get("/")
async def root():
    return {"message": "Welcome to the Celebrity Matching API!"}

@app.post("/upload-photo")
async def upload_photo(file: UploadFile = File(...)):
    contents = await file.read()
    try:
        image = read_imagefile(contents)
    except Exception as e:
        print("DEBUG: 이미지 파일 읽기 실패:", e)
        raise HTTPException(status_code=400, detail="이미지 파일 읽기 실패")
    
    user_embedding = extract_aligned_embedding(image)
    if user_embedding is None:
        raise HTTPException(status_code=400, detail="업로드된 이미지에서 얼굴 임베딩을 추출할 수 없습니다.")
    
    k = 10
    candidate_indices = annoy_index.get_nns_by_vector(user_embedding.tolist(), k, include_distances=False)
    
    results = []
    for idx in candidate_indices:
        candidate_embedding = celebrity_embeddings[idx]
        cosine_sim = float(np.dot(user_embedding, candidate_embedding))
        sim_percent = round(cosine_sim * 100, 2)
        celeb_name = celebrity_names[idx] if idx < len(celebrity_names) else "Unknown"
        results.append({"celebrity": celeb_name, "similarity": sim_percent})
    
    results = sorted(results, key=lambda x: x["similarity"], reverse=True)
    
    return {"top10": results}

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
