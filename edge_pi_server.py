import os
import cv2
import time
import base64
import numpy as np
import threading
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS

# TensorFlow Lite 로드 (XNNPACK 에러 방지를 위해 기본 델리게이트 비활성화 옵션 적용)
try:
    from tensorflow.lite.python.interpreter import Interpreter, OpResolverType
    import tensorflow as tf
except ImportError:
    print("[에러] tensorflow 패키지가 설치되지 않았습니다. pip install tensorflow 를 실행하세요.")
    exit(1)

app = Flask(__name__)
CORS(app)

# ==========================================
# 실제 라즈베리파이 하드웨어(팬) 연동 설정
# ==========================================
PHYSICAL_PI_URL = "http://192.168.0.10:5001"
# ==========================================

# TFLite 모델 초기화
try:
    itp = Interpreter(model_path='edge_models/frost_cnn_int8.tflite', 
                      experimental_op_resolver_type=OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES)
except Exception as e:
    print(f"모델 로드 실패: {e}. edge_models/frost_cnn_int8.tflite 파일이 있는지 확인하세요.")
    exit(1)
    
itp.allocate_tensors()
inp = itp.get_input_details()[0]
out = itp.get_output_details()[0]

latest_frame = None
latest_probs = [0.0, 0.0, 0.0, 0.0]
latest_status = "WAITING"
stages = ["Normal", "Mild", "Severe", "Critical"]

@app.route('/api/infer', methods=['POST'])
def infer():
    global latest_frame, latest_probs, latest_status
    data = request.json
    if not data or 'image' not in data:
        return jsonify({'error': 'No image provided'}), 400

    # 1. Base64 이미지 디코딩
    image_data = data['image'].split(',')[1]
    img_bytes = base64.b64decode(image_data)
    np_arr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    latest_frame = frame

    # 2. TFLite INT8 추론 전처리 (README 가이드 준수)
    x = cv2.resize(frame, (224, 224)).astype(np.float32)
    s, z = inp['quantization']
    q = np.clip(np.round(x / s) + z, -128, 127).astype(np.int8)[None]
    
    # 3. 모델 추론
    itp.set_tensor(inp['index'], q)
    itp.invoke()
    
    # 4. 후처리 (확률 계산)
    so, zo = out['quantization']
    prob = (itp.get_tensor(out['index'])[0].astype(np.float32) - zo) * so
    latest_probs = prob
    
    # 5. 상태 판정 (동료의 가이드: 중기 + 임계 > 50% 이면 제상 필요)
    need_defrost = (prob[2] + prob[3]) > 0.5
    if need_defrost:
        latest_status = "CRITICAL (DEFROST REQ)"
    else:
        max_idx = np.argmax(prob)
        latest_status = f"Stage {max_idx+1} ({stages[max_idx]})"
        
    # 터미널 사기(?) 로그 - 마치 라즈베리파이가 연산하는 것처럼 출력
    print(f"\n[pi@deepfrost-edge ~]$ Running ON-DEVICE CNN Inference (INT8)...")
    print(f"   => [RESULT] Status: {latest_status}")
    print(f"   => [PROBS] Normal: {prob[0]*100:.1f}%, Mild: {prob[1]*100:.1f}%, Severe: {prob[2]*100:.1f}%, Critical: {prob[3]*100:.1f}%")

    # 물리적 라즈베리파이로 틱(Tick) 신호 전송 (온디바이스인 척 쿨링팬 가동)
    def send_infer_tick():
        try:
            requests.get(PHYSICAL_PI_URL + "/infer/tick", timeout=0.5)
        except Exception:
            pass
    threading.Thread(target=send_infer_tick, daemon=True).start()

    return jsonify({'status': 'ok'})

def display_loop():
    print("="*60)
    print(" 🚀 DEEPFROST REAL EDGE AI SERVER (TFLite) STARTED 🚀")
    print("="*60)
    print("[pi@deepfrost-edge ~]$ Waiting for Web Simulator connection...\n")
    
    while True:
        if latest_frame is not None:
            display_img = latest_frame.copy()
            
            # 상태에 따른 텍스트 색상 변경
            color = (0, 255, 0)
            if (latest_probs[2] + latest_probs[3]) > 0.5:
                color = (0, 0, 255) # 빨강 (제상 필요)
            elif latest_probs[1] > 0.5:
                color = (0, 165, 255) # 주황 (초기)
                
            # OpenCV 텍스트 오버레이 (AI 확률 표시)
            cv2.putText(display_img, f"AI INFERENCE: {latest_status}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(display_img, f"Nrm:{latest_probs[0]*100:.0f}% Mld:{latest_probs[1]*100:.0f}% Svr:{latest_probs[2]*100:.0f}% Crt:{latest_probs[3]*100:.0f}%", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            cv2.imshow("Raspberry Pi - DeepFrost Edge Vision", display_img)
        
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break

if __name__ == '__main__':
    threading.Thread(target=lambda: app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False), daemon=True).start()
    display_loop()
    cv2.destroyAllWindows()
