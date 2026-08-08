# edge_models/ — 웹 B 인수인계 묶음

> 홍준기 → 박성찬 · 2026-07-31 · Phase 2 산출물

`model/quantization.py` 가 만든 배포용 파일 4개. **네 개를 모두 가져가야 한다.**

| 파일 | 용도 |
| --- | --- |
| `frost_cnn_int8.tflite` | 비전 추론 (1.2 MB, float32 대비 **86 % 감소**) |
| `frost_cnn_int8_edge.json` | **필수** — 입출력 int8 스케일 규격 |
| `frost_xgb.json` | 수치 추론 |
| `frost_xgb_meta.json` | 특징 순서 |

---

## 1. CNN — INT8 이라 스케일 변환이 필요하다

입출력이 float 가 아니라 **int8** 이다. 변환식을 안 쓰면 결과가 조용히 틀어진다.

```python
import numpy as np
from tflite_runtime.interpreter import Interpreter

itp = Interpreter(model_path='frost_cnn_int8.tflite')
itp.allocate_tensors()
inp, out = itp.get_input_details()[0], itp.get_output_details()[0]

# 전처리: 224×224 리사이즈, 0~255 float (crop 없음 — 전체 이미지 모델)
x = resize(img, (224, 224)).astype(np.float32)

s, z = inp['quantization']                       # scale 1.0, zero_point -128
q = np.clip(np.round(x / s) + z, -128, 127).astype(np.int8)[None]
itp.set_tensor(inp['index'], q)
itp.invoke()

so, zo = out['quantization']                     # scale 0.00390625, zero_point -128
prob = (itp.get_tensor(out['index'])[0].astype(np.float32) - zo) * so
```

`prob` 는 길이 4 (`[정상, 초기, 중기, 임계]`).

> ⚠️ **XNNPACK 델리게이트 준비에 실패하는 CPU 가 있다**
> (`RuntimeError: failed to create XNNPACK runtime`). 라즈베리파이 모사 환경에서 나면
> 델리게이트 없이 생성할 것:
> ```python
> Interpreter(model_path=..., experimental_op_resolver_type=OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES)
> ```

## 2. ★ 제상 트리거는 argmax 가 아니라 "중기 이상 확률의 합"

CNN 의 **임계 단계 재현율이 17 %** 다 — 임계를 중기로 한 단계 낮춰 보는 경향이 강하다.
`argmax` 로 판정하면 정작 위험한 순간을 놓친다.

```python
need_defrost = (prob[2] + prob[3]) > 0.5        # 중기 + 임계
```

이 기준이면 "제상이 필요한 상황"의 **84 %** 를 잡아낸다(argmax 는 훨씬 낮음).
`frost_stage` 컬럼에는 `argmax` 를 그대로 저장하되, **릴레이 토글 판단은 위 합산 확률**로 할 것.

## 3. XGBoost — 정상·초기 전문가로 쓸 것

특징 순서(반드시 이 순서):

```python
['outdoor_temp', 'outdoor_humidity', 'compressor_power', 'evaporator_temp', 'current']
```

**중기와 임계는 XGBoost 로 구분할 수 없다.** 증발기 출구가 습증기가 되는 순간
온도가 포화온도(−26.33 ℃)에 고정돼 두 단계가 물리적으로 동일한 값을 갖기 때문이다.
4단계 검증 정확도 68.8 % 는 이 한계 때문이며, **정상·초기 구간은 94 %** 로 정확하다.

→ 앙상블 시 **정상·초기는 XGBoost, 중기·임계는 CNN** 에 가중을 두는 게 물리적으로 맞다.

> ⚠️ **버전 확인 필요.** 이 모델은 xgboost 3.2.0 으로 저장했다. 웹 B 가 2.x 면 로드가
> 실패할 수 있다. 실패하면 웹 B 환경에서 `model/train_xgb.py` 를 다시 돌려 재생성할 것
> (수 초면 끝난다).

---

## 미완 — 다음 세션

`quantization.py` 의 3단계(변환 전후 정확도·추론시간 비교표)가 아직 안 돌았다.
GPU 서버에서 아래를 실행하면 `out/quantization_report.json` 이 채워지고,
그 표가 그대로 **P4 성과 자료**가 된다.

```bash
cd model && python quantization.py --deploy ../edge_models
```
