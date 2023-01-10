# [DenseCRF] ReadMe
<!--Written by Joel J. Park on 2023JAN04-->

## Dependency
`pydensecrf`가 C++ 기반 라이브러리임에 따라 `Cython` 등이 필요
- [gcc 이슈 참고 링크](https://stackoverflow.com/questions/11912878/gcc-error-gcc-error-trying-to-exec-cc1-execvp-no-such-file-or-directory)

### Libraries
- Numpy
- scikit-image
- Pandas
- tqdm
- Cython
- [pydensecrf](https://github.com/lucasb-eyer/pydensecrf)

```shell
pip install cython numpy scikit-image pandas tqdm git+https://github.com/lucasb-eyer/pydensecrf.git
```

## 사용법
- `dcrf.py` 파일 내 `DIR_PATH`, `FILE_NAME`(확장자 제외), `DATASET_PATH` 변수 설정 후 실행 <***가급적 절대경로 사용 요망***>
- 실행 결과로 `DIR_PATH`에 `FILE_NAME` + `_crf.csv` 파일이 생성됨


## 참고자료
- [CRF 적용 방법 @Kaggle](https://www.kaggle.com/code/meaninglesslives/apply-crf-unet-resnet)
- [DenseCRF2D() 사용 예시](https://www.programcreek.com/python/example/106424/pydensecrf.densecrf.DenseCRF2D)
- [논문 리뷰 및 코드 구현 by wsshin](https://wsshin.tistory.com/8)