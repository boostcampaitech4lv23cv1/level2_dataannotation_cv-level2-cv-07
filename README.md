# level2_dataannotation_cv-level2-cv-07

## 팀원 
<table>
    <th colspan=5>블랙박스</th>
    <tr height="160px">
        <td align="center" width="150px">
            <a href="https://github.com/kimk-ki"><img height="120px" width="120px" src="https://avatars.githubusercontent.com/u/110472164?v=4"/></a>
            <br />
            <a href="https://github.com/kimk-ki"><strong>🙈 김기용</strong></a>
            <br />
        </td>
        <td align="center" width="150px">
            <a href="https://github.com/SeongSuKim95"><img height="120px" width="120px" src="https://avatars.githubusercontent.com/u/62092317?v=4"/></a>
            <br/>
            <a href="https://github.com/SeongSuKim95"><strong>🐒 김성수</strong></a>
            <br />
        </td>
        <td align="center" width="150px">
            <a href="https://github.com/juye-ops"><img height="120px" width="120px" src="https://avatars.githubusercontent.com/u/103459155?v=4"/></a>
            <br/>
            <a href="https://github.com/juye-ops"><strong>🙉 김주엽</strong></a>
            <br />
        </td>
        <td align="center" width="150px">
            <a href="https://github.com/99sphere"><img height="120px" width="120px" src="https://avatars.githubusercontent.com/u/59161083?v=4"/></a>
            <br />
            <a href="https://github.com/99sphere"><strong>🙊 이  구</strong></a>
            <br />
        </td>
        <td align="center" width="150px">
            <a href="https://github.com/thlee00"><img height="120px" width="120px" src="https://avatars.githubusercontent.com/u/56151577?v=4"/></a>
            <br/>
            <a href="https://github.com/thlee00"><strong>🐵 이태희</strong></a>
            <br />
        </td>
    </tr>
</table>

- 김기용_T4020 : 학습 돌려보기
- 김성수_T4039 : 기업 연계 프로젝트 제안서 작성 리딩
- 김주엽_T4048 : Augmentation 적용 후 학습 진행, Git review
- 이    구_T4145 : 데이터 전처리, 실험 초반 setting (train/val loop, random seed, logging 등)
- 이태희_T4172 :  train/val dataset split, WandB Sweep 적용

## 프로젝트 개요
![image](https://user-images.githubusercontent.com/56151577/208023624-587455d7-c669-4bc0-a8b8-68ac51206c8e.png)

> 스마트폰으로 카드를 결제하거나, 카메라로 카드를 인식할 경우 자동으로 카드 번호가 입력되는 경우가 있다. 또 주차장에 들어가면 차량 번호가 자동으로 인식되는 경우도 흔히 있다. 이처럼 OCR (Optimal Character Recognition) 기술은 사람이 직접 쓰거나 이미지 속에 있는 문자를 얻은 다음 이를 컴퓨터가 인식할 수 있도록 하는 기술로, 컴퓨터 비전 분야에서 현재 널리 쓰이는 대표적인 기술 중 하나이다.
> OCR task는 글자 검출 (text detection), 글자 인식 (text recognition), 정렬기 (Serializer) 등의 모듈로 이루어져 있으며, 본 대회에서는 '글자 검출' task 만을 해결한다.

## Dataset
- **Input** : 글자가 포함된 전체 이미지
- **Output :** bbox 좌표가 포함된 UFO(Upstage  Format for OCR) 형식

## 프로젝트 환경
모든 실험은 아래의 환경에서 진행되었다.

- Ubuntu 18.04.5 LTS   
- Intel(R) Xeon(R) Gold 5120 CPU @ 2.20GHz   
- NVIDIA Tesla V100-SXM2-32GB   

## 프로젝트 수행 일정
<img width="1512" alt="Untitled (2)" src="https://user-images.githubusercontent.com/56151577/208024122-cb7d5507-8c1f-4939-9814-65d7d7472e98.png">

- 기업 연계 프로젝트 제안서 작성
    - 프로젝트 파악 및 연구 동향 조사(12.5 ~ 12.7)
    - 아이디어 구체화(12.8 ~ 12.9)
    - 퇴고 및 수정(12.10 ~ 12.11)
- OCR 대회
    - Course & baseline 코드 분석(12.5 ~ 12.11)
        - 강의 수강 및 코드 오류 해결
    - EDA (12.12 ~ 12.13)
        - Train Dataset(ICDAR17_Korean, Upstage dataset) 에 대한 분석 및 전처리 진행
    - Model Training (12.14 ~ 12.15)
        - EAST 모델 학습

## Git-Flow
![Untitled (3)](https://user-images.githubusercontent.com/56151577/208024180-7d0c3074-312c-4a86-a2dc-c413f75e2758.png)

## Wrap-Up Report
[![image](https://user-images.githubusercontent.com/62556539/200262300-3765b3e4-0050-4760-b008-f218d079a770.png)](https://www.notion.so/Wrap-Up-c40db17022f848c89d41551a246985f2)

## Result
- Rank : 11/19


|  | f1 score (↑) | recall (↑) | precision (↑) |
| --- | --- | --- | --- |
| Public Leaderboard | 0.6487 | 0.5435 | 0.8046 |
| Private Leaderboard | 0.6478 | 0.5449 | 0.7988 |
