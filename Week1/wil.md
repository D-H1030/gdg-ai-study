# gdg 1주차 과제 
## 실습 과정
 일단 venv 가상환경을 설치하기 위해

 `python -m venv venv`

 명령어를 입력했다. 

 그 후,

 `./venv/scripts/activate`

 를 입력해 가상환경을 실행하려 했는데 관리자 권한 변경이 필요해서 powershell 에 들어가 

 `Set-ExecutionPolicy RemoteSigned -Scope CurrentUser`

 명령어를 입력했다. 그랬더니 vs code 에서 venv 가상환경이 실행되었다.

 그 후,

 `pip install torch torchvision matplotlib numpy`

명령어를 통해 필요한 모듈을 설치했다. 

그리고 파일을 실행시켜 이미지 분류 결과를 확인했다. 


-------------

</br>

## KNN 
KNN 방식은 새로운 데이터가 들어왔을 때, 이 값을 이전의 분류 중 하나로 분류하는 방식이다. 

새로운 데이터가 들어왔을 때, 이미 알고 있는 데이터들 중 가장 가까운 k개의 이웃을 찾은 후 그 이웃들 중 다수가 속한 클래스로 새 데이터를 분류한다. 

만약 k=1 이면 새 데이터와 가장 가까운 기존의 데이터가 속해 있는 클래스로 새 데이터를 분류하는 것이고, k=5 이면 새 데이터와 가까운 기존의 데이터를 순서대로 5개 골라 그 중 다수가 속한 클래스로 새 데이터를 분류한다. 