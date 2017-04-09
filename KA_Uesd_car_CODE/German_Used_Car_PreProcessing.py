import pandas as pd                                  #pandas 불러오기(null이 포함된 인스턴스 지우는데 필요함)
a = pd.read_csv('autos.csv', na_values='Null')       #na_values : 비어있는 데이터를 Null데이터로 처리함
a=a.dropna(axis = 0)                                  #dropna : null값 있는 인스턴스 지우기
                                                      #axis = 0 이면 column, 1 이면 row
a.to_csv("autos_drop_NULL.csv", index = False )

print("1차완료")

with open("autos_drop_NULL.csv", "r", encoding = 'utf-8') as rf:
    a = [i.split(',') for i in rf.readlines()]
print("2차완료")
seller = list(set([a[j][0] for j in range(1,len(a))]))
offertype = list(set([a[j][1] for j in range(1,len(a))]))
abtest =list(set([a[j][2] for j in range(1,len(a))]))
vehicleType = list(set([a[j][3] for j in range(1,len(a))]))
gearbox =list(set([a[j][4] for j in range(1,len(a))]))
model=list(set([a[j][5] for j in range(1,len(a))]))
fuelType=list(set([a[j][6] for j in range(1,len(a))]))
brand=list(set([a[j][7] for j in range(1,len(a))]))
notRepairedDamage= list(set([a[j][8] for j in range(1,len(a))]))

kjy = [seller,offertype,abtest,vehicleType,gearbox,model,fuelType,brand,notRepairedDamage]
print("3차 완료")
#이부분은  labeling 이 필요한 row들을 identifying하는 부분이고, 범주형 변수 종류들로 이뤄진 list를 만듦
#kjy는 나중에 dataframe의 값들을 바꿀 때 index역할을 한다
#li = [list(set([a[j][k] for j in range(1,len(a))])) for k in range(7)]
#일반화된 경우. 우리가 만든 코드는 feature 하나마다 리스트를 하나씩 만들고 합쳤는데,
#어떤 feature가 labeling이 필요한지 컴퓨터가 identify할 수 있는 경우, li 리스트처럼 list comprehension으로
#만들고 아래 for문을 돌리면 훨씬 간단하다. 또 어떤 데이터에 대해서도 적용 가능
for l in kjy:
    for j in range(len(a)):
        for k in l:
            if a[j][kjy.index(l)] == k:
                a[j][kjy.index(l)] = int(l.index(k))
print("4차완료")
#이거 각각 요소를 kjy의 label값이랑 비교해서 그 인덱스값으로 바꿔주는 과정
a = pd.DataFrame(a)
print("5차완료")
a.to_csv("autos_Indexing.csv",index = False)
#여기서 csv로 다시써서 저장할 때 다시 pandas 사용
