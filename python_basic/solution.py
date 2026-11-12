import pandas as pd
import numpy as np
import copy

def error_string_builder(err_str, to_add):
    if err_str:
        err_str += ' / '
    err_str += to_add
    return err_str

def data_load_and_check(file_name: str):
    try:
        df_house = pd.read_csv(file_name)
        print('first row:', df_house.iloc[0])
        print('total df length:', len(df_house))
        print('first row price dtype:', type(df_house['price'].iloc[0]))
        return df_house   
    except FileNotFoundError:
        print('파일명과 업로드 여부를 확인하세요.')
        return -1


TYPE_MAPPING = {
    'ID': 'int',
    'size': 'int',
    'year': 'int',
    'price': 'float'
}

def prob2(df: pd.DataFrame) -> pd.DataFrame:
    houses = []
    exc = 0
    for idx, row in df.iterrows():
        na_pos = row.isna()
        na = na_pos.sum()
        err_str = ''
        if na:
            err_str = error_string_builder(err_str, f"NA: {row.columns[na_pos]}")
        
        not_convertable = []
        converted = {}
        for col in TYPE_MAPPING:
            try:
                if col in ['ID', 'size', 'year', 'price']:
                    converted[col] = int(row[col])
                elif col in ['price']:
                    converted[col] = float(row[col])
            except:
                not_convertable.append(col)
        
        if not_convertable:
            err_str = error_string_builder(err_str, f"Not conv: {not_convertable}")
            
        if err_str:
            print(f'[ERROR](row_id: {idx}) {err_str}')
            exc += 1
            continue
        
        houses.append(dict(converted))
    
    print('Loaded rows:', len(houses))
    print('Excluded rows:', exc)
    
    return houses

COLUMNS = ['ID', 'size', 'year', 'price']

def prob3(houses_dict: dict):
    houses_test = copy.copy(houses_dict)
    houses_test += [
        {"ID": 101, "year": 5, "price": 3},  # size 키 없음
        {"ID": 102, "size": 35, "year": 5, "price": None},  # 값 없음
        {"ID": 103, "size": "오류", "year": 5, "price": 3},  # 숫자로 변환할 수 없음
        {"ID": 104, "size": 35, "year": 5, "price": 3}  
    ]

    selected = []
    for house in houses_test:
        # preprocess
        no_key = []
        no_item = []
        not_convertable = []
        for col in COLUMNS:
            if house.get(col, -1) == -1:
                no_key.append(col)
            elif not house[col] or pd.isna(house[col]):
                no_item.append(col)
            
            try:
                int(house[col])
            except:
                not_convertable.append(col)
        err_str = ''
        if no_key:
            err_str = error_string_builder(err_str, f'No keys: {no_key}') 
        if no_item:
            err_str = error_string_builder(err_str, f'No items: {no_item}')
        if not_convertable:
            err_str = error_string_builder(err_str, f'Not convertable: {not_convertable}')
        
        if err_str:
            print(f'[ERROR](row_id: {house["ID"]}) {err_str}')
            continue
        
        # search
        if house['size'] <= 0 or house['price'] < 0:
            continue
        
        if house['size'] >= 30 and house['price'] <= 4:
            selected.append(house)
    
    print('selected houses:', end='')
    for i, house in enumerate(selected):
        print(house['ID'], end=', ' if i < len(selected)-1 else '\n')
    print('number of selected houses:', len(selected))


def unit_price(price, size):
    try:
        price = int(price)
        size = int(size)
        if price <= 0:
            print('오류: 가격이 0 이하입니다.')
            return None
        if size <= 0:
            print('오류: 크기가 0 이하입니다.')
            return None
    except ValueError:
        print('값 오류: 숫자로 변환할 수 없는 문자열 입니다.')
        return None
    except TypeError:
        print('자료형 오류: 숫자 또는 숫자 형태의 문자열을 입력하세요.')
        return None

    return price / size


def prob4(houses):
    # sub problem 1
    for house in houses:
        print('house ID:', house['ID'], ' / per unit price:', unit_price(house['price'], house['size']))
    
    # sub problem 2
    print('unit_price(4, 40) =', unit_price(4, 40))
    print('unit_price("4", "40") =', unit_price("4", "40"))
    print('unit_price(4, 0) =', unit_price(4, 0))
    print('unit_price(-4, 40) =', unit_price(-4, 40))
    print('unit_price("오류", 40) =', unit_price("오류", 40))
    print('unit_price(None, 40) =', unit_price(None, 40))
    print('unit_price(6, 30) =', unit_price(6, 30))

def prob5(houses):
    ids = [] # 원래 순서 보존을 위한 리스트, 집합 사용시 순서 보존 불가능
    for house in houses:
        ids.append(house['ID'])
    print('IDs(order):', ids)

    years = set({}) # 중복 없이 추가를 위한 집합, 리스트 사용시 중복 모두 허용
    for house in houses:
        years.add(house['year'])
    print('Years(no-dup):', years)

    dict_house = {} # 집 정보의 ID 기반 조회, 리스트 사용시 ID가 항상 완전히 채워진 형태만 허용
    for house in houses:
        h = copy.copy(house)
        del h['ID']
        dict_house[house['ID']] = h
    print('Price of ID 3:', dict_house[3]['price'])

    searched_houses = [] # 하나의 묶음으로 저장, 생성 뒤 내부 항목을 변경 불가능하려면 튜플
    for house in houses:
        if house['size'] >= 30 and house['price'] <= 4:
            searched_houses.append(house)
    
    searched_houses = tuple(searched_houses) 
    print('Searched house tuples: ', searched_houses)


def main():
    print("========= Problem 1 =========")
    print('정상 처리 케이스')
    df_house = data_load_and_check('./1.01.house.price.csv')
    print()
    print('오류 처리 케이스')
    data_load_and_check('./null.csv')

    print("========= Problem 2 =========")
    houses = prob2(df_house)

    print("========= Problem 3 =========")
    prob3(houses)

    print("========= Problem 4 =========")
    prob4(houses)

    print("========= Problem 5 =========")
    prob5(houses)



if __name__ == "__main__":
    main()