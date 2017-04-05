from bs4 import BeautifulSoup
from selenium import webdriver
from time import *
def main():
    driver = webdriver.Chrome("C:/Users/KA/PycharmProjects/chromedriver.exe")
    driver.get("http://mlb.mbcsportsplus.com/stats/sortable_team.php?t=hitting")
    final = []
    def spide():
        page = driver.page_source
        soup = BeautifulSoup(page, 'html.parser')
        data = []
        table_body = soup.find('tbody')
        rows = table_body.find_all('tr')
        for row in rows:
            cols = row.find_all('td')
            cols = [ele.text.strip() for ele in cols]
            data.append([ele for ele in cols if ele])
        for i in data[1:]:
            final.append(i)
    def save():
        with open('Train_major.csv', "w") as wf:
            for i in final[:1600]:
                for j in i:
                    wf.write(j + ',')
                wf.write("\n")
        with open('Test_major.csv',"w") as wf:
            for i in final[1601:]:
                for j in i:
                    wf.write(j + ',')
                wf.write("\n")
    def season(n):
        a= str(n)
        sear = driver.find_element_by_id("sp_hitting_season")
        webdriver.ActionChains(driver).move_to_element(sear).click(sear).perform()
        sear = driver.find_element_by_id('sp_hitting_season')
        for option in sear.find_elements_by_tag_name('option'):
            if option.text == a:
                option.click()
                break
    def circuit():
        for i in range(1950,2017):
            season(i)
            sleep(2.5)
            spide()
    circuit()
    save()
main()
